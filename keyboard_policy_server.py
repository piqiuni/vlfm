"""
Keyboard-Controlled WebSocket Policy Server for BEHAVIOR Environment

This server allows you to control a 23-DOF robot through keyboard input.
Press keys to increment/decrement joint values, and the action persists until changed.

Usage:
    # Start server on default port (8000)
    python keyboard_policy_server.py
    
    # Custom port
    python keyboard_policy_server.py --port 9000

Then run the environment client:
    python -m omnigibson.examples.environments.behavior_env_web --host localhost --port 8000

Keyboard Controls:
    1, 2            - Decrement / increment the joint index to control (0-22)
    [, ]            - Move the selected joint backwards / forwards
    r               - Reset all joints to zero
    q               - Quit server
    h               - Show help
"""

import asyncio
import websockets
import msgpack
import numpy as np
import argparse
import logging
import functools
import sys
import termios
import tty
import select
from typing import Dict, Any, Optional
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("keyboard_policy_server")

# Reduce websockets logging to WARNING to avoid cluttering console
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('websockets.server').setLevel(logging.WARNING)


# NumPy array support for msgpack (same as network_utils.py)
def pack_array(obj):
    """Pack NumPy arrays for msgpack serialization."""
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    """Unpack NumPy arrays from msgpack."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


# Create packer/unpacker with NumPy support
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class KeyboardPolicy:
    """
    Keyboard-controlled policy that maintains action state.
    Actions persist until modified by keyboard input.
    """
    
    def __init__(self, action_dim: int = 23, action_delta: float = 0.1):
        """
        Initialize the keyboard policy.
        
        Args:
            action_dim (int): Dimension of action space (23 for R1Pro robot)
            action_delta (float): Amount to change joint value per keypress
        """
        self.action_dim = action_dim
        self.action_delta = action_delta
        self.step_count = 0
        
        # Current action state (persists across steps)
        self.action = np.zeros(self.action_dim, dtype=np.float32)
        
        # Current joint being controlled
        self.selected_joint = 0
        
        # Terminal settings for keyboard input
        self.old_settings = None
        
        logger.info(f"Initialized KeyboardPolicy with action_dim={action_dim}, delta={action_delta}")
        self.print_help()
        # Print initial status line
        print(f"[CTRL] Joint: {self.selected_joint:2d} | Value: {self.action[self.selected_joint]:+.3f} | Steps: {self.step_count:4d}" + " "*20, end='', flush=True)
    
    def print_help(self):
        """Print keyboard control instructions."""
        print("\n" + "="*60)
        print("Keyboard Control Instructions")
        print("="*60)
        print("  1, 2     - Decrement / increment joint index (current: {})".format(self.selected_joint))
        print("  [, ]     - Move joint backwards / forwards (delta: ±{})".format(self.action_delta))
        print("  r        - Reset all joints to zero")
        print("  h        - Show this help")
        print("  q        - Quit server")
        print("="*60)
        print(f"Current action: {self.action}")
        print(f"Selected joint: {self.selected_joint}")
        print("="*60 + "\n")
    
    def reset(self) -> None:
        """Reset policy state."""
        self.action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_count = 0
        self.selected_joint = 0
        print("\n[RESET] Environment reset signal received - all joints zeroed" + " "*20)
        print(f"[CTRL] Joint: {self.selected_joint:2d} | Value: {self.action[self.selected_joint]:+.3f} | Steps: {self.step_count:4d}" + " "*20, end='', flush=True)
    
    def update_action(self, key: str) -> bool:
        """
        Update action based on keyboard input.
        
        Args:
            key (str): Key pressed
            
        Returns:
            bool: True if should continue, False if should quit
        """
        if key == '1':
            # Decrement joint index
            self.selected_joint = (self.selected_joint - 1) % self.action_dim
            print(f"\r[CTRL] Joint: {self.selected_joint:2d} | Value: {self.action[self.selected_joint]:+.3f} | Steps: {self.step_count:4d}" + " "*20, end='', flush=True)
        
        elif key == '2':
            # Increment joint index
            self.selected_joint = (self.selected_joint + 1) % self.action_dim
            print(f"\r[CTRL] Joint: {self.selected_joint:2d} | Value: {self.action[self.selected_joint]:+.3f} | Steps: {self.step_count:4d}" + " "*20, end='', flush=True)
        
        elif key == '[':
            # Move joint backwards
            self.action[self.selected_joint] = np.clip(
                self.action[self.selected_joint] - self.action_delta,
                -1.0, 1.0
            )
            print(f"\r[CTRL] Joint: {self.selected_joint:2d} | Value: {self.action[self.selected_joint]:+.3f} | Steps: {self.step_count:4d}" + " "*20, end='', flush=True)
        
        elif key == ']':
            # Move joint forwards
            self.action[self.selected_joint] = np.clip(
                self.action[self.selected_joint] + self.action_delta,
                -1.0, 1.0
            )
            print(f"\r[CTRL] Joint: {self.selected_joint:2d} | Value: {self.action[self.selected_joint]:+.3f} | Steps: {self.step_count:4d}" + " "*20, end='', flush=True)
        
        elif key == 'r':
            # Reset all joints
            self.action = np.zeros(self.action_dim, dtype=np.float32)
            print("\n[RESET] All joints reset to zero" + " "*30)
            print(f"[CTRL] Joint: {self.selected_joint:2d} | Value: {self.action[self.selected_joint]:+.3f} | Steps: {self.step_count:4d}" + " "*20, end='', flush=True)
        
        elif key == 'h':
            # Show help
            self.print_help()
        
        elif key == 'q':
            # Quit
            print("\nQuitting...")
            return False
        
        
        return True
    
    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Return current action state (persists until modified).
        
        Args:
            obs (dict): Observation dictionary (not used in keyboard control)
        
        Returns:
            np.ndarray: Current action array of shape (action_dim,)
        """
        self.step_count += 1
        
        # Update status display every 50 steps (without logging)
        if self.step_count % 50 == 0:
            # Print status on same line without newline
            print(f"\r[CTRL] Joint: {self.selected_joint:2d} | Value: {self.action[self.selected_joint]:+.3f} | Steps: {self.step_count:4d}" + " "*20, end='', flush=True)
        
        return self.action.copy()


class KeyboardInputThread(threading.Thread):
    """Thread for non-blocking keyboard input."""
    
    def __init__(self, policy: KeyboardPolicy):
        super().__init__(daemon=True)
        self.policy = policy
        self.running = True
        self.old_settings = None
    
    def run(self):
        """Run keyboard input loop."""
        # Save terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode for single character input
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    # Update action based on key
                    if not self.policy.update_action(key):
                        self.running = False
                        break
        
        finally:
            # Restore terminal settings
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def stop(self):
        """Stop the keyboard input thread."""
        self.running = False


async def handle_client(websocket, policy: KeyboardPolicy):
    """
    Handle WebSocket client connection.
    
    Args:
        websocket: WebSocket connection object
        policy: Policy instance to generate actions
    """
    client_address = websocket.remote_address
    print(f"\n[CONN] Client connected: {client_address[0]}:{client_address[1]}" + " "*30)
    print(f"[CTRL] Joint: {policy.selected_joint:2d} | Value: {policy.action[policy.selected_joint]:+.3f} | Steps: {policy.step_count:4d}" + " "*20, end='', flush=True)
    
    try:
        # Send server metadata as first message
        metadata = {"action_dim": policy.action_dim, "server": "keyboard_policy_server"}
        metadata_bytes = packb(metadata)
        await websocket.send(metadata_bytes)
        
        async for message in websocket:
            try:
                # Deserialize observation
                obs = unpackb(message)
                
                # Check if this is a reset signal
                if isinstance(obs, dict) and obs.get("reset", False):
                    logger.info("Received reset signal")
                    policy.reset()
                    continue
                
                # Get action from policy (returns current persisted action)
                action = policy.predict(obs)
                
                # Prepare response
                response = {"action": action}
                
                # Serialize and send response
                response_bytes = packb(response)
                await websocket.send(response_bytes)
                
            except msgpack.exceptions.ExtraData as e:
                logger.error(f"msgpack ExtraData error: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
    
    except websockets.exceptions.ConnectionClosedOK:
        print(f"\n[DISC] Client disconnected normally" + " "*30)
        print(f"[CTRL] Joint: {policy.selected_joint:2d} | Value: {policy.action[policy.selected_joint]:+.3f} | Steps: {policy.step_count:4d}" + " "*20, end='', flush=True)
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"\n[ERROR] Connection closed with error: {e}" + " "*30)
        print(f"[CTRL] Joint: {policy.selected_joint:2d} | Value: {policy.action[policy.selected_joint]:+.3f} | Steps: {policy.step_count:4d}" + " "*20, end='', flush=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


async def main(host: str = "0.0.0.0", port: int = 8000, action_dim: int = 23, action_delta: float = 0.1):
    """
    Start the keyboard-controlled WebSocket server.
    
    Args:
        host (str): Host address to bind to
        port (int): Port to listen on
        action_dim (int): Action space dimension
        action_delta (float): Amount to change joint value per keypress
    """
    # Initialize policy
    policy = KeyboardPolicy(action_dim=action_dim, action_delta=action_delta)
    
    # Start keyboard input thread
    keyboard_thread = KeyboardInputThread(policy)
    keyboard_thread.start()
    
    logger.info("=" * 60)
    logger.info("Keyboard-Controlled WebSocket Server for BEHAVIOR")
    logger.info("=" * 60)
    logger.info(f"Server address: ws://{host}:{port}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"Action delta: ±{action_delta}")
    logger.info("=" * 60)
    logger.info("Keyboard input thread started")
    logger.info("Waiting for client connection...")
    logger.info("")
    
    # Helper to handle HTTP requests
    def process_request(connection, request):
        """Handle HTTP health checks."""
        from websockets.http11 import Response
        from websockets.datastructures import Headers
        
        if request.path == "/healthz":
            logger.info("Health check request received")
            return Response(
                status_code=200,
                reason_phrase="OK",
                headers=Headers([("Content-Type", "text/plain")]),
                body=b"OK\n"
            )
        
        conn_hdr = request.headers.get("Connection", "")
        upgrade_hdr = request.headers.get("Upgrade", "")
        
        if "upgrade" in conn_hdr.lower() and "websocket" in upgrade_hdr.lower():
            return None
        
        return Response(
            status_code=426,
            reason_phrase="Upgrade Required",
            headers=Headers([("Content-Type", "text/plain")]),
            body=b"426 Upgrade Required\n"
        )
    
    # Start WebSocket server
    try:
        async with websockets.serve(
            lambda ws: handle_client(ws, policy),
            host,
            port,
            process_request=process_request,
            max_size=100 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
        ):
            # Wait for keyboard thread to finish (user quits)
            while keyboard_thread.is_alive():
                await asyncio.sleep(0.1)
            
            logger.info("Keyboard thread stopped, shutting down server...")
    
    finally:
        keyboard_thread.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keyboard-controlled WebSocket server for BEHAVIOR environment"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=23,
        help="Action space dimension for R1Pro robot (default: 23)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Amount to change joint value per keypress (default: 0.1)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run server
    try:
        asyncio.run(main(
            host=args.host,
            port=args.port,
            action_dim=args.action_dim,
            action_delta=args.delta
        ))
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)

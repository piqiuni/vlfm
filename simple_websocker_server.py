"""
Simple WebSocket Server for BEHAVIOR Environment

This is a minimal WebSocket server that receives observations from the BEHAVIOR environment
and returns zero actions. It serves as a template for building custom policy servers.

Usage:
    # Start server on default port (8000)
    python simple_websocket_server.py
    
    # Custom port
    python simple_websocket_server.py --port 9000
    
    # Custom host and port
    python simple_websocket_server.py --host 0.0.0.0 --port 8080

Then run the environment client:
    python OmniGibson/omnigibson/examples/environments/behavior_env_web.py --host localhost --port 8000
"""

import asyncio
import websockets
import msgpack
import numpy as np
import argparse
import logging
import functools
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("websocket_server")


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


class SimplePolicy:
    """
    A simple policy that returns zero actions.
    Replace this with your own policy implementation.
    """
    
    def __init__(self, action_dim: int = 23):
        """
        Initialize the policy.
        
        Args:
            action_dim (int): Dimension of action space (23 for R1Pro robot)
        """
        self.action_dim = action_dim
        self.step_count = 0
        logger.info(f"Initialized SimplePolicy with action_dim={action_dim}")
    
    def reset(self) -> None:
        """Reset policy state."""
        self.step_count = 0
        logger.info("Policy reset")
    
    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Compute action from observation.
        
        Args:
            obs (dict): Observation dictionary containing:
                - RGB images from cameras
                - Proprioception data
                - Camera relative poses
                - Task ID
        
        Returns:
            np.ndarray: Action array of shape (action_dim,)
        """
        self.step_count += 1
        
        # Log observation info every 10 steps
        if self.step_count % 10 == 1:
            logger.info(f"Step {self.step_count}: Received observation with keys: {list(obs.keys())}")
            
            # Log some observation details
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif self.step_count == 1:
            logger.info(f"First observation received with keys: {list(obs.keys())}")
        
        # Return zero action (robot will remain stationary)
        action = np.zeros(self.action_dim, dtype=np.float32)
        action = [-1]*self.action_dim  # For testing purposes, return -1 actions
        action = np.array(action, dtype=np.float32)
        
        
        return action


async def handle_client(websocket, policy: SimplePolicy):
    """
    Handle WebSocket client connection.
    
    Args:
        websocket: WebSocket connection object
        policy: Policy instance to generate actions
    """
    client_address = websocket.remote_address
    logger.info(f"Client connected from {client_address}")
    
    try:
        # Send server metadata as first message (expected by WebsocketClientPolicy)
        metadata = {"action_dim": policy.action_dim, "server": "simple_websocket_server"}
        metadata_bytes = packb(metadata)
        await websocket.send(metadata_bytes)
        logger.info("Sent server metadata to client")
        
        async for message in websocket:
            try:
                # Deserialize observation using msgpack with NumPy support
                obs = unpackb(message)
                print("get msg")
                # Check if this is a reset signal
                if isinstance(obs, dict) and obs.get("reset", False):
                    logger.info("Received reset signal")
                    policy.reset()
                    continue
                
                # Get action from policy
                action = policy.predict(obs)
                logger.debug(f"Generated action: {action.shape}")
                
                # Prepare response
                response = {"action": action}
                
                # Serialize and send response with NumPy support
                response_bytes = packb(response)
                await websocket.send(response_bytes)
                
            except msgpack.exceptions.ExtraData as e:
                logger.error(f"msgpack ExtraData error: {e}")
                await websocket.send(f"Error: msgpack ExtraData - {str(e)}")
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await websocket.send(f"Error: {str(e)}")
    
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_address} disconnected normally")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error with client {client_address}: {e}", exc_info=True)
    finally:
        logger.info(f"Connection closed with {client_address}")


async def main(host: str = "0.0.0.0", port: int = 8000, action_dim: int = 23):
    """
    Start the WebSocket server.
    
    Args:
        host (str): Host address to bind to
        port (int): Port to listen on
        action_dim (int): Action space dimension
    """
    # Initialize policy
    policy = SimplePolicy(action_dim=action_dim)
    
    logger.info("=" * 60)
    logger.info("Starting Simple WebSocket Server for BEHAVIOR Environment")
    logger.info("=" * 60)
    logger.info(f"Server address: ws://{host}:{port}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"Policy: {policy.__class__.__name__} (returns zero actions)")
    logger.info("=" * 60)
    logger.info("Waiting for client connection...")
    logger.info("")
    
    # Helper to handle HTTP requests (like health checks)
    def process_request(connection, request):
        """
        Handle HTTP requests before WebSocket upgrade.
        - /healthz: Return 200 OK for health checks
        - Other non-WebSocket requests: Return 426 Upgrade Required
        - WebSocket upgrade requests: Allow to proceed (return None)
        """
        from websockets.http11 import Response
        from websockets.datastructures import Headers
        
        # Check if this is a health check endpoint
        if request.path == "/healthz":
            logger.info("Health check request received")
            return Response(
                status_code=200,
                reason_phrase="OK",
                headers=Headers([("Content-Type", "text/plain")]),
                body=b"OK\n"
            )
        
        # Check if this is a WebSocket upgrade request
        conn_hdr = request.headers.get("Connection", "")
        upgrade_hdr = request.headers.get("Upgrade", "")
        
        if "upgrade" in conn_hdr.lower() and "websocket" in upgrade_hdr.lower():
            # This is a valid WebSocket upgrade request, allow it to proceed
            return None
        
        # Non-WebSocket HTTP request - return 426
        logger.warning(f"Non-WebSocket request to {request.path}")
        return Response(
            status_code=426,
            reason_phrase="Upgrade Required",
            headers=Headers([("Content-Type", "text/plain")]),
            body=b"426 Upgrade Required: This endpoint expects a WebSocket connection.\n"
        )

    # Start WebSocket server
    async with websockets.serve(
        lambda ws: handle_client(ws, policy),
        host,
        port,
        process_request=process_request,
        max_size=100 * 1024 * 1024,  # 100 MB max message size (for large images)
        ping_interval=20,             # Send ping every 20 seconds
        ping_timeout=10,              # Wait 10 seconds for pong
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple WebSocket server for BEHAVIOR environment control"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0 for all interfaces)"
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
        asyncio.run(main(host=args.host, port=args.port, action_dim=args.action_dim))
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)

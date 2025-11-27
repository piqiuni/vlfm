"""
WebSocket Server for BEHAVIOR Environment with VLFM Policy

This WebSocket server integrates BehaviorITMPolicyV2 to control robots in the BEHAVIOR
environment. It receives RGB-D observations via WebSocket and returns navigation actions
computed by the VLFM policy with visual odometry.

Requirements:
    1. Start BLIP2-ITM server first:
       python -m vlfm.vlm.blip2itm --port 12182
    
    2. Start this WebSocket server:
       python websocket_vlfm_server.py --port 8000 --target chair
    
    3. Run BEHAVIOR environment client:
       python behavior_env_web.py --host localhost --port 8000

Usage Examples:
    # Default settings (target=chair, zed camera)
    python websocket_vlfm_server.py
    
    # Custom target and camera
    python websocket_vlfm_server.py --target bottle --camera left
    
    # Full configuration
    python websocket_vlfm_server.py --port 9000 --target chair --camera zed \
        --camera-fx 400.0 --visualize --verbose

Author: VLFM BEHAVIOR-1K Integration
Date: 2025-11-26
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import msgpack
import numpy as np
import torch
import websockets

# Add vlfm to path
sys.path.insert(0, str(Path(__file__).parent))

from vlfm.policy.behavior_policies import BehaviorITMPolicyV2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vlfm_websocket_server")


class VLFMBehaviorPolicy:
    """
    VLFM Policy wrapper for BEHAVIOR environment.
    Integrates BehaviorITMPolicyV2 for object-goal navigation with visual odometry.
    """

    # Camera configuration mapping
    CAMERA_CONFIGS = {
        "left": {
            "rgb_key": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
            "depth_key": "robot_r1::robot_r1:left_realsense_link:Camera:0::depth",
            "width": 480,
            "height": 480,
        },
        "right": {
            "rgb_key": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
            "depth_key": "robot_r1::robot_r1:right_realsense_link:Camera:0::depth",
            "width": 480,
            "height": 480,
        },
        "zed": {
            "rgb_key": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
            "depth_key": "robot_r1::robot_r1:zed_link:Camera:0::depth",
            "width": 720,
            "height": 720,
        },
    }

    def __init__(
        self,
        target_object: str = "chair",
        camera_name: str = "zed",
        camera_fx: Optional[float] = None,
        camera_fy: Optional[float] = None,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        text_prompt: Optional[str] = None,
        visualize: bool = False,
        action_dim: int = 23,
    ):
        """
        Initialize VLFM policy for BEHAVIOR environment.

        Args:
            target_object: Name of target object to navigate to
            camera_name: Which camera to use ('left', 'right', or 'zed')
            camera_fx: Camera focal length in x (auto-calculated if None)
            camera_fy: Camera focal length in y (auto-calculated if None)
            min_depth: Minimum valid depth value (meters)
            max_depth: Maximum valid depth value (meters)
            text_prompt: Custom text prompt for ITM model
            visualize: Whether to generate visualization outputs
            action_dim: Dimension of action space (23 for R1Pro robot)
        """
        if camera_name not in self.CAMERA_CONFIGS:
            raise ValueError(f"Invalid camera_name: {camera_name}. Must be one of {list(self.CAMERA_CONFIGS.keys())}")

        self.target_object = target_object
        self.camera_name = camera_name
        self.camera_config = self.CAMERA_CONFIGS[camera_name]
        self.action_dim = action_dim
        self.step_count = 0

        # Auto-calculate focal lengths if not provided (assume 90° FOV)
        if camera_fx is None:
            camera_fx = self.camera_config["width"] / 2.0
        if camera_fy is None:
            camera_fy = self.camera_config["height"] / 2.0

        # Default text prompt if not provided
        if text_prompt is None:
            text_prompt = "This place looks like it has a target_object | " "This place looks promising for exploration"

        logger.info("=" * 80)
        logger.info("Initializing BehaviorITMPolicyV2...")
        logger.info(f"  Target object: {target_object}")
        logger.info(f"  Selected camera: {camera_name} ({self.camera_config['width']}x{self.camera_config['height']})")
        logger.info(f"  Camera params: fx={camera_fx:.1f}, fy={camera_fy:.1f}")
        logger.info(f"  Depth range: [{min_depth}, {max_depth}] meters")
        logger.info(f"  Text prompt: {text_prompt}")
        logger.info("=" * 80)

        try:
            # Initialize VLFM policy with visual odometry
            self.policy = BehaviorITMPolicyV2(
                camera_fx=camera_fx,
                camera_fy=camera_fy,
                min_depth=min_depth,
                max_depth=max_depth,
                text_prompt=text_prompt,
                visualize=visualize,
            )
            logger.info("✓ BehaviorITMPolicyV2 initialized successfully")
            logger.info("  - Visual odometry enabled (SimpleVisualOdometry)")
            logger.info("  - Obstacle mapping enabled")
            logger.info("  - Frontier-based exploration enabled")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"✗ Failed to initialize policy: {e}", exc_info=True)
            logger.error("Make sure BLIP2-ITM server is running!")
            raise

        # Episode state
        self.masks = torch.tensor([[1.0]])  # Episode continues
        self.rnn_hidden_states = None
        self.prev_actions = None

    def reset(self) -> None:
        """Reset policy and episode state for new episode."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"EPISODE RESET - Target: {self.target_object}")
        logger.info("=" * 80)

        self.step_count = 0
        self.masks = torch.tensor([[1.0]])
        self.rnn_hidden_states = None
        self.prev_actions = None

        # Reset policy internal state (visual odometry, maps, etc.)
        self.policy._reset()
        logger.info("Policy state reset complete")

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Compute action from observation.

        Expected observation format:
            obs = {
                "robot_r1::proprio": np.ndarray (256,),
                "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.ndarray (480, 480, 4),
                "robot_r1::robot_r1:left_realsense_link:Camera:0::depth": np.ndarray (480, 480, 1),
                "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.ndarray (480, 480, 4),
                "robot_r1::robot_r1:right_realsense_link:Camera:0::depth": np.ndarray (480, 480, 1),
                "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.ndarray (720, 720, 4),
                "robot_r1::robot_r1:zed_link:Camera:0::depth": np.ndarray (720, 720, 1),
                "robot_r1::cam_rel_poses": np.ndarray (21,),
                "task_id": np.ndarray (1,),
            }

        Args:
            obs: Observation dictionary from BEHAVIOR environment

        Returns:
            np.ndarray: Action array of shape (action_dim,) with base motion commands
        """
        self.step_count += 1

        try:
            # Extract RGB and depth from selected camera
            rgb, depth = self._extract_rgbd(obs)

            if rgb is None or depth is None:
                logger.warning(f"Step {self.step_count}: Missing RGB or depth, returning zero action")
                return np.zeros(self.action_dim, dtype=np.float32)

            # Log observation info periodically
            if self.step_count == 1 or self.step_count % 50 == 0:
                logger.info(f"Step {self.step_count}:")
                logger.info(f"  RGB: shape={rgb.shape}, dtype={rgb.dtype}, range=[{rgb.min()}, {rgb.max()}]")
                logger.info(
                    f"  Depth: shape={depth.shape}, dtype={depth.dtype}, "
                    f"range=[{depth.min():.3f}, {depth.max():.3f}]m"
                )
                if "task_id" in obs:
                    logger.info(f"  Task ID: {obs['task_id']}")

            # Prepare observation for VLFM policy
            policy_obs = {
                "rgb": rgb,  # (H, W, 3) uint8
                "depth": depth,  # (H, W) float32 in meters
                "objectgoal": self.target_object,
            }

            # Get action from VLFM policy
            action_tensor, self.rnn_hidden_states = self.policy.act(
                observations=policy_obs,
                rnn_hidden_states=self.rnn_hidden_states,
                prev_actions=self.prev_actions,
                masks=self.masks,
                deterministic=True,
            )

            # Extract action values: (angular_vel, linear_vel)
            vlfm_action = action_tensor.cpu().numpy()[0]  # Shape: (2,)
            angular_vel = float(vlfm_action[0])
            linear_vel = float(vlfm_action[1])

            # Log non-zero actions
            if self.step_count % 10 == 0 or abs(angular_vel) > 0.01 or abs(linear_vel) > 0.01:
                logger.info(f"  VLFM Action: angular={angular_vel:.3f} rad/s, linear={linear_vel:.3f} m/s")

            # Convert VLFM action to R1Pro robot action space
            # R1Pro action space (23D): [base_motion, arm_joints, gripper, etc.]
            # For now: only control base (first 2-3 DOFs)
            robot_action = np.zeros(self.action_dim, dtype=np.float32)

            # Map to base control (adjust indices based on actual R1Pro action space)
            # TODO: Verify correct action indices from R1Pro documentation
            robot_action[0] = angular_vel  # Base rotation (yaw)
            robot_action[1] = linear_vel  # Base forward velocity
            # robot_action[2] = 0.0        # Base lateral velocity (if applicable)

            self.prev_actions = action_tensor

            return robot_action

        except Exception as e:
            logger.error(f"Step {self.step_count}: Error in predict: {e}", exc_info=True)
            return np.zeros(self.action_dim, dtype=np.float32)

    def _extract_rgbd(self, obs: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract RGB and depth images from BEHAVIOR observation.

        Args:
            obs: Observation dictionary with camera data

        Returns:
            tuple: (rgb, depth) as numpy arrays
                - rgb: (H, W, 3) uint8, range [0, 255]
                - depth: (H, W) float32, in meters
                Returns (None, None) if extraction fails
        """
        try:
            # Get camera keys
            rgb_key = self.camera_config["rgb_key"]
            depth_key = self.camera_config["depth_key"]

            # Extract RGB
            if rgb_key not in obs:
                logger.error(f"RGB key '{rgb_key}' not found in observation")
                logger.error(f"Available keys: {list(obs.keys())}")
                return None, None

            rgb = obs[rgb_key]  # Shape: (H, W, 4) uint8 (RGBA)

            # Convert RGBA to RGB
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]  # Drop alpha channel

            # Ensure uint8
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            # Extract depth
            if depth_key not in obs:
                logger.error(f"Depth key '{depth_key}' not found in observation")
                return None, None

            depth = obs[depth_key]  # Shape: (H, W, 1) float32

            # Squeeze to 2D if needed
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]

            # Ensure float32
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)

            # Depth should already be in meters from BEHAVIOR
            # If values seem too large (e.g., in mm), convert
            if depth.max() > 100.0:
                logger.warning(f"Depth values seem large (max={depth.max():.1f}), converting mm to meters")
                depth = depth / 1000.0

            return rgb, depth

        except Exception as e:
            logger.error(f"Error extracting RGB-D: {e}", exc_info=True)
            return None, None


async def handle_client(websocket, path, policy: VLFMBehaviorPolicy):
    """
    Handle WebSocket client connection.

    Args:
        websocket: WebSocket connection object
        path: Connection path
        policy: VLFMBehaviorPolicy instance to generate actions
    """
    client_address = websocket.remote_address
    logger.info(f"Client connected from {client_address}")

    try:
        async for message in websocket:
            try:
                # Deserialize observation using msgpack
                obs = msgpack.unpackb(message, raw=False)

                # Check if this is a reset signal
                if isinstance(obs, dict) and obs.get("reset", False):
                    logger.info("Received reset signal from client")
                    policy.reset()

                    # Send acknowledgment
                    response = {"status": "reset_ok"}
                    response_bytes = msgpack.packb(response, use_bin_type=True)
                    await websocket.send(response_bytes)
                    continue

                # Get action from policy
                action = policy.predict(obs)

                # Prepare response
                response = {"action": action}

                # Serialize and send response
                response_bytes = msgpack.packb(response, use_bin_type=True)
                await websocket.send(response_bytes)

            except msgpack.exceptions.ExtraData as e:
                logger.error(f"msgpack ExtraData error: {e}")
                error_response = msgpack.packb({"error": f"msgpack ExtraData - {str(e)}"}, use_bin_type=True)
                await websocket.send(error_response)
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                error_response = msgpack.packb({"error": str(e)}, use_bin_type=True)
                await websocket.send(error_response)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_address} disconnected normally")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error with client {client_address}: {e}", exc_info=True)
    finally:
        logger.info(f"Connection closed with {client_address}")


async def main(
    host: str = "0.0.0.0",
    port: int = 8000,
    target_object: str = "chair",
    camera_name: str = "zed",
    camera_fx: Optional[float] = None,
    camera_fy: Optional[float] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    text_prompt: Optional[str] = None,
    visualize: bool = False,
    action_dim: int = 23,
):
    """
    Start the WebSocket server with VLFM policy.

    Args:
        host: Host address to bind to
        port: Port to listen on
        target_object: Target object name
        camera_name: Camera to use ('left', 'right', or 'zed')
        camera_fx: Camera focal length x (auto if None)
        camera_fy: Camera focal length y (auto if None)
        min_depth: Minimum depth in meters
        max_depth: Maximum depth in meters
        text_prompt: Custom text prompt
        visualize: Enable visualization
        action_dim: Action space dimension
    """
    # Check if BLIP2ITM server is accessible
    blip2_port = int(os.environ.get("BLIP2ITM_PORT", "12182"))
    logger.info("")
    logger.info("=" * 80)
    logger.info("PREREQUISITES CHECK")
    logger.info("=" * 80)
    logger.info(f"BLIP2-ITM server should be running at: http://localhost:{blip2_port}/blip2itm")
    logger.warning("If not started yet, run in another terminal:")
    logger.warning(f"  python -m vlfm.vlm.blip2itm --port {blip2_port}")
    logger.info("=" * 80)
    logger.info("")

    # Initialize policy
    try:
        policy = VLFMBehaviorPolicy(
            target_object=target_object,
            camera_name=camera_name,
            camera_fx=camera_fx,
            camera_fy=camera_fy,
            min_depth=min_depth,
            max_depth=max_depth,
            text_prompt=text_prompt,
            visualize=visualize,
            action_dim=action_dim,
        )
    except Exception as e:
        logger.error(f"Failed to initialize VLFM policy: {e}")
        logger.error("Make sure BLIP2-ITM server is running!")
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info("VLFM WEBSOCKET SERVER - READY")
    logger.info("=" * 80)
    logger.info(f"Server address: ws://{host}:{port}")
    logger.info(f"Target object: {target_object}")
    logger.info(f"Camera: {camera_name}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info("=" * 80)
    logger.info("Waiting for BEHAVIOR environment client to connect...")
    logger.info("")
    logger.info("Start the client with:")
    client_host = "localhost" if host == "0.0.0.0" else host
    logger.info(f"  python behavior_env_web.py --host {client_host} --port {port}")
    logger.info("=" * 80)
    logger.info("")

    # Start WebSocket server
    async with websockets.serve(
        lambda ws, path: handle_client(ws, path, policy),
        host,
        port,
        max_size=100 * 1024 * 1024,  # 100 MB max message size (for large images)
        ping_interval=20,  # Send ping every 20 seconds
        ping_timeout=10,  # Wait 10 seconds for pong
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLFM WebSocket server for BEHAVIOR environment control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--target", type=str, default="chair", help="Target object to navigate to (e.g., chair, bottle, apple)"
    )
    parser.add_argument(
        "--camera", type=str, default="zed", choices=["left", "right", "zed"], help="Which camera to use for navigation"
    )
    parser.add_argument(
        "--camera-fx", type=float, default=None, help="Camera focal length in x direction (auto-calculated if not set)"
    )
    parser.add_argument(
        "--camera-fy", type=float, default=None, help="Camera focal length in y direction (auto-calculated if not set)"
    )
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum valid depth in meters")
    parser.add_argument("--max-depth", type=float, default=10.0, help="Maximum valid depth in meters")
    parser.add_argument("--text-prompt", type=str, default=None, help="Custom text prompt for ITM model")
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization outputs (value maps, trajectories)"
    )
    parser.add_argument("--action-dim", type=int, default=23, help="Action space dimension for R1Pro robot")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (debug level)")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("vlfm").setLevel(logging.DEBUG)

    # Run server
    try:
        asyncio.run(
            main(
                host=args.host,
                port=args.port,
                target_object=args.target,
                camera_name=args.camera,
                camera_fx=args.camera_fx,
                camera_fy=args.camera_fy,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                text_prompt=args.text_prompt,
                visualize=args.visualize,
                action_dim=args.action_dim,
            )
        )
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Server stopped by user (Ctrl+C)")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)

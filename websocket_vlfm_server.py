"""
WebSocket Server for BEHAVIOR Environment with VLFM Policy

This WebSocket server integrates BehaviorITMPolicyV2 to control robots in the BEHAVIOR
environment. It receives RGB-D observations via WebSocket and returns navigation actions
computed by the VLFM policy with visual odometry.

Requirements:
    1. Configure settings in behavior.yaml file
    
    2. Start BLIP2-ITM server first:
       python -m vlfm.vlm.blip2itm --port 12182
    
    3. Start this WebSocket server:
       python websocket_vlfm_server.py
    
    4. Run BEHAVIOR environment client:
       python behavior_env_web.py --host localhost --port 8000

Usage Examples:
    # Use default configuration from behavior.yaml
    python websocket_vlfm_server.py
    
    # Override specific settings
    python websocket_vlfm_server.py --target bottle --camera left --port 9000
    
    # Use custom configuration file
    python websocket_vlfm_server.py --config my_behavior.yaml
    
    # Enable visualization and verbose logging
    python websocket_vlfm_server.py --visualize --verbose

Configuration:
    All parameters are configured in behavior.yaml. Command-line arguments
    can override specific settings without modifying the config file.

Author: VLFM BEHAVIOR-1K Integration
Date: 2025-11-28
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import msgpack
import functools
import numpy as np
import torch
import websockets
import yaml
import cv2

# Add vlfm to path
sys.path.insert(0, str(Path(__file__).parent))

# NumPy array support for msgpack (same as simple_websocket_server)
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
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}

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

from vlfm.policy.behavior_policies import BehaviorITMPolicyV2

# Setup logging (will be reconfigured after loading config)
logger = logging.getLogger("vlfm_websocket_server")


class VLFMBehaviorPolicy:
    """
    VLFM Policy wrapper for BEHAVIOR environment.
    Integrates BehaviorITMPolicyV2 for object-goal navigation with visual odometry.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VLFM policy for BEHAVIOR environment.

        Args:
            config: Configuration dictionary loaded from behavior.yaml
        """
        # Extract configuration
        self.target_object = config["target"]["object"]
        self.camera_name = config["camera"]["name"]
        self.camera_configs = config["camera_configs"]
        self.action_dim = config["policy"]["action_dim"]
        self.step_count = 0

        if self.camera_name not in self.camera_configs:
            raise ValueError(f"Invalid camera_name: {self.camera_name}. Must be one of {list(self.camera_configs.keys())}")

        self.camera_config = self.camera_configs[self.camera_name]

        # Camera parameters
        camera_fx = config["camera"]["fx"]
        camera_fy = config["camera"]["fy"]
        min_depth = config["camera"]["min_depth"]
        max_depth = config["camera"]["max_depth"]
        
        # Auto-calculate focal lengths if not provided (assume 90° FOV)
        if camera_fx is None:
            camera_fx = self.camera_config["width"] / 2.0
        if camera_fy is None:
            camera_fy = self.camera_config["height"] / 2.0

        # Text prompt and visualization
        text_prompt = config["policy"]["text_prompt"]
        visualize = config["policy"]["visualize"]

        # VLM server configurations
        vlm_servers = config["vlm_servers"]
        blip2itm_host = vlm_servers["blip2itm"]["host"]
        blip2itm_port = vlm_servers["blip2itm"]["port"]
        grounding_dino_host = vlm_servers["grounding_dino"]["host"]
        grounding_dino_port = vlm_servers["grounding_dino"]["port"]
        yolov7_host = vlm_servers["yolov7"]["host"]
        yolov7_port = vlm_servers["yolov7"]["port"]
        mobile_sam_host = vlm_servers["mobile_sam"]["host"]
        mobile_sam_port = vlm_servers["mobile_sam"]["port"]
        blip2_host = vlm_servers["blip2"]["host"]
        blip2_port = vlm_servers["blip2"]["port"]
        use_vqa = vlm_servers["blip2"].get("enabled", False)
        
        pointnav_policy_path = config["policy"].get("pointnav_policy_path", None)
        depth_image_shape = (self.camera_config["height"], self.camera_config["width"])
        pointnav_stop_radius = config["policy"].get("pointnav_stop_radius", 0.2)
        object_map_erosion_size= config["policy"].get("object_map_erosion_size", 5)

        logger.info("=" * 80)
        logger.info("Initializing BehaviorITMPolicyV2...")
        logger.info(f"  Target object: {self.target_object}")
        logger.info(f"  Selected camera: {self.camera_name} ({self.camera_config['width']}x{self.camera_config['height']})")
        logger.info(f"  Camera params: fx={camera_fx:.1f}, fy={camera_fy:.1f}")
        logger.info(f"  Depth range: [{min_depth}, {max_depth}] meters")
        logger.info(f"  Text prompt: {text_prompt}")
        logger.info("=" * 80)
        logger.info("VLM Server Configurations:")
        logger.info(f"  BLIP2-ITM:      {blip2itm_host}:{blip2itm_port}")
        logger.info(f"  Grounding DINO: {grounding_dino_host}:{grounding_dino_port}")
        logger.info(f"  YOLOv7:         {yolov7_host}:{yolov7_port}")
        logger.info(f"  Mobile SAM:     {mobile_sam_host}:{mobile_sam_port}")
        logger.info(f"  BLIP2 (VQA):    {blip2_host}:{blip2_port} (enabled={use_vqa})")
        logger.info("=" * 80)

        try:
            # Initialize VLFM policy with visual odometry and VLM server configs
            self.policy = BehaviorITMPolicyV2(
                camera_fx=camera_fx,
                camera_fy=camera_fy,
                min_depth=min_depth,
                max_depth=max_depth,
                text_prompt=text_prompt,
                visualize=visualize,
                # BLIP2-ITM server config
                blip2itm_host=blip2itm_host,
                blip2itm_port=blip2itm_port,
                # Object detection servers
                grounding_dino_host=grounding_dino_host,
                grounding_dino_port=grounding_dino_port,
                yolov7_host=yolov7_host,
                yolov7_port=yolov7_port,
                # Segmentation server
                mobile_sam_host=mobile_sam_host,
                mobile_sam_port=mobile_sam_port,
                # VQA server (optional)
                use_vqa=use_vqa,
                blip2_host=blip2_host,
                blip2_port=blip2_port,
                
                pointnav_policy_path = pointnav_policy_path,
                depth_image_shape=depth_image_shape,
                pointnav_stop_radius=pointnav_stop_radius,
                object_map_erosion_size=object_map_erosion_size,
                
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

            # Save visualization for debugging
            try:
                
                # Normalize depth for visualization (0-5m range mapped to 0-255)
                depth_vis = np.clip(depth, 0, 5.0) / 5.0
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                
                # Resize depth to match RGB if needed (though they should match based on config)
                if depth_vis.shape[:2] != rgb.shape[:2]:
                    depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]))
                
                # Concatenate horizontally
                vis_img = np.hstack((rgb, depth_vis))
                
                # Add text info
                text = f"Step: {self.step_count} | Target: {self.target_object}"
                cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Save to absolute path
                save_path = f"/home/wxy/Experiments/vlfm/vis_debug/step_{self.step_count:04d}.jpg"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Convert RGB to BGR for OpenCV saving
                cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                logger.warning(f"Failed to save debug visualization: {e}")

            # Get action from VLFM policy
            action_tensor, self.rnn_hidden_states = self.policy.act(
                observations=policy_obs,
                rnn_hidden_states=self.rnn_hidden_states,
                prev_actions=self.prev_actions,
                masks=self.masks,
                deterministic=True,
            )
            logger.info(f"Step {self.step_count}: VLFM policy action tensor: {action_tensor}")
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
            robot_action[2] = angular_vel  # Base rotation (yaw)
            robot_action[0] = linear_vel  # Base forward velocity
            robot_action[3] = -0.2
            # robot_action[2] = 0.0        # Base lateral velocity (if applicable)
            print(f"robot_action: {robot_action}")
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


async def handle_client(websocket, policy: VLFMBehaviorPolicy):
    """
    Handle WebSocket client connection.

    Args:
        websocket: WebSocket connection object
        policy: VLFMBehaviorPolicy instance to generate actions
    """
    client_address = websocket.remote_address
    logger.info(f"Client connected from {client_address}")

    try:
        # Send server metadata as first message (expected by clients)
        metadata = {"action_dim": policy.action_dim, "server": "vlfm_websocket_server"}
        metadata_bytes = packb(metadata)
        await websocket.send(metadata_bytes)
        logger.info("Sent server metadata to client")

        async for message in websocket:
            try:
                # Deserialize observation using msgpack with NumPy support
                obs = unpackb(message)

                # Check if this is a reset signal
                if isinstance(obs, dict) and obs.get("reset", False):
                    logger.info("Received reset signal from client")
                    policy.reset()

                    # Send acknowledgment
                    response = {"status": "reset_ok"}
                    response_bytes = packb(response)
                    await websocket.send(response_bytes)
                    continue

                # Get action from policy
                action = policy.predict(obs)

                # Prepare response
                response = {"action": action}

                # Serialize and send response with NumPy support
                response_bytes = packb(response)
                await websocket.send(response_bytes)

            except msgpack.exceptions.ExtraData as e:
                logger.error(f"msgpack ExtraData error: {e}")
                await websocket.send(packb({"error": f"msgpack ExtraData - {str(e)}"}))
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await websocket.send(packb({"error": str(e)}))

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_address} disconnected normally")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error with client {client_address}: {e}", exc_info=True)
    finally:
        logger.info(f"Connection closed with {client_address}")


async def main(config: Dict[str, Any]):
    """
    Start the WebSocket server with VLFM policy.

    Args:
        config: Configuration dictionary loaded from behavior.yaml
    """
    # Extract server configuration
    host = config["server"]["host"]
    port = config["server"]["port"]
    max_size = config["server"]["max_message_size"]
    ping_interval = config["server"]["ping_interval"]
    ping_timeout = config["server"]["ping_timeout"]
    
    # Check if BLIP2ITM server is accessible
    blip2_host = config["blip2itm"]["host"]
    blip2_port = config["blip2itm"]["port"]
    logger.info("")
    logger.info("=" * 80)
    logger.info("PREREQUISITES CHECK")
    logger.info("=" * 80)
    logger.info(f"BLIP2-ITM server should be running at: http://{blip2_host}:{blip2_port}/blip2itm")
    logger.warning("If not started yet, run in another terminal:")
    logger.warning(f"  python -m vlfm.vlm.blip2itm --port {blip2_port}")
    logger.info("=" * 80)
    logger.info("")

    # Initialize policy
    try:
        policy = VLFMBehaviorPolicy(config)
    except Exception as e:
        logger.error(f"Failed to initialize VLFM policy: {e}")
        logger.error("Make sure BLIP2-ITM server is running!")
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info("VLFM WEBSOCKET SERVER - READY")
    logger.info("=" * 80)
    logger.info(f"Server address: ws://{host}:{port}")
    logger.info(f"Target object: {config['target']['object']}")
    logger.info(f"Camera: {config['camera']['name']}")
    logger.info(f"Action dimension: {config['policy']['action_dim']}")
    logger.info("=" * 80)
    logger.info("Waiting for BEHAVIOR environment client to connect...")
    logger.info("")
    logger.info("Start the client with:")
    client_host = "localhost" if host == "0.0.0.0" else host
    logger.info(f"  python behavior_env_web.py --host {client_host} --port {port}")
    logger.info("=" * 80)
    logger.info("")

    # Helper to handle HTTP requests (like health checks)
    def process_request(connection, request):
        """
        Handle HTTP requests before WebSocket upgrade.
        - /healthz: Return 200 OK for health checks
        - Other non-WebSocket requests: Return 426 Upgrade Required
        - WebSocket upgrade requests: Allow to proceed (return None)
        """
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        # Health check
        if request.path == "/healthz":
            logger.info("Health check request received")
            return Response(
                status_code=200,
                reason_phrase="OK",
                headers=Headers([("Content-Type", "text/plain")]),
                body=b"OK\n",
            )

        # Validate Upgrade headers
        conn_hdr = request.headers.get("Connection", "")
        upgrade_hdr = request.headers.get("Upgrade", "")

        if "upgrade" in conn_hdr.lower() and "websocket" in upgrade_hdr.lower():
            return None

        logger.warning(f"Non-WebSocket request to {request.path}")
        return Response(
            status_code=426,
            reason_phrase="Upgrade Required",
            headers=Headers([("Content-Type", "text/plain")]),
            body=b"426 Upgrade Required: This endpoint expects a WebSocket connection.\n",
        )

    # Start WebSocket server
    async with websockets.serve(
        lambda ws: handle_client(ws, policy),
        host,
        port,
        process_request=process_request,
        max_size=max_size,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    ):
        await asyncio.Future()  # Run forever


def load_config(config_path: str = "behavior.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(__file__).parent / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLFM WebSocket server for BEHAVIOR environment control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="behavior.yaml", 
        help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--target", 
        type=str, 
        default=None, 
        help="Override target object from config"
    )
    parser.add_argument(
        "--camera", 
        type=str, 
        default=None, 
        choices=["left", "right", "zed"], 
        help="Override camera selection from config"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None, 
        help="Override server port from config"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Enable visualization (overrides config)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging (debug level)"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Apply command-line overrides
    if args.target is not None:
        config["target"]["object"] = args.target
    if args.camera is not None:
        config["camera"]["name"] = args.camera
    if args.port is not None:
        config["server"]["port"] = args.port
    if args.visualize:
        config["policy"]["visualize"] = True

    # Configure logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, config["logging"]["level"])
    logging.basicConfig(
        level=log_level,
        format=config["logging"]["format"]
    )
    
    if args.verbose:
        logging.getLogger("vlfm").setLevel(logging.DEBUG)

    # Run server
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Server stopped by user (Ctrl+C)")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)

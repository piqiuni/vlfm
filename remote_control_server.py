"""
Remote Control WebSocket Server for BEHAVIOR Environment

This server allows you to control the robot using WASD keys while running VLM detections
and visual odometry in real-time.

Features:
    - WASD Control: Control linear and angular velocity
    - VLM Detection: Runs GroundingDINO, YOLOv7, MobileSAM, BLIP2 on observations
    - Visual Odometry: Estimates and prints robot pose
    - Visualization: Shows detections and top-down map of robot path

Usage:
    python remote_control_server.py

Requirements:
    - VLM servers running (ports 12181-12185)
    - behavior.yaml configuration file
"""

import argparse
import asyncio
import functools
import logging
import select
import sys
import termios
import threading
import tty
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import msgpack
import numpy as np
import websockets
import yaml

# Add vlfm to path
sys.path.insert(0, str(Path(__file__).parent))

from vlfm.policy.behavior_policies import SimpleVisualOdometry
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.grounding_dino import GroundingDINOClient
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client
from vlfm.vlm.detections import annotate

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("remote_control_server")
logging.getLogger("websockets").setLevel(logging.WARNING)


# NumPy array support for msgpack
def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class RemoteControlPolicy:
    """
    Policy that combines manual control, VLM detection, and Visual Odometry.
    """

    def __init__(self, config: Dict[str, Any]):
        self.action_dim = config["policy"]["action_dim"]
        self.camera_name = config["camera"]["name"]
        self.camera_config = config["camera_configs"][self.camera_name]
        
        # Control state
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.max_lin_vel = 0.3
        self.max_ang_vel = 0.3
        
        # Initialize VLM clients
        vlm_cfg = config["vlm_servers"]
        self.gd_client = GroundingDINOClient(host=vlm_cfg["grounding_dino"]["host"], port=vlm_cfg["grounding_dino"]["port"])
        self.yolo_client = YOLOv7Client(host=vlm_cfg["yolov7"]["host"], port=vlm_cfg["yolov7"]["port"])
        self.sam_client = MobileSAMClient(host=vlm_cfg["mobile_sam"]["host"], port=vlm_cfg["mobile_sam"]["port"])
        self.blip2_client = BLIP2Client(host=vlm_cfg["blip2"]["host"], port=vlm_cfg["blip2"]["port"])
        
        # Initialize Visual Odometry
        fx = config["camera"]["fx"] or (self.camera_config["width"] / 2.0)
        fy = config["camera"]["fy"] or (self.camera_config["height"] / 2.0)
        self.vo = SimpleVisualOdometry(fx=fx, fy=fy)
        
        # Visualization state
        self.map_size = 250
        self.map_scale = 50  # pixels per meter
        self.map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.traj_points = []
        
        self.step_count = 0
        self.print_controls()

    def print_controls(self):
        logger.info("\n" + "=" * 60)
        logger.info("Remote Control Instructions")
        logger.info("=" * 60)
        logger.info("  w        - Move Forward")
        logger.info("  s        - Move Backward")
        logger.info("  a        - Turn Left")
        logger.info("  d        - Turn Right")
        logger.info("  space    - Stop")
        logger.info("  q        - Quit")
        logger.info("=" * 60 + "\n")

    def update_control(self, key: str) -> bool:
        """Update velocity based on key press."""
        if key == 'w':
            self.linear_vel = self.max_lin_vel
        elif key == 's':
            self.linear_vel = -self.max_lin_vel
        elif key == 'a':
            self.angular_vel = self.max_ang_vel
        elif key == 'd':
            self.angular_vel = -self.max_ang_vel
        elif key == ' ':
            self.linear_vel = 0.0
            self.angular_vel = 0.0
        elif key == 'q':
            return False
        return True

    def reset(self):
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.vo.reset()
        self.traj_points = []
        self.map_img.fill(0)
        self.step_count = 0
        print("[RESET] Policy reset")

    def _extract_rgbd(self, obs: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rgb_key = self.camera_config["rgb_key"]
        depth_key = self.camera_config["depth_key"]
        
        if rgb_key not in obs or depth_key not in obs:
            return None, None
            
        rgb = obs[rgb_key]
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
            
        depth = obs[depth_key]
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        if depth.max() > 100.0:
            depth = depth / 1000.0
            
        return rgb, depth

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        if self.step_count == 0:
            self.init_robot_xyz = np.array(obs["robot_pos"])
        self.step_count += 1
        rgb, depth = self._extract_rgbd(obs)
        logger.info(f"\n[STEP {self.step_count}] Predict called")
        if rgb is not None and depth is not None:
            # 1. Visual Odometry
            # self.vo.estimate_motion(rgb, depth)
            
            robot_xyz = np.array(obs["robot_pos"]) # x,y,z
            x, y, z, w = np.array(obs["robot_ori"]) # 4元数
            robot_heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
            
            
            pos = robot_xyz[:2]
            heading = robot_heading
            print(f"\r[VO] Pos: ({pos[0]:.2f}, {pos[1]:.2f}) Heading: {np.rad2deg(heading):.1f}°", end="", flush=True)
            
            # 2. Map Visualization (Update first so we can use it in VLM vis)
            self._update_map_vis(pos)

            # 3. VLM Detection (every 30 steps to avoid lag)
            if self.step_count % 30 == 0:
                print("\n[VLM] Running detections...", flush=True)
                try:
                    detections_start = asyncio.get_event_loop().time()
                    # Grounding DINO
                    detections = self.gd_client.predict(rgb, caption="chair . table . door . window . sofa . bed . monitor . keyboard . mouse . bottle . cup . laptop . phone . backpack . clock . plant . picture frame . fan . light")
                    print(f"  GD: Found {detections.num_detections} objects", flush=True)
                    
                    # YOLOv7
                    yolo_dets = self.yolo_client.predict(rgb)
                    print(f"  YOLO: Found {yolo_dets.num_detections} objects", flush=True)
                    
                    answer = ""
                    # BLIP2
                    # answer = self.blip2_client.ask(rgb, "What is the most prominent object?")
                    # print(f"  BLIP2: {answer}")
                    
                    detections_end = asyncio.get_event_loop().time()
                    print(f"  VLM Detections took {detections_end - detections_start:.2f} seconds", flush=True)
                    
                    # Visualization
                    vis_img = rgb.copy()
                    
                    # Draw Grounding DINO
                    if detections.num_detections > 0:
                        detections.phrases = [f"GD: {p}" for p in detections.phrases]
                        vis_img = annotate(vis_img, detections.boxes, detections.logits, detections.phrases)
                        
                    # Draw YOLOv7
                    if yolo_dets.num_detections > 0:
                        yolo_dets.phrases = [f"YOLO: {p}" for p in yolo_dets.phrases]
                        vis_img = annotate(vis_img, yolo_dets.boxes, yolo_dets.logits, yolo_dets.phrases)

                    cv2.putText(vis_img, f"BLIP2: {answer}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Combine with Map
                    # Resize map to match image height
                    h, w, _ = vis_img.shape
                    map_resized = cv2.resize(self.map_img, (h, h), interpolation=cv2.INTER_NEAREST)
                    
                    combined_img = np.hstack((vis_img, map_resized))
                    
                    # Save visualization
                    cv2.imwrite(f"vis_debug/remote_{self.step_count:04d}.jpg", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f"vis_debug/remote.jpg", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
                    
                except Exception as e:
                    logger.error(f"  VLM Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save raw RGB for debugging (similar to websocket_vlfm_server)
        # if rgb is not None:
             # Save to absolute path
            # save_path = f"/home/wxy/Experiments/vlfm/vis_debug/step_{self.step_count:04d}.jpg"
            # Convert RGB to BGR for OpenCV saving
            # cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Return action [angular, linear, ...]
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[2] = self.angular_vel  # Yaw
        action[0] = self.linear_vel   # Forward
        action[3] = -0.2   # Forward
        
        # Reset action after sending (so robot doesn't keep moving if key is released)
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        return action

    def _update_map_vis(self, pos):
        # Convert world pos to map pixels
        cx, cy = self.map_size // 2, self.map_size // 2
        pos = pos - self.init_robot_xyz[:2]
        mx = int(cx + pos[0] * self.map_scale)
        my = int(cy - pos[1] * self.map_scale) # Flip Y
        
        self.traj_points.append((mx, my))
        
        # Draw trajectory
        if len(self.traj_points) > 1:
            cv2.polylines(self.map_img, [np.array(self.traj_points)], False, (0, 255, 255), 1)
            
        # Draw robot
        cv2.circle(self.map_img, (mx, my), 3, (0, 0, 255), -1)
        
        # Save map occasionally
        if self.step_count % 10 == 0:
            cv2.imwrite("vis_debug/map_vis.jpg", self.map_img)


class InputThread(threading.Thread):
    def __init__(self, policy):
        super().__init__(daemon=True)
        self.policy = policy
        self.running = True

    def run(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if not self.policy.update_control(key):
                        self.running = False
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


async def handle_client(websocket, policy):
    logger.info(f"\n[CONN] Client connected: {websocket.remote_address}")
    try:
        # Send metadata
        metadata = {"action_dim": policy.action_dim, "server": "remote_control"}
        await websocket.send(packb(metadata))

        async for message in websocket:
            obs = unpackb(message)
            if isinstance(obs, dict) and obs.get("reset", False):
                policy.reset()
                await websocket.send(packb({"status": "reset_ok"}))
                continue
                
            action = policy.predict(obs)
            await websocket.send(packb({"action": action}))
            
    except Exception as e:
        logger.error(f"Connection error: {e}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="behavior.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)


    host = config["server"]["host"]
    port = config["server"]["port"]
    max_size = config["server"]["max_message_size"]
    ping_interval = config["server"].get("ping_interval", 20)
    ping_timeout = config["server"].get("ping_timeout", 20)

    logger.info(f"init RemoteControlPolicy with config: {args.config}")
    policy = RemoteControlPolicy(config)
    logger.info(f"init ok")
    
    # Start input thread
    input_thread = InputThread(policy)
    input_thread.start()

    # HTTP Handler
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

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

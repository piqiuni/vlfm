# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""
BEHAVIOR-1K Simulator Policy Adapter with Visual Odometry.

This module provides policy adapters for running ITMPolicyV3 in BEHAVIOR-1K simulator
or other simulation platforms with limited sensor inputs. It addresses the challenge of
using policies designed for rich sensor suites (multi-camera, GPS, compass) in environments
that only provide a single RGB-D camera stream.

Key Features:
    - SimpleVisualOdometry: RGB-D based visual odometry using ORB features for pose estimation
    - BehaviorMixin: Adapter layer that bridges single-camera observations to ITMPolicyV3 requirements
    - BehaviorITMPolicyV3: Complete policy implementation ready for BEHAVIOR-1K integration

Main Components:
    1. SimpleVisualOdometry: Estimates robot pose from consecutive RGB-D frames using feature
       matching and RANSAC-based rigid transformation. Can be replaced with more sophisticated
       SLAM systems (e.g., ORB-SLAM3, pyslam) by implementing the same interface.

    2. BehaviorMixin: Transforms single RGB-D camera observations into the format expected
       by ITMPolicyV3, including:
       - Robot localization via visual odometry
       - Obstacle map construction and updates
       - Frontier detection for exploration
       - Value map and object map generation

    3. BehaviorITMPolicyV3: Final policy class combining BehaviorMixin and ITMPolicyV3 for
       object-goal navigation in BEHAVIOR-1K simulator.

Usage Example:
    ```python
    from vlfm.policy.behavior_policies import BehaviorITMPolicyV3

    # Initialize policy
    policy = BehaviorITMPolicyV3(
        camera_fx=320.0,  # Camera focal length in pixels
        camera_fy=320.0,
        min_depth=0.1,    # Minimum valid depth in meters
        max_depth=10.0,   # Maximum valid depth in meters
        text_prompt="This looks like a target_object | This looks promising for exploration",
        visualize=True,
    )

    # In simulation loop
    observations = {
        'rgb': rgb_image,      # (H, W, 3) uint8
        'depth': depth_image,  # (H, W) float, normalized [0,1] or in meters
        'objectgoal': 'chair'  # Target object name
    }
    masks = torch.tensor([[1.0]])  # Episode continuation mask
    action, _ = policy.act(observations, None, None, masks, deterministic=True)
    # Returns: action tensor (1, 2) with [angular_velocity, linear_velocity]
    ```

Design Philosophy:
    - Modularity: Visual odometry is isolated and easily replaceable
    - Compatibility: Maintains ITMPolicyV3 interface while adapting to limited sensors
    - Flexibility: Works with normalized depth [0,1] or metric depth in meters
    - Extensibility: Can be extended to support other simulators with similar constraints

Notes:
    - The simple VO implementation may accumulate drift over long trajectories
    - For production use, consider replacing SimpleVisualOdometry with ORB-SLAM3 or similar
    - Depth normalization is automatically handled based on input range detection
    - All coordinate frames follow the convention: x-forward, y-left, z-up in world frame

Author: Auto-generated adapter for BEHAVIOR-1K integration
Date: 2025-11-26
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.itm_policy import ITMPolicyV3


class SimpleVisualOdometry:
    """
    Simple visual odometry using ORB features and RGB-D data.
    This implementation can be easily replaced with more sophisticated SLAM systems.
    """

    def __init__(self, fx: float = 320.0, fy: float = 320.0):
        """
        Initialize visual odometry.

        Args:
            fx: Camera focal length in x direction (pixels)
            fy: Camera focal length in y direction (pixels)
        """
        self.fx = fx
        self.fy = fy

        # State variables
        self.prev_rgb: Optional[np.ndarray] = None
        self.prev_depth: Optional[np.ndarray] = None
        self.position = np.array([0.0, 0.0])  # (x, y) in meters
        self.heading = 0.0  # radians

        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # RANSAC parameters
        self.min_matches = 10
        self.ransac_threshold = 0.01  # meters

    def reset(self) -> None:
        """Reset odometry state."""
        self.prev_rgb = None
        self.prev_depth = None
        self.position = np.array([0.0, 0.0])
        self.heading = 0.0

    def estimate_motion(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate camera motion from consecutive RGB-D frames.

        Args:
            rgb: Current RGB image (H, W, 3), uint8
            depth: Current depth image (H, W), normalized [0, 1]

        Returns:
            position: Current position (x, y) in meters
            heading: Current heading in radians
        """
        # First frame - just store and return initial pose
        if self.prev_rgb is None:
            self.prev_rgb = rgb.copy()
            self.prev_depth = depth.copy()
            return self.position.copy(), self.heading

        # Convert to grayscale for feature detection
        # gray_prev = cv2.cvtColor(self.prev_rgb, cv2.COLOR_RGB2GRAY)
        # gray_curr = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Detect and match features
        kp1, des1 = self.orb.detectAndCompute(self.prev_rgb, None)
        kp2, des2 = self.orb.detectAndCompute(rgb, None)
        # Handle case where no features are detected
        if des1 is None or des2 is None or len(des1) < self.min_matches:
            print("Warning: Insufficient features detected for VO, ")
            self.prev_rgb = rgb.copy()
            self.prev_depth = depth.copy()
            return self.position.copy(), self.heading

        # Match features
        matches = self.matcher.match(des1, des2)
        if len(matches) < self.min_matches:
            print("Warning: Insufficient feature matches for VO, len: ", len(matches))
            self.prev_rgb = rgb.copy()
            self.prev_depth = depth.copy()
            return self.position.copy(), self.heading

        # Sort matches by distance and take the best ones
        matches = sorted(matches, key=lambda x: x.distance)[: min(len(matches), 500)]

        # Extract 3D points from matched features
        pts1_3d = []
        pts2_3d = []

        h, w = depth.shape
        cx, cy = w / 2.0, h / 2.0

        for m in matches:
            # Get pixel coordinates
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt

            # Get depth values (normalized depth needs to be scaled)
            # Assuming depth range is normalized, we use a reasonable scale
            d1 = self.prev_depth[int(v1), int(u1)] * 10.0  # scale to meters
            d2 = depth[int(v2), int(u2)] * 10.0

            # Filter invalid depths
            if d1 <= 0.01 or d1 >= 9.9 or d2 <= 0.01 or d2 >= 9.9:
                continue

            # Back-project to 3D (camera coordinates: z-forward, x-right, y-down)
            x1 = (u1 - cx) * d1 / self.fx
            y1 = (v1 - cy) * d1 / self.fy
            pts1_3d.append([x1, y1, d1])

            x2 = (u2 - cx) * d2 / self.fx
            y2 = (v2 - cy) * d2 / self.fy
            pts2_3d.append([x2, y2, d2])

        # Need sufficient 3D correspondences
        if len(pts1_3d) < self.min_matches:
            print(f"Warning: Only {len(pts1_3d)} valid 3D correspondences")
            self.prev_rgb = rgb.copy()
            self.prev_depth = depth.copy()
            return self.position.copy(), self.heading

        pts1_3d = np.array(pts1_3d, dtype=np.float64)
        pts2_3d = np.array(pts2_3d, dtype=np.float64)

        # Estimate rigid transformation using RANSAC
        try:
            success, transform, inliers = cv2.estimateAffine3D(
                pts1_3d, pts2_3d, ransacThreshold=self.ransac_threshold, confidence=0.99
            )

            if not success or inliers is None or inliers.sum() < self.min_matches:
                print("Warning: Transformation estimation failed or insufficient inliersm inliers: ", inliers.sum() if inliers is not None else 0)
                self.prev_rgb = rgb.copy()
                self.prev_depth = depth.copy()
                return self.position.copy(), self.heading

            # Extract rotation and translation from affine transform (3x4 matrix)
            R = transform[:3, :3]
            t = transform[:3, 3]

            # Ensure rotation matrix is valid
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt

            # Extract motion in camera frame
            # Camera coordinates: z is forward, x is right, y is down
            delta_forward = t[2]  # z component (forward/backward)
            delta_right = t[0]  # x component (left/right)

            # Extract yaw rotation (rotation around y-axis)
            delta_yaw = np.arctan2(R[0, 2], R[2, 2])

            # Update global pose
            self.heading += delta_yaw
            self.heading = np.arctan2(np.sin(self.heading), np.cos(self.heading))  # Normalize to [-pi, pi]

            # Transform camera-frame motion to world frame
            cos_h = np.cos(self.heading)
            sin_h = np.sin(self.heading)

            # Update position (map z-forward and x-right to world x-y)
            self.position[0] += delta_forward * cos_h - delta_right * sin_h
            self.position[1] += delta_forward * sin_h + delta_right * cos_h

            print(
                f"VO: Moved ({delta_forward:.3f}, {delta_right:.3f})m, "
                f"rotated {np.rad2deg(delta_yaw):.1f}°, "
                f"inliers: {inliers.sum()}/{len(pts1_3d)}"
            )

        except Exception as e:
            print(f"Warning: VO estimation error: {e}")

        # Update previous frame
        self.prev_rgb = rgb.copy()
        self.prev_depth = depth.copy()

        return self.position.copy(), self.heading


class BehaviorMixin:
    """
    This Python mixin contains code for running ITMPolicyV3 in BEHAVIOR-1K simulator.
    It uses simple visual odometry for localization with a single RGB-D camera.
    """

    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    _load_yolo: bool = False
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}

    def __init__(
        self: Union["BehaviorMixin", ITMPolicyV3],
        camera_fx: float = 320.0,
        camera_fy: float = 320.0,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize BehaviorMixin.

        Args:
            camera_fx: Camera focal length in x direction (pixels)
            camera_fy: Camera focal length in y direction (pixels)
            min_depth: Minimum valid depth value (meters)
            max_depth: Maximum valid depth value (meters)
        """
        super().__init__(sync_explored_areas=True, *args, **kwargs)  # type: ignore

        # Camera parameters
        self._camera_fx = camera_fx
        self._camera_fy = camera_fy
        self._min_depth = min_depth
        self._max_depth = max_depth

        # Initialize visual odometry
        # This can be easily replaced with a more sophisticated SLAM system
        self._vo = SimpleVisualOdometry(fx=camera_fx, fy=camera_fy)

        print(f"BehaviorMixin initialized with camera params: fx={camera_fx}, fy={camera_fy}")

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self._done_initializing = not self._num_steps < 1 # type: ignore
        action = [2]
    
        return torch.tensor([action], dtype=torch.float32)
        
    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> Any:
        """Create policy from config."""
        policy_config: VLFMConfig = config.policy
        kwargs = {k: policy_config[k] for k in VLFMConfig.kwaarg_names}  # type: ignore

        # Add camera-specific parameters from config
        if hasattr(policy_config, "camera_fx"):
            kwargs["camera_fx"] = policy_config.camera_fx
        if hasattr(policy_config, "camera_fy"):
            kwargs["camera_fy"] = policy_config.camera_fy
        if hasattr(policy_config, "min_depth"):
            kwargs["min_depth"] = policy_config.min_depth
        if hasattr(policy_config, "max_depth"):
            kwargs["max_depth"] = policy_config.max_depth

        return cls(**kwargs)

    def act(
        self: Union["BehaviorMixin", ITMPolicyV3],
        observations: Dict[str, Any],
        rnn_hidden_states: Union[Tensor, Any],
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Get action from policy.

        Args:
            observations: Dictionary containing 'rgb', 'depth', and 'objectgoal'
            rnn_hidden_states: RNN hidden states (not used)
            prev_actions: Previous actions (not used)
            masks: Episode continuation mask
            deterministic: Whether to use deterministic action selection

        Returns:
            action: Tensor of shape (1, 2) with [angular_vel, linear_vel]
            rnn_hidden_states: Updated RNN hidden states
        """
        # Update non-COCO caption if needed
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = observations["objectgoal"] + " . " + self._non_coco_caption

        # Call parent class act method
        parent_cls: ITMPolicyV3 = super()  # type: ignore
        action, rnn_hidden_states = parent_cls.act(observations, rnn_hidden_states, prev_actions, masks, deterministic)
        
        
        action = self.action_transfer(action)
        
        return action, rnn_hidden_states, self._policy_info

    def action_transfer(self, action: Tensor):
        
        # STOP = torch.tensor([[0]], dtype=torch.long)
        # MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
        # TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
        # TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
        action = action.cpu()
        
        if action == torch.tensor([[0]]):
            return [0.0, 0.0] 
        elif action == torch.tensor([[1]]):
            return [0.5, 0.0]         
        elif action == torch.tensor([[2]]):
            return [0.3, 0.5]
        elif action == torch.tensor([[3]]):
            return [0.3,-0.5]
        

    def _reset(self: Union["BehaviorMixin", ITMPolicyV3]) -> None:
        """Reset policy state."""
        parent_cls: ITMPolicyV3 = super()  # type: ignore
        parent_cls._reset()

        # Reset visual odometry
        self._vo.reset()

        print("Visual odometry reset")

    def _cache_observations(self: Union["BehaviorMixin", ITMPolicyV3], observations: Dict[str, Any]) -> None:
        """
        Cache observations and estimate robot pose using visual odometry.

        Expected observations format:
        {
            'rgb': np.ndarray (H, W, 3) uint8,
            'depth': np.ndarray (H, W) float, normalized [0, 1] or in meters,
            'objectgoal': str
        }

        Args:
            observations: Dictionary containing RGB, depth, and object goal
        """
        if len(self._observations_cache) > 0:
            return

        # Extract observations
        rgb = observations["rgb"]  # (H, W, 3) uint8
        depth = observations["depth"]  # (H, W) float

        # Ensure depth is in correct format [0, 1]
        if depth.max() > 1.0:
            # Assume depth is in meters, normalize it
            depth = np.clip(depth, self._min_depth, self._max_depth)
            depth_normalized = (depth - self._min_depth) / (self._max_depth - self._min_depth)
        else:
            depth_normalized = depth.copy()

        # Estimate robot pose using visual odometry
        # Note: estimate_motion() returns accumulated pose (not deltas)
        use_vo = False
        if use_vo == True:
            robot_xy, robot_heading = self._vo.estimate_motion(rgb, depth_normalized)
        else:
            robot_xy = np.array(observations["robot_pos"])[:2] # x,y,z
            x, y, z, w = np.array(observations["robot_ori"]) # 4元数
            robot_heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
            

        print(f"VO pose: position=({robot_xy[0]:.2f}, {robot_xy[1]:.2f}), " f"heading={np.rad2deg(robot_heading):.1f}°")

        # Build transformation matrix (camera to world frame)
        # Camera is at robot_xy with heading robot_heading
        cos_h = np.cos(robot_heading)
        sin_h = np.sin(robot_heading)

        tf_camera_to_episodic = observations.get("tf_camera_to_episodic", None)
        
        

        # Calculate field of view
        h, w = depth.shape
        fov = 2 * np.arctan(w / (2 * self._camera_fx))

        # Update obstacle map with current observations
        self._obstacle_map: ObstacleMap
        self._obstacle_map.update_map(
            depth_normalized,
            tf_camera_to_episodic,
            self._min_depth,
            self._max_depth,
            self._camera_fx,
            self._camera_fy,
            fov,
            explore=True,  # Mark visible area as explored
            update_obstacles=True,  # Update obstacle locations
        )

        # Update agent trajectory for visualization
        self._obstacle_map.update_agent_traj(robot_xy, robot_heading)

        # Get frontiers for exploration
        frontiers = self._obstacle_map.frontiers
        # print(f"Found {len(frontiers)} frontier points")

        # Prepare depth tensor for PointNav
        depth_tensor = torch.from_numpy(depth_normalized).reshape(1, h, w, 1)
        if torch.cuda.is_available():
            depth_tensor = depth_tensor.to("cuda")

        # Cache all observations in the expected format
        self._observations_cache = {
            "frontier_sensor": frontiers,  # (N, 2) array of frontier points
            "nav_depth": depth_tensor,  # (1, H, W, 1) tensor for PointNav
            "robot_xy": robot_xy,  # (2,) position in meters
            "robot_heading": robot_heading,  # float, heading in radians
            "object_map_rgbd": [
                # List of tuples: (rgb, depth, tf, min_depth, max_depth, fx, fy)
                (
                    rgb,
                    depth_normalized,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fx,
                    self._camera_fy,
                )
            ],
            "value_map_rgbd": [
                # List of tuples: (rgb, depth, tf, min_depth, max_depth, fov)
                (rgb, depth_normalized, tf_camera_to_episodic, self._min_depth, self._max_depth, fov)
            ],
        }

    def _get_policy_info(
        self: Union["BehaviorMixin", ITMPolicyV3],
        detections: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get policy info including VO pose."""
        info = super()._get_policy_info(detections)
        info["vo_position"] = self._observations_cache.get("robot_xy", None)
        info["vo_heading"] = self._observations_cache.get("robot_heading", None)
        return info

@dataclass
class BehaviorConfig(DictConfig):
    """Configuration for BehaviorITMPolicyV3"""

    policy: VLFMConfig = VLFMConfig()


class BehaviorITMPolicyV3(BehaviorMixin, ITMPolicyV3):
    """
    ITMPolicyV3 adapted for BEHAVIOR-1K simulator with visual odometry.

    This policy uses a single RGB-D camera and estimates robot pose using
    simple visual odometry. The VO implementation can be easily replaced
    with more sophisticated SLAM systems.

    Example usage:
        policy = BehaviorITMPolicyV3(
            camera_fx=320.0,
            camera_fy=320.0,
            min_depth=0.1,
            max_depth=10.0,
            text_prompt="This looks like a target_object | This looks promising",
            # ... other ITMPolicyV3 parameters
        )

        # In your simulation loop:
        obs = {
            'rgb': rgb_image,  # (H, W, 3) uint8
            'depth': depth_image,  # (H, W) float
            'objectgoal': 'chair'
        }
        action, _ = policy.act(obs, None, None, masks, deterministic=True)
    """

    pass

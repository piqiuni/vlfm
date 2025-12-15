import numpy as np
import open3d as o3d
import sys

from vlfm.utils.geometry_utils import transform_points


def get_point_cloud(depth_image: np.ndarray, mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """
    Calculates the 3D coordinates (x, y, z) of points in the depth image based on
    the horizontal field of view (HFOV), the image width and height, the depth values,
    and the pixel x and y coordinates.

    Args:
        depth_image (np.ndarray): 2D depth image.
        mask (np.ndarray): 2D binary mask identifying relevant pixels.
        fx (float): Focal length in the x direction.
        fy (float): Focal length in the y direction.

    Returns:
        np.ndarray: Array of 3D coordinates (x, y, z) of the points in the image plane.
    """
    v, u = np.where(mask)
    r = depth_image[v, u]  # 假设这里读取的是径向距离 (Radial Distance)
    
    cx = depth_image.shape[1] // 2
    cy = depth_image.shape[0] // 2

    # 1. 计算像素在归一化平面上的坐标 (x_norm, y_norm)
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy

    # 2. 修正深度计算
    # 如果 r 是径向距离 (sqrt(x^2 + y^2 + z^2))
    # 那么 z = r / sqrt(1 + x_norm^2 + y_norm^2)
    # 
    # 如果你的 depth 已经是 Z-depth，则不需要除以这个系数。
    # 但既然你遇到了弯曲，说明很可能需要这个校正。
    scale_factor = np.sqrt(1 + x_norm**2 + y_norm**2)
    z = r / scale_factor 

    # 3. 恢复 x, y
    x = x_norm * z
    y = y_norm * z
    
    # VLFM convention: X=Forward(Z), Y=Left(-X), Z=Up(-Y)
    cloud = np.stack((z, -x, -y), axis=-1)

    return cloud

def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]

def main():
    depth_path = "vis_debug/scaled_depth.npy"
    depth = np.load(depth_path)
    
    # Create a mask for valid depth values
    mask = depth > 0
    fx = 300.0  # Example focal length in x
    fy = 300.0  # Example focal length in y
    
    print(f"Generating point cloud with fx={fx}, fy={fy}...")
    cloud_points = get_point_cloud(depth, mask, fx, fy)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_points)
    
    # Add coordinate frame (Red=X, Green=Y, Blue=Z)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    print(f"Visualizing {len(cloud_points)} points. Close the window to continue...")
    o3d.visualization.draw_geometries([pcd, coord_frame], 
                                            window_name=f"Point Cloud (fx={fx}, fy={fy})",
                                            width=1024, height=768)

    tf_camera_to_episodic = np.load("vis_debug/tf_camera_to_episodic.npy")
    
    # theta = np.radians(90)
    # c, s = np.cos(theta), np.sin(theta)
    # rot_y = np.array([
    #     [c, 0, s, 0],
    #     [0, 1, 0, 0],
    #     [-s, 0, c, 0],
    #     [0, 0, 0, 1]
    # ])
    
    # # 右乘: 相对于相机自身坐标系旋转 (修正相机朝向)
    # tf_camera_to_episodic = tf_camera_to_episodic @ rot_y
    
    # theta = np.radians(-90)
    # c, s = np.cos(theta), np.sin(theta)
    # rot_x = np.array([
    #     [1, 0, 0, 0],
    #     [0, c, -s, 0],
    #     [0, s, c, 0],
    #     [0, 0, 0, 1]
    # ])
    # tf_camera_to_episodic = tf_camera_to_episodic @ rot_x
    
    # theta = np.radians(90)
    # c, s = np.cos(theta), np.sin(theta)
    # rot_z = np.array([
    #     [c, -s, 0, 0],
    #     [s, c, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])
    # tf_camera_to_episodic = tf_camera_to_episodic @ rot_z
    
    print(tf_camera_to_episodic)
    point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, cloud_points)
    
    
    pcd_episodic = o3d.geometry.PointCloud()
    pcd_episodic.points = o3d.utility.Vector3dVector(point_cloud_episodic_frame)
    o3d.io.write_point_cloud("vis_debug/test_point_cloud_episodic.ply", pcd_episodic)
    
    
    obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, 0.15, 0.88)
    
    # 1. 创建世界坐标系 (原点)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # 2. 创建相机坐标系，并变换到相机当前的位置
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    camera_frame.transform(tf_camera_to_episodic)

    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(point_cloud_episodic_frame)
    
    # 3. 在可视化列表中加入 camera_frame
    print("Visualizing Global Point Cloud with Camera Frame...")
    o3d.visualization.draw_geometries([global_pcd, world_frame, camera_frame], 
                                            window_name=f"Global Map with Camera Pose",
                                            width=1024, height=768)
    
    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obstacle_cloud)
    
    # 同样可以在障碍物视图中加入相机坐标系
    o3d.visualization.draw_geometries([obs_pcd, world_frame, camera_frame], 
                                            window_name=f"Obstacle Cloud with Camera Pose",
                                            width=1024, height=768)


        
if __name__ == "__main__":
    main()
import numpy as np
import open3d as o3d
import os


class PointCloudProcessor:
    def __init__(self, calib_path, voxel_size=0.05):
        """
        点云处理模块初始化。
        :param calib_path: 标定文件路径，用于加载投影矩阵。
        :param voxel_size: 体素降采样的大小。
        """
        self.calib_path = calib_path
        self.voxel_size = voxel_size
        self.calibration_data = self.load_calibration_data()

    def load_calibration_data(self):
        """
        加载标定数据，用于 LiDAR 到相机的投影。
        :return: 包含标定矩阵的字典。
        """
        calib = {}
        with open(self.calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()]).reshape(-1, 4)
        return calib

    def read_point_cloud(self, file_path):
        """
        读取二进制格式的点云文件。
        :param file_path: 点云文件路径。
        :return: Numpy 数组格式的点云数据。
        """
        point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return point_cloud

    def voxel_downsample(self, point_cloud):
        """
        对点云数据进行体素降采样。
        :param point_cloud: 原始点云数据 (N x 3)。
        :return: 降采样后的点云数据。
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return np.asarray(downsampled_pcd.points)

    def project_to_image(self, point_cloud):
        """
        使用标定矩阵将点云投影到相机图像平面。
        :param point_cloud: 原始点云数据。
        :return: 投影后的像素坐标及深度。
        """
        # 获取投影矩阵
        P = self.calibration_data['P2']  # 投影矩阵
        Tr_velo_to_cam = self.calibration_data['Tr_velo_to_cam']  # LiDAR 到相机的转换矩阵

        # 扩展点云以匹配矩阵乘法
        points_hom = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
        points_cam = np.dot(Tr_velo_to_cam, points_hom.T).T  # 转换到相机坐标系
        points_proj = np.dot(P, points_cam.T).T  # 投影到图像平面

        # 归一化齐次坐标
        points_proj[:, 0] /= points_proj[:, 2]
        points_proj[:, 1] /= points_proj[:, 2]

        # 仅保留可见点
        mask = (points_proj[:, 2] > 0)  # 深度必须大于 0
        points_proj = points_proj[mask]

        return points_proj[:, :2], points_proj[:, 2]  # 返回像素坐标和深度

    def process_point_cloud(self, file_path):
        """
        完整的点云处理流程，包括读取、降采样、和投影。
        :param file_path: 点云文件路径。
        :return: 降采样后的点云及其投影结果。
        """
        raw_cloud = self.read_point_cloud(file_path)
        downsampled_cloud = self.voxel_downsample(raw_cloud)
        projected_points, depths = self.project_to_image(downsampled_cloud)
        return downsampled_cloud, projected_points, depths


# 测试代码
if __name__ == "__main__":
    processor = PointCloudProcessor(
        calib_path="E:\\pointnet\\data_object_calib\\training\\calib\\000000.txt"
    )
    pc_file = "E:\\pointnet\\data_object_velodyne\\training\\velodyne\\000000.bin"
    cloud, proj_points, depths = processor.process_point_cloud(pc_file)

    print(f"降采样点云数量: {cloud.shape[0]}")
    print(f"投影点数量: {proj_points.shape[0]}")

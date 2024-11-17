import cv2
import os

import numpy as np


class DatasetLoader:
    def __init__(self, velodyne_path, image_path, label_path, calib_path):
        self.velodyne_path = velodyne_path
        self.image_path = image_path
        self.label_path = label_path
        self.calib_path = calib_path

    def load_data(self, frame_id):
        """
        加载指定帧的点云和相关数据
        """
        try:
            # 加载点云文件
            velodyne_file = os.path.join(self.velodyne_path, f"{frame_id}.bin")
            if not os.path.exists(velodyne_file):
                raise FileNotFoundError(f"点云文件不存在: {velodyne_file}")

            point_cloud = self._load_velodyne(velodyne_file)

            # 加载图像文件
            image_file = os.path.join(self.image_path, f"{frame_id}.png")
            image = None
            if os.path.exists(image_file):
                image = self._load_image(image_file)

            # 加载标签文件
            label_file = os.path.join(self.label_path, f"{frame_id}.txt")
            labels = None
            if os.path.exists(label_file):
                labels = self._load_labels(label_file)

            # 加载校准文件
            calib_file = os.path.join(self.calib_path, f"{frame_id}.txt")
            calib = None
            if os.path.exists(calib_file):
                calib = self._load_calibration(calib_file)

            return point_cloud, image, labels, calib
        except Exception as e:
            raise RuntimeError(f"加载数据失败: {str(e)}")

    def _load_velodyne(self, velodyne_file):
        """
        加载 .bin 格式的点云数据
        """
        try:
            # 使用 numpy 从 .bin 文件中加载点云数据
            point_cloud = np.fromfile(velodyne_file, dtype=np.float32).reshape(-1, 4)
            return point_cloud
        except Exception as e:
            raise RuntimeError(f"加载点云文件失败: {velodyne_file}, 错误: {str(e)}")

    def _load_image(self, image_file):
        """
        加载图像文件
        """
        # 这里可以使用 OpenCV 或 PIL 加载图像
        import cv2
        image = cv2.imread(image_file)
        return image

    def _load_labels(self, label_file):
        """
        加载标签文件
        """
        with open(label_file, 'r') as f:
            labels = f.readlines()
        return labels

    def _load_calibration(self, calib_file):
        """
        加载校准文件
        """
        with open(calib_file, 'r') as f:
            calib = f.readlines()
        return calib

# 测试代码
if __name__ == "__main__":
    loader = DatasetLoader(
        velodyne_path="E:/pointnet/data_object_velodyne/training/velodyne",
        image_path = "E:/pointnet/data_object_image_2/training/image_2",
        label_path = "E:/pointnet/data_object_label_2/training/label_2",
        calib_path = "E:/pointnet/data_object_calib/training/calib"
    )
    point_cloud, image, labels, calib = loader.load_data("000000")
    print(f"点云点数量: {point_cloud.shape[0]}")
    print(f"图像尺寸: {image.shape}")
    print(f"标注数量: {len(labels)}")

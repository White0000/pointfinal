import os
import numpy as np
from models.yolo.yolo_model import YOLOv8Model
from models.transformer.transformer_model import TransformerTrajectoryModel
from visualization.pointcloud_visualizer import PointCloudVisualizer
from preprocessing.dataset_loader import DatasetLoader
import torch
import torch.nn as nn
import torch.optim as optim


def load_transformer_model(model_path, input_dim=4, embed_dim=128, num_heads=4, num_layers=2, output_dim=10):
    """
    加载训练好的 Transformer 模型。
    :param model_path: Transformer 模型的权重路径。
    :param input_dim: 输入特征维度。
    :param embed_dim: 嵌入维度。
    :param num_heads: 多头注意力头数。
    :param num_layers: Transformer 层数。
    :param output_dim: 输出特征维度。
    :return: 加载好的 Transformer 模型。
    """
    model = TransformerTrajectoryModel(input_dim, embed_dim, num_heads, num_layers, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model


def main():
    """
    项目主入口。
    整合 YOLOv8 检测、Transformer 轨迹预测和多模态可视化。
    """
    # 路径配置
    point_cloud_path = "E:/pointnet/data_object_velodyne/training/velodyne"
    image_path = "E:/pointnet/data_object_image_2/training/image_2"
    label_path = "E:/pointnet/data_object_label_2/training/label_2"
    calib_path = "E:/pointnet/data_object_calib/training/calib"
    transformer_model_path = "./transformer_model.pth"
    yolo_model_path = "yolov8s.pt"

    # 初始化数据加载器
    dataset_loader = DatasetLoader(
        velodyne_path=point_cloud_path,
        image_path=image_path,
        label_path=label_path,
        calib_path=calib_path
    )

    # 初始化 YOLOv8 模型
    yolo_model = YOLOv8Model(model_path=yolo_model_path, num_classes=2)

    # 加载 Transformer 模型
    transformer_model = load_transformer_model(transformer_model_path)

    # 初始化可视化工具
    visualizer = PointCloudVisualizer()

    # 加载测试数据（以编号为 "000000" 的数据为例）
    frame_id = "000000"
    point_cloud, image, labels, calib = dataset_loader.load_data(frame_id)

    # YOLOv8 检测
    print("开始 YOLOv8 检测...")
    detection_results = yolo_model.predict(image_path=os.path.join(image_path, f"{frame_id}.png"))
    print(f"检测完成，结果：{detection_results}")

    # 提取边界框（模拟提取，实际应从 YOLO 检测结果解析）
    boxes = [
        [0.2, 0.2, 0.5, 0.5],  # 示例边界框 1
        [0.6, 0.6, 0.8, 0.8]   # 示例边界框 2
    ]

    # 提取轨迹输入特征（模拟，实际应基于检测结果和点云数据生成）
    input_features = np.random.rand(10, 4)  # 假设有 10 帧历史数据

    # Transformer 轨迹预测
    print("开始 Transformer 轨迹预测...")
    predicted_trajectory = transformer_model(torch.tensor(input_features).unsqueeze(1))
    print(f"轨迹预测完成，结果：{predicted_trajectory}")

    # 可视化
    print("启动可视化...")
    visualizer.render_point_cloud(point_cloud[:, :3])  # 点云渲染
    visualizer.render_2d_boxes(point_cloud, boxes)  # 边界框渲染
    visualizer.render_trajectory([predicted_trajectory.detach().numpy()])  # 轨迹渲染
    visualizer.run()  # 启动可视化界面


if __name__ == "__main__":
    main()

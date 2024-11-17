import torch
import torch.nn as nn
import math


class TransformerTrajectoryModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim, sequence_length, future_steps):
        super(TransformerTrajectoryModel, self).__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, output_dim)
        self.future_steps = future_steps

    def forward(self, x, return_dynamics=False):
        batch_size, seq_len, _ = x.size()
        positional_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.embedding(x) + positional_encoding

        # 使用 TransformerEncoder 处理输入
        encoded_output = self.transformer_encoder(x)

        # 预测未来位置点
        predicted_positions = self.fc(encoded_output)

        # 裁剪结果，仅保留最后 future_steps 的时间步
        predicted_positions = predicted_positions[:, -self.future_steps:, :]

        if not return_dynamics:
            return predicted_positions

        # 确保输出的速度和加速度形状正确
        if predicted_positions.shape[1] < 3:
            raise ValueError("输入序列长度不足，无法计算加速度")

        # 计算速度和加速度
        predicted_velocities = predicted_positions[:, 1:, :] - predicted_positions[:, :-1, :]
        predicted_accelerations = predicted_velocities[:, 1:, :] - predicted_velocities[:, :-1, :]

        # 返回预测结果
        return predicted_positions, predicted_velocities, predicted_accelerations

    def compute_mse(predicted, ground_truth):
        """
        计算均方误差 (MSE)
        :param predicted: 模型预测的轨迹 (N, T, D)
        :param ground_truth: 真实轨迹 (N, T, D)
        :return: MSE 值
        """
        mse = torch.mean((predicted - ground_truth) ** 2)
        return mse.item()

    def compute_iou(box1, box2):
        """
        计算两个检测框的 IoU
        :param box1: [x1, y1, x2, y2]
        :param box2: [x1, y1, x2, y2]
        :return: IoU 值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union



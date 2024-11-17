import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from project_root.models.transformer.transformer_model import TransformerTrajectoryModel
from project_root.preprocessing.dataset_loader import DatasetLoader
from project_root.utils.metrics import (
    compute_mse,
    compute_mae,
    compute_rmse,
    compute_iou,
    compute_f1_score,
    save_training_curve,
    save_results_to_csv,
)
import time

def summarize_training_results(metrics, num_epochs, log_callback=None):
    """
    总结训练结果并输出到控制台或 Qt 控制台。
    :param metrics: 包含训练过程中各项指标的字典
    :param num_epochs: 训练的总轮数
    :param log_callback: Qt 控制台日志回调函数（可选）
    """
    summary = []
    summary.append("\n======================== 训练总结 ========================\n")
    summary.append(f"总训练周期数 (Epochs): {num_epochs}")
    summary.append(f"总训练时间 (Total Time): {sum(metrics['Training Time']):.2f}s")
    summary.append(f"每周期平均时间 (Avg Time per Epoch): {sum(metrics['Training Time']) / num_epochs:.2f}s\n")

    for metric_name in ['MSE', 'MAE', 'RMSE', 'IoU', 'F1 Score']:
        avg_metric = sum(metrics[metric_name]) / len(metrics[metric_name])
        last_metric = metrics[metric_name][-1]
        summary.append(f"指标: {metric_name}")
        summary.append(f"  平均值 (Average): {avg_metric:.4f}")
        summary.append(f"  最后一周期值 (Last Epoch): {last_metric:.4f}")
        summary.append("--------------------------------------------------------\n")

    summary.append("======================== 训练结束 ========================\n")

    # 控制台输出
    for line in summary:
        print(line)
        if log_callback:
            log_callback(line)  # 将日志写入 Qt 控制台



def train_transformer_model(data_dir, model_save_path, log_callback=None):
    """
    Transformer 模型训练代码
    """
    data_dir = "E:/pointnet/data_object_velodyne/training"
    model_save_path = "transformer_model.pth"
    curve_save_path = "training_curve.png"
    results_csv_path = "training_results.csv"
    seq_len = 10
    future_steps = 5

    # 数据加载
    dataset_loader = DatasetLoader(
        velodyne_path=os.path.join(data_dir, "velodyne"),
        image_path=os.path.join(data_dir, "image_2"),
        label_path=os.path.join(data_dir, "label_2"),
        calib_path=os.path.join(data_dir, "calib"),
    )

    # 模拟数据 (训练时替换为真实数据)
    dataset = [
        (
            torch.rand(seq_len, 4),
            torch.rand(future_steps, 3)
        )
        for _ in range(100)
    ]
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerTrajectoryModel(
        input_dim=4,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        output_dim=3,
        sequence_length=seq_len,
        future_steps=future_steps
    ).to(device)

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    num_epochs = 50
    results = []
    metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'IoU': [], 'F1 Score': [], 'Training Time': []}

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_mse, epoch_mae, epoch_rmse, epoch_iou, epoch_f1 = 0, 0, 0, 0, 0

        for batch_idx, (input_features, ground_truth_trajectory) in enumerate(dataloader):
            input_features, ground_truth_trajectory = input_features.to(device), ground_truth_trajectory.to(device)
            optimizer.zero_grad()

            # 使用 forward 预测
            predictions = model(input_features.float(), return_dynamics=False)

            # 计算损失
            mse_loss = criterion(predictions, ground_truth_trajectory.float())
            iou_loss = compute_iou(predictions, ground_truth_trajectory).mean()
            loss = 0.8 * mse_loss + 0.2 * (1 - iou_loss)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 计算指标
            batch_mse = compute_mse(predictions, ground_truth_trajectory)
            batch_mae = compute_mae(predictions, ground_truth_trajectory)
            batch_rmse = compute_rmse(predictions, ground_truth_trajectory)
            batch_iou = compute_iou(predictions, ground_truth_trajectory).mean().item()
            batch_f1 = compute_f1_score(predictions, ground_truth_trajectory)

            epoch_mse += batch_mse
            epoch_mae += batch_mae
            epoch_rmse += batch_rmse
            epoch_iou += batch_iou
            epoch_f1 += batch_f1

            if log_callback:
                log_callback(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], "
                             f"Loss: {loss.item():.4f}, MSE: {batch_mse:.4f}, MAE: {batch_mae:.4f}, "
                             f"RMSE: {batch_rmse:.4f}, IoU: {batch_iou:.4f}, F1 Score: {batch_f1:.4f}")

        # 记录每个 epoch 的平均值
        avg_mse = epoch_mse / len(dataloader)
        avg_mae = epoch_mae / len(dataloader)
        avg_rmse = epoch_rmse / len(dataloader)
        avg_iou = epoch_iou / len(dataloader)
        avg_f1 = epoch_f1 / len(dataloader)
        metrics['MSE'].append(avg_mse)
        metrics['MAE'].append(avg_mae)
        metrics['RMSE'].append(avg_rmse)
        metrics['IoU'].append(avg_iou)
        metrics['F1 Score'].append(avg_f1)
        metrics['Training Time'].append(time.time() - epoch_start_time)

        results.append([epoch + 1, avg_mse, avg_mae, avg_rmse, avg_iou, avg_f1])
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Time: {metrics['Training Time'][-1]:.2f}s")

    # 保存模型与结果
    torch.save(model.state_dict(), model_save_path)
    save_training_curve(metrics, curve_save_path)
    save_results_to_csv(results, results_csv_path)

    # 在训练代码结束时调用总结函数
    summarize_training_results(metrics, num_epochs, log_callback)
    print("训练完成，模型已保存。")


if __name__ == "__main__":
    train_transformer_model("E:/pointnet/data_object_velodyne/training", "transformer_model.pth")

import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

# ========================
# 指标计算函数
# ========================

def compute_mse(predictions, ground_truth):
    """计算均方误差 (MSE)"""
    return torch.mean((predictions - ground_truth) ** 2).item()

def compute_mae(predictions, ground_truth):
    """计算平均绝对误差 (MAE)"""
    return torch.mean(torch.abs(predictions - ground_truth)).item()

def compute_rmse(predictions, ground_truth):
    """计算均方根误差 (RMSE)"""
    return torch.sqrt(torch.mean((predictions - ground_truth) ** 2)).item()

def compute_iou(predictions, ground_truth, threshold=0.5):
    """计算交并比 (IoU)"""
    predictions = (predictions > threshold).float()
    ground_truth = (ground_truth > threshold).float()
    intersection = torch.sum(predictions * ground_truth)
    union = torch.sum(predictions) + torch.sum(ground_truth) - intersection
    return (intersection / union).item() if union > 0 else 0.0

def compute_f1_score(predictions, ground_truth, threshold=0.5):
    """计算F1分数"""
    predictions = (predictions > threshold).float()
    ground_truth = (ground_truth > threshold).float()
    tp = torch.sum(predictions * ground_truth)
    fp = torch.sum(predictions * (1 - ground_truth))
    fn = torch.sum((1 - predictions) * ground_truth)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)).item() if (precision + recall) > 0 else 0.0

def compute_dynamic_loss_weights(metrics, target_metric='IoU'):
    """
    动态调整损失权重
    :param metrics: 当前训练的所有指标字典
    :param target_metric: 优先优化的目标指标
    :return: 动态权重
    """
    if target_metric in metrics and len(metrics[target_metric]) > 1:
        recent_value = metrics[target_metric][-1]
        if recent_value < 0.5:
            # 如果目标指标表现较差，提高其损失权重
            return 0.6, 0.4  # 比例：MSE权重, IoU权重
        else:
            # 如果目标指标表现较好，平衡损失
            return 0.8, 0.2
    return 0.8, 0.2  # 默认权重

# ========================
# 可视化函数
# ========================

def save_training_curve(metrics, save_path):
    """保存训练曲线图"""
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ========================
# 结果保存函数
# ========================

def save_results_to_csv(results, save_path):
    """将结果保存到CSV文件"""
    header = ['Epoch', 'MSE', 'MAE', 'RMSE', 'IoU', 'F1 Score', 'Dynamic Weight (MSE)', 'Dynamic Weight (IoU)']
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)

# ========================
# 新增：IoU损失函数
# ========================

def compute_iou(predictions, ground_truth, threshold=0.5):
    predictions = (predictions > threshold).float()
    ground_truth = (ground_truth > threshold).float()
    intersection = torch.sum(predictions * ground_truth, dim=1)
    union = torch.sum(predictions, dim=1) + torch.sum(ground_truth, dim=1) - intersection
    iou = intersection / union
    return iou  # 返回张量


import torch
from project_root.models.transformer.transformer_model import TransformerTrajectoryModel


def predict_trajectory(model_path, input_features):
    """
    使用训练好的 Transformer 模型预测目标轨迹。
    :param model_path: 模型权重路径。
    :param input_features: 输入特征（时间序列数据）。
    :return: 预测的未来轨迹。
    """
    # 模型加载
    input_dim = 4,
    embed_dim = 256,
    num_heads = 8,
    num_layers = 6,
    output_dim = 3,
    sequence_length = 10,
    future_steps = 5

    model = TransformerTrajectoryModel(input_dim, embed_dim, num_heads, num_layers, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 预测
    with torch.no_grad():
        input_features = torch.tensor(input_features).unsqueeze(1)  # 增加 batch 维度
        predicted_trajectory = model(input_features)
    return predicted_trajectory.numpy()


# 测试代码
if __name__ == "__main__":
    test_input = [[1, 2, 3, 0], [2, 3, 4, 1]]  # 示例输入
    result = predict_trajectory("transformer_model.pth", test_input)
    print("预测轨迹:", result)

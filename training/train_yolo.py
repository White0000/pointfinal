import os
from project_root.models.yolo.yolo_model import YOLOv8Model

def create_data_config(train_path, classes, output_path="./data.yaml"):
    """
    创建 YOLOv8 数据配置文件。如果没有验证集，验证路径默认与训练路径相同。
    :param train_path: 训练数据的图片目录路径。
    :param classes: 类别列表（如 ["Car", "Pedestrian"]）。
    :param output_path: 数据配置文件的保存路径。
    :return: 数据配置文件路径。
    """
    val_path = train_path  # 没有验证集时，用训练路径代替

    data_config = {
        "train": train_path,
        "val": val_path,
        "names": classes
    }

    # 修复 YAML 写入方式，避免 f-string 中的未转义反斜杠问题
    with open(output_path, "w") as f:
        f.write("train: {}\n".format(train_path))
        f.write("val: {}\n".format(val_path))
        f.write("names:\n")
        for cls in classes:
            f.write(f"  - {cls}\n")

    print(f"数据配置文件已保存到: {output_path}")
    return output_path


def main():
    """
    主函数：执行 YOLOv8 的训练流程。
    """
    # 定义训练数据路径
    train_images_path = os.path.join("E:", "pointnet", "data_object_image_2", "training", "image_2")
    label_classes = ["Car", "Pedestrian"]  # 目标类别

    # 创建 YOLO 数据配置文件
    data_config_path = create_data_config(
        train_path=train_images_path,
        classes=label_classes
    )

    # 初始化 YOLOv8 模型
    yolo_model = YOLOv8Model(
        model_path="yolov8s.pt",  # 使用默认 YOLOv8s 预训练模型
        num_classes=len(label_classes)
    )

    # 开始训练
    print("开始训练 YOLOv8 模型...")
    yolo_model.train(
        data_config=data_config_path,
        epochs=50,  # 训练轮数
        batch_size=16,  # 批量大小
        project_path="./runs/train"  # 结果保存路径
    )
    print("YOLOv8 模型训练完成！")


if __name__ == "__main__":
    main()

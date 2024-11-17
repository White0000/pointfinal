import cv2


class YOLOv8Model:
    def __init__(self, model_path=None, num_classes=2):
        """
        初始化 YOLOv8 模型。
        :param model_path: 预训练权重文件路径，如果为 None，则加载默认的 YOLOv8s 模型。
        :param num_classes: 需要检测的类别数量（默认为车辆和行人）。
        """
        from ultralytics import YOLO
        self.num_classes = num_classes
        self.model = YOLO(model_path if model_path else "yolov8s.pt")

    def train(self, data_config, epochs=50, batch_size=16, project_path="./runs/train"):
        """
        训练 YOLOv8 模型。
        :param data_config: 数据配置文件路径（包含训练和验证数据路径）。
        :param epochs: 训练的总轮数。
        :param batch_size: 每个批次的大小。
        :param project_path: 训练结果保存的目录。
        """
        self.model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            project=project_path,
            name="yolov8_train",
            imgsz=640,
            device=0  # 使用 GPU 训练
        )

    def predict(self, frame, conf=0.25):
        """
        使用 YOLOv8 模型检测视频帧。
        :param frame: 输入的图像帧 (BGR 格式)。
        :param conf: 检测置信度阈值。
        :return: 检测结果列表，每个检测结果格式为 [x1, y1, x2, y2, conf, cls]。
        """
        # 模型预测
        results = self.model.predict(source=frame, conf=conf, save=False, device=0)  # 指定 GPU/CPU

        detections = []

        # 提取检测结果
        for result in results:
            if hasattr(result, "boxes"):  # 确保结果中有检测框信息
                # 提取框坐标、置信度和类别标签
                boxes = result.boxes.xyxy.cpu().numpy()  # 检测框坐标
                confidences = result.boxes.conf.cpu().numpy()  # 检测框置信度
                classes = result.boxes.cls.cpu().numpy()  # 检测框类别

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confidences[i]
                    cls = classes[i]
                    detections.append([x1, y1, x2, y2, conf, cls])
            else:
                # 如果没有检测框，记录日志
                print("未检测到任何目标。")
        return detections

    def predict_and_save(self, video_path, save_path):
        """
        对视频进行检测，并使用 YOLO 的内置方法绘制检测框并保存结果。
        :param video_path: 输入视频路径。
        :param save_path: 输出视频保存路径。
        """
        try:
            # 打开输入视频
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置输出视频编码格式
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            print("开始视频检测...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO 模型预测
                results = self.model.predict(source=frame, conf=0.25, save=False, device=0)

                # 使用 YOLO 的内置绘图方法绘制检测框
                if results and len(results) > 0:
                    detected_frame = results[0].plot()
                    out.write(detected_frame)  # 写入视频

            cap.release()
            out.release()
            print(f"检测完成，视频已保存至: {save_path}")

        except Exception as e:
            print(f"视频检测失败: {e}")

    def predict_and_save_image(self, image_path, save_path):
        """
        对图像进行检测，并使用 YOLO 的内置方法绘制检测框。
        :param image_path: 输入图像路径。
        :param save_path: 检测后图像保存路径。
        :return: 检测后的图像（OpenCV 格式）。
        """
        try:
            # 读取输入图像
            frame = cv2.imread(image_path)

            # YOLO 模型预测
            results = self.model.predict(source=frame, conf=0.25, save=False)

            # 使用 YOLO 内置绘图方法绘制检测框
            if results and len(results) > 0:
                detected_image = results[0].plot()  # 使用 YOLO 内置绘制方法

                # 保存检测后的图像
                cv2.imwrite(save_path, detected_image)
                print(f"图像检测完成，结果已保存至: {save_path}")

                return detected_image
            else:
                print("未检测到任何目标。")
                return None

        except Exception as e:
            print(f"图像检测失败: {e}")
            return None





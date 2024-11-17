import glob
import os
import time
import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, QWidget, QLabel, QSplitter
from torch import layout

from project_root.models.yolo.yolo_model import YOLOv8Model
from project_root.models.transformer.transformer_model import TransformerTrajectoryModel
from project_root.visualization.pointcloud_visualizer import PointCloudVisualizer
from project_root.preprocessing.dataset_loader import DatasetLoader


class TransformerTrainingThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, data_dir, model_save_path, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.model_save_path = model_save_path

    def run(self):
        from project_root.training.train_transformer import train_transformer_model
        try:
            train_transformer_model(
                data_dir=self.data_dir,
                model_save_path=self.model_save_path,
                log_callback=self.log_signal.emit
            )
            self.log_signal.emit("Transformer 模型训练完成！")
        except Exception as e:
            self.log_signal.emit(f"Transformer 模型训练失败: {str(e)}")

class VideoPlayerThread(QThread):
    frame_signal = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()  # 播放完成信号

    def __init__(self, video_path, yolo_model, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.yolo_model = yolo_model  # YOLO 模型实例
        self.running = True  # 控制线程运行状态
        self.class_names = ["人", "车", "单车"]  # 类别名称映射，根据模型定义

    def run(self):
        try:
            # 检查视频文件路径
            if not os.path.exists(self.video_path):
                self.log_signal.emit(f"视频文件不存在：{self.video_path}")
                return

            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.log_signal.emit("无法打开视频文件，请检查文件路径或格式！")
                return

            # 获取帧率并计算时间间隔
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = 1.0 / fps if fps > 0 else 0.03  # 默认间隔 30ms

            self.log_signal.emit("视频播放线程已启动，请勿操作，YOLO 检测中...")

            while self.running:
                ret, frame = cap.read()
                if not ret:  # 视频播放结束
                    self.log_signal.emit("视频播放结束。")
                    self.finished_signal.emit()
                    break

                # 检查帧是否为空
                if frame is None or frame.size == 0:
                    self.log_signal.emit("读取到空帧，跳过。")
                    continue

                try:
                    # YOLO 检测并绘制结果
                    results = self.yolo_model.model.predict(frame, save=False, conf=0.25)
                    self.log_signal.emit(f"YOLO 结果: {results}")
                    if results and len(results) > 0:
                        detected_frame = results[0].plot()
                        if detected_frame is None:
                            self.log_signal.emit("检测帧为空，跳过该帧。")
                            continue

                        # 获取 YOLO 推理速度信息和检测结果
                        summary = results[0].speed  # 提取速度信息
                        detections = results[0].boxes if results else None

                        if detections is not None and len(detections) > 0:
                            detection_log = f"检测到 {len(detections)} 个目标"
                            self.log_signal.emit(detection_log)

                            # 遍历每个检测框
                            for box in detections:
                                coords = box.xyxy.cpu().numpy().flatten()
                                conf = box.conf.cpu().item()
                                cls = int(box.cls.cpu().item())

                                if len(coords) == 4:
                                    x1, y1, x2, y2 = coords
                                    label = f"类别: {self.class_names[cls] if cls < len(self.class_names) else '未知'}, " \
                                            f"置信度: {conf:.2f}, 框: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
                                    self.log_signal.emit(label)
                                else:
                                    self.log_signal.emit("检测框格式错误，跳过该目标。")
                        else:
                            self.log_signal.emit("未检测到目标。")
                    else:
                        self.log_signal.emit("YOLO 未返回任何结果，跳过该帧。")
                        detected_frame = frame  # 使用原始帧作为检测失败的显示

                    # 转换帧为 RGB 格式并发送信号
                    rgb_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
                    self.frame_signal.emit(qimg)

                except Exception as e:
                    self.log_signal.emit(f"帧处理失败：{str(e)}")
                    break

                # 控制播放速度
                time.sleep(frame_interval)

            cap.release()  # 释放资源
            self.log_signal.emit("视频播放线程已安全退出。")
        except Exception as e:
            self.log_signal.emit(f"视频播放线程异常：{str(e)}")

    def stop(self):
        self.running = False
        self.wait()  # 等待线程完全停止

class YoloDetectionThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, yolo_model, image_path):
        super().__init__()
        self.yolo_model = yolo_model
        self.image_path = image_path

    def run(self):
        try:
            results = self.yolo_model.predict(self.image_path)
            self.result_signal.emit(f"YOLOv8 检测完成！结果: {results}")
        except Exception as e:
            self.result_signal.emit(f"YOLOv8 检测失败: {str(e)}")


class TransformerPredictionThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, transformer_model, input_features):
        super().__init__()
        self.transformer_model = transformer_model
        self.input_features = input_features

    def run(self):
        try:
            predicted_trajectory = self.transformer_model(self.input_features)
            self.result_signal.emit(f"Transformer 预测完成！结果: {predicted_trajectory}")
        except Exception as e:
            self.result_signal.emit(f"Transformer 预测失败: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 & Transformer 多模态检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化模型和工具
        self.yolo_model = YOLOv8Model(model_path="yolov8s.pt", num_classes=2)
        self.transformer_model = None
        self.visualizer = PointCloudVisualizer()
        self.dataset_loader = None
        self.video_thread = None

        # 初始化界面
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        # 左侧：用于显示图片或视频
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid black;")
        self.preview_label.setFixedSize(600, 500)

        # 中间按钮区
        self.select_button = QPushButton("选择文件", self)
        self.select_button.clicked.connect(self.select_file)

        self.detect_button = QPushButton("YOLOv8 检测", self)
        self.detect_button.clicked.connect(self.detect_file)
        self.detect_button.setEnabled(False)

        self.visualize_button = QPushButton("启动可视化", self)
        self.visualize_button.clicked.connect(self.run_visualization)

        self.train_transformer_button = QPushButton("训练 Transformer 模型", self)
        self.train_transformer_button.clicked.connect(self.train_transformer)

        self.transformer_button = QPushButton("执行 Transformer 预测", self)  # 恢复按钮
        self.transformer_button.clicked.connect(self.run_transformer)  # 绑定事件

        self.load_data_button = QPushButton("加载数据", self)
        self.load_data_button.clicked.connect(self.load_data)

        # 日志区域
        self.log_area = QTextEdit(self)
        self.log_area.setReadOnly(True)

        # 按钮布局
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.visualize_button)
        button_layout.addWidget(self.train_transformer_button)
        button_layout.addWidget(self.transformer_button)
        button_layout.addWidget(self.load_data_button)

        # 右侧布局
        right_layout = QVBoxLayout()
        right_layout.addLayout(button_layout)
        right_layout.addWidget(self.log_area)

        # 主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.preview_label)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def log(self, message):
        self.log_area.append(message)
        self.log_area.ensureCursorVisible()

    def select_file(self):
        """选择文件并显示预览"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片或视频文件", "", "Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi)"
        )
        if not file_path:
            self.log("未选择文件。")
            return

        self.file_path = file_path
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in [".png", ".jpg", ".jpeg"]:
            self.show_image(file_path)
        elif file_extension in [".mp4", ".avi"]:
            self.play_video(file_path)
        else:
            self.log("不支持的文件格式！")
            self.file_path = None
            return

        self.log(f"已选择文件: {file_path}")
        self.detect_button.setEnabled(True)

    def show_image(self, image_path):
        """显示图片预览"""
        pixmap = QPixmap(image_path)
        self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio))

    def play_video(self, video_path):
        """播放视频预览并实时检测"""
        # 停止旧线程
        if self.video_thread and self.video_thread.isRunning():
            self.log("正在停止当前视频线程...")
            self.video_thread.stop()

        # 检查视频路径有效性
        if not os.path.exists(video_path):
            self.log(f"视频文件不存在：{video_path}")
            return

        # 创建新线程
        self.video_thread = VideoPlayerThread(video_path, self.yolo_model)
        self.video_thread.frame_signal.connect(self.update_video_frame)
        self.video_thread.log_signal.connect(self.log)
        self.video_thread.finished_signal.connect(self.on_video_finished)
        self.video_thread.start()

        self.log(f"开始播放视频并实时检测：{video_path}")

    def update_video_frame(self, qimg):
        """
        更新视频帧，并在 QLabel 中自适应原比例显示
        :param qimg: 当前帧的 QImage 格式
        """
        # 将 QImage 转换为 QPixmap
        pixmap = QPixmap.fromImage(qimg)

        # 根据 QLabel 的尺寸调整 Pixmap，保持宽高比
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),  # QLabel 的当前大小
            Qt.KeepAspectRatio,  # 保持宽高比
            Qt.SmoothTransformation  # 使用平滑缩放
        )

        # 在 QLabel 上显示
        self.preview_label.setPixmap(scaled_pixmap)

    def on_video_finished(self):
        """视频播放完成后的处理"""
        self.log("视频播放已完成！")

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.video_thread and self.video_thread.isRunning():
            self.log("正在停止视频线程...")
            self.video_thread.stop()
        event.accept()  # 确保窗口正常关闭

    def detect_file(self):
        """YOLOv8 检测"""
        if not hasattr(self, 'file_path') or not self.file_path:
            self.log("未选择文件，无法进行检测！")
            return

        file_extension = os.path.splitext(self.file_path)[1].lower()
        try:
            if file_extension in [".png", ".jpg", ".jpeg"]:
                self.log("开始检测图片...")

                # YOLO 模型预测并返回检测后的图像
                detected_image = self.yolo_model.predict_and_save_image(self.file_path,
                                                                  save_path="./runs/detect/detected_image.jpg")

                if detected_image is not None:
                    self.log(f"检测完成！标签详细结果已保存至runs/detect/..")
                    self.show_image_cv2(detected_image)  # 显示图像
                else:
                    self.log("检测失败，未生成结果图像。")
            elif file_extension in [".mp4", ".avi"]:
                self.log("开始检测视频...")
                self.yolo_model.predict(self.file_path)  # 视频检测（未实时显示）
                self.log(f"视频检测完成！结果保存在: ./runs/detect")
            else:
                self.log("不支持的文件格式！")
        except Exception as e:
            self.log(f"检测失败: {str(e)}")

    def show_image_cv2(self, image):
        """
        将 OpenCV 图像按原比例显示到 QLabel 中。
        :param image: OpenCV 格式的图像。
        """
        try:
            # 将 BGR 图像转换为 RGB 格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 转换为 QImage 格式
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 根据 QLabel 大小调整图像显示比例
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,  # 保持宽高比
                Qt.SmoothTransformation  # 平滑缩放
            )

            # 在 QLabel 中显示图像
            self.preview_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.log(f"显示图像失败: {str(e)}")

    def find_detected_image(self, detect_dir):
        """
        在检测结果目录中查找图片文件
        :param detect_dir: 检测结果目录路径
        :return: 检测到的图片文件路径（如果存在）
        """
        # 支持的图片扩展名
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]

        # 在目录中搜索图片文件
        for ext in image_extensions:
            image_files = glob.glob(os.path.join(detect_dir, ext))
            if image_files:
                # 返回最新生成的图片文件
                return max(image_files, key=os.path.getmtime)

        return None

    def run_visualization(self):
        self.log("启动可视化功能...")

    def train_transformer(self):
        self.log("启动 Transformer 模型训练...")

    def load_data(self):
        self.log("加载数据完成...")

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()

    def train_transformer(self):
        data_dir = QFileDialog.getExistingDirectory(self, "选择训练数据目录")
        if not data_dir:
            self.log("未选择训练数据目录。")
            return

        model_save_path = QFileDialog.getSaveFileName(self, "保存 Transformer 模型", "transformer_model.pth")[0]
        if not model_save_path:
            self.log("未指定模型保存路径。")
            return

        self.log("开始训练 Transformer 模型...")

        self.training_thread = TransformerTrainingThread(data_dir, model_save_path)
        self.training_thread.log_signal.connect(self.log)
        self.training_thread.start()

    def log(self, message):
        self.log_area.append(message)
        self.log_area.ensureCursorVisible()

    def load_data(self):
        """加载数据：支持图片、视频、点云数据和标签"""
        try:
            # 打开文件对话框，允许多选
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "选择数据文件",
                "",
                "所有支持文件 (*.png *.jpg *.jpeg *.mp4 *.avi *.bin *.txt *.json);;"
                "图片文件 (*.png *.jpg *.jpeg);;"
                "视频文件 (*.mp4 *.avi);;"
                "点云文件 (*.bin);;"
                "标签文件 (*.txt *.json)"
            )

            if not file_paths:
                self.log("未选择任何文件！")
                return

            for file_path in file_paths:
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension in [".png", ".jpg", ".jpeg"]:
                    self.log(f"加载图片文件: {file_path}")
                    self.image_path = file_path
                elif file_extension in [".mp4", ".avi"]:
                    self.log(f"加载视频文件: {file_path}")
                    self.video_path = file_path
                elif file_extension in [".bin"]:
                    self.log(f"加载点云文件: {file_path}")
                    self.point_cloud_data = self.load_point_cloud(file_path)  # 加载点云数据
                elif file_extension in [".txt", ".json"]:
                    self.log(f"加载标签文件: {file_path}")
                    self.label_path = file_path
                else:
                    self.log(f"不支持的文件类型: {file_path}")

            self.log("数据加载完成！")

        except Exception as e:
            self.log(f"加载数据失败: {str(e)}")

    def load_point_cloud(self, file_path):
        """
        加载点云数据文件（.bin 格式）。
        :param file_path: 点云文件路径
        :return: 点云数据（NumPy 数组）
        """
        try:
            point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            self.log(f"点云数据加载成功，文件: {file_path}")
            return point_cloud
        except Exception as e:
            self.log(f"点云数据加载失败: {str(e)}")
            return None

    def run_yolo(self):
        if self.dataset_loader is None:
            self.log("请先加载数据！")
            return

        if self.yolo_model is None:
            self.log("初始化 YOLOv8 模型...")
            self.yolo_model = YOLOv8Model(model_path="yolov8s.pt", num_classes=2)

        image_path = os.path.join(self.dataset_loader.image_path, "000000.png")
        if not os.path.exists(image_path):
            self.log(f"检测失败: 图像文件不存在: {image_path}")
            return

        if self.yolo_thread is None or not self.yolo_thread.isRunning():
            self.yolo_thread = YoloDetectionThread(self.yolo_model, image_path)
            self.yolo_thread.result_signal.connect(self.log)
            self.yolo_thread.start()
        else:
            self.log("YOLOv8 检测线程正在运行，请稍后重试。")

    def run_transformer(self):
        if not hasattr(self, 'point_cloud_data') or self.point_cloud_data is None:
            self.log("请先加载点云数据！")
            return

        try:
            if self.transformer_model is None:
                model_path = "transformer_model.pth"
                if not os.path.exists(model_path):
                    self.log("Transformer 模型权重文件未找到，请先训练模型！")
                    return

                # 初始化 Transformer 模型
                self.transformer_model = TransformerTrajectoryModel(
                    input_dim=4,
                    embed_dim=256,
                    num_heads=8,
                    num_layers=6,
                    output_dim=3,
                    sequence_length=10,
                    future_steps=5
                )
                self.transformer_model.load_state_dict(
                    torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                )
                self.transformer_model.eval()

            # 使用加载的点云数据作为输入
            input_features = torch.tensor(self.point_cloud_data[:10], dtype=torch.float32).unsqueeze(0)
            self.log(f"输入特征形状: {input_features.shape}")

            # 获取预测结果
            predicted_positions, predicted_velocities, predicted_accelerations = self.transformer_model(
                input_features, return_dynamics=True
            )

            # 将预测结果存储为类属性
            self.predicted_positions = predicted_positions.squeeze(0).detach().numpy()
            self.predicted_velocities = predicted_velocities.squeeze(0).detach().numpy()
            self.predicted_accelerations = predicted_accelerations.squeeze(0).detach().numpy()

            # 在控制台输出轨迹点、速度和加速度的统计信息
            self.log("轨迹预测完成！")
            self.log(f"轨迹点数: {len(self.predicted_positions)}")
            self.log(f"速度范围: {self.predicted_velocities.min()} ~ {self.predicted_velocities.max()}")
            self.log(f"加速度范围: {self.predicted_accelerations.min()} ~ {self.predicted_accelerations.max()}")
            # 在 QT 控制台中逐一输出每个轨迹点的速度、加速度和未来位置
            for i, position in enumerate(self.predicted_positions):
                velocity = self.predicted_velocities[i] if i < len(self.predicted_velocities) else [0, 0, 0]
                acceleration = self.predicted_accelerations[i] if i < len(self.predicted_accelerations) else [0, 0, 0]

                self.log(f"轨迹点 {i + 1}:")
                self.log(f"  未来位置: {position}")
                self.log(f"  速度: {velocity}")
                self.log(f"  加速度: {acceleration}")
            # 输出颜色说明
            self.log("可视化颜色说明：")
            self.log("  - 黑色：XYZ坐标轴，用于标识空间方向")
            self.log("  - 红色：点云数据，表示激光雷达采样的环境或物体点")
            self.log("  - 绿色：预测轨迹线，表示 Transformer 预测的未来位置")
            self.log("  - 蓝色：速度向量，表示目标运动的速度大小和方向")
            self.log("  - 橙色：加速度向量，表示目标运动加速度的大小和方向")

        except Exception as e:
            self.predicted_positions = None
            self.predicted_velocities = None
            self.predicted_accelerations = None
            self.log(f"Transformer 预测失败: {str(e)}")

    def run_visualization(self):
        if not hasattr(self, 'point_cloud_data') or self.point_cloud_data is None:
            self.log("请先加载点云数据！")
            return

        if not hasattr(self, 'predicted_positions') or self.predicted_positions is None:
            self.log("请先运行 Transformer 轨迹预测！")
            return

        try:
            self.log("启动可视化...")

            print(f"轨迹形状: {self.predicted_positions.shape}")
            print(f"速度形状: {self.predicted_velocities.shape}")
            print(f"加速度形状: {self.predicted_accelerations.shape}")

            point_cloud = self.point_cloud_data

            self.visualizer.render_point_cloud(
                point_cloud=point_cloud[:, :3],
                trajectories=self.predicted_positions,
                velocities=self.predicted_velocities,
                accelerations=self.predicted_accelerations
            )
            self.visualizer.run()

        except Exception as e:
            self.log(f"可视化失败: {str(e)}")









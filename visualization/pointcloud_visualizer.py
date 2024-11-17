import open3d as o3d
import numpy as np


class PointCloudVisualizer:
    def __init__(self):
        """
        初始化可视化工具，创建窗口并初始化点云对象。
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud Visualization")
        self.pcd = o3d.geometry.PointCloud()  # 初始化点云对象


    def render_point_cloud(self, point_cloud, trajectories=None, velocities=None, accelerations=None):
        self.vis.clear_geometries()

        # 添加点云
        self.pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        self.vis.add_geometry(self.pcd)

        # 添加黑色坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50.0,  # 坐标系大小
            origin=[0, 0, 0]  # 坐标原点
        )
        # 将坐标轴设为黑色
        coordinate_frame.paint_uniform_color([0, 0, 0])
        self.vis.add_geometry(coordinate_frame)

        # 可视化轨迹
        if trajectories is not None:
            self._draw_trajectories(trajectories)

        # 可视化速度向量
        if velocities is not None and trajectories is not None:
            self._draw_vectors(trajectories[:-1], velocities, color=[0, 0, 1])  # 蓝色

        # 可视化加速度向量
        if accelerations is not None and trajectories is not None:
            self._draw_vectors(trajectories[:-2], accelerations, color=[1, 0.5, 0])  # 橙色

    def _draw_detection(self, detection, label=None):
        """
        绘制单个检测框。
        :param detection: [x1, y1, x2, y2, confidence, class_id]
        :param label: 检测框的类别或标签 (可选)
        """
        x1, y1, x2, y2, confidence, class_id = detection

        # 将 2D 检测框扩展为 3D 的边框（假设 z=0）
        lines = [
            [x1, y1, 0], [x2, y1, 0],  # 上边
            [x2, y1, 0], [x2, y2, 0],  # 右边
            [x2, y2, 0], [x1, y2, 0],  # 下边
            [x1, y2, 0], [x1, y1, 0],  # 左边
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(lines)
        line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(0, len(lines) - 1, 2)])
        line_set.paint_uniform_color([1, 0, 0])  # 红色
        self.vis.add_geometry(line_set)

        if label:
            text_3d = o3d.geometry.Text3D(str(label), (x1, y1, 0), font_size=10)
            self.vis.add_geometry(text_3d)

    def run(self):
        """
        启动可视化。
        """
        self.vis.run()
        self.vis.destroy_window()

    def _draw_vectors(self, origins, vectors, color):
        for origin, vector in zip(origins, vectors):
            if origin.shape != (3,) or vector.shape != (3,):
                print(f"向量形状不匹配，跳过: origin={origin.shape}, vector={vector.shape}")
                continue
            line_set = o3d.geometry.LineSet()
            points = [origin, origin + vector]
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color(color)
            self.vis.add_geometry(line_set)

    def _draw_trajectories(self, trajectories):
        if len(trajectories) < 2:
            print("轨迹数据不足，无法绘制！")
            return

        line_set = o3d.geometry.LineSet()
        points = [tuple(point) for point in trajectories]
        lines = [[i, i + 1] for i in range(len(points) - 1)]
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0, 1, 0])  # 绿色
        self.vis.add_geometry(line_set)


import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在 pyplot 之前！
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
import torch
from collections import deque
import time

# ======================
# 初始化可视化组件
# ======================
class EntropyVisualizer:
    def __init__(self, video_path="entropy_visualization.mp4", fps=20, 
                 history_len=200, wrist_size=(256, 256), global_size=(256, 256)):
        """
        初始化熵可视化器
        :param video_path: 输出视频路径
        :param fps: 视频帧率
        :param history_len: 熵历史记录长度（用于曲线平滑）
        :param wrist_size: 腕部视角显示尺寸 (width, height)
        :param global_size: 全局视角显示尺寸 (width, height)
        """
        self.video_path = video_path
        self.fps = fps
        self.history_len = history_len
        self.wrist_size = wrist_size
        self.global_size = global_size
        
        # 视频写入器初始化
        self.frame_width = wrist_size[0] + global_size[0] + 300  # 300为曲线图宽度
        self.frame_height = max(wrist_size[1], global_size[1], 256)
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (self.frame_width, self.frame_height)
        )
        
        # 熵历史记录 (双端队列自动维护固定长度)
        self.entropy_history = deque(maxlen=history_len)
        self.time_steps = deque(maxlen=history_len)
        self.current_step = 0
        
        # 性能优化：预生成matplotlib画布
        self.fig, self.ax = plt.subplots(figsize=(3.5, 2.5), dpi=80)
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title('Policy Entropy', fontsize=10)
        self.ax.set_xlabel('Time Step', fontsize=8)
        self.ax.set_ylabel('Entropy (bits)', fontsize=8)
        self.ax.tick_params(axis='both', which='major', labelsize=7)
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5, label='Entropy')
        self.point, = self.ax.plot([], [], 'ro', markersize=6, label='Current')
        self.ax.legend(loc='upper right', fontsize=7)
        
        # 预分配曲线图内存
        self.curve_img = np.ones((256, 300, 3), dtype=np.uint8) * 255
        
        # 性能计时
        self.last_frame_time = time.time()
        self.render_count = 0

    def update(self,obs_rgb, current_entropy):
        """更新可视化并写入视频帧"""
        global_img, wrist_img  = obs_rgb[0,-1,:,:,0:3], obs_rgb[0,-1,:,:,3:6]
        # 1. 更新熵历史
        self.entropy_history.append(current_entropy)
        self.time_steps.append(self.current_step)
        self.current_step += 1
        
        # 2. 生成三部分图像
        wrist_view = self._process_view(wrist_img, self.wrist_size)
        global_view = self._process_view(global_img, self.global_size)
        entropy_plot = self._generate_entropy_plot()
        
        # 3. 拼接三部分
        combined = self._combine_views(wrist_view, global_view, entropy_plot)
        
        # 4. 写入视频
        self.video_writer.write(combined)
        self.render_count += 1
        
        # 5. 性能监控 (每30帧打印FPS)
        if self.render_count % 30 == 0:
            fps = 30 / (time.time() - self.last_frame_time)
            print(f"Rendering FPS: {fps:.1f} | Current Entropy: {current_entropy:.3f}")
            self.last_frame_time = time.time()

    def _process_view(self, img, target_size):
        """处理视角图像：归一化+调整大小"""
        # 1. 转换为 NumPy 数组 (处理 PyTorch Tensor)
        if hasattr(img, 'cpu'):
            img = img.detach().cpu().numpy()  # PyTorch Tensor
        elif not isinstance(img, np.ndarray):
            img = np.array(img) 
        # 转换为uint8 (0-255)
        img = (img * 255).astype(np.uint8)
        
        # 调整通道顺序 (HWC to CHW if needed)
        if img.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            img = np.transpose(img, (1, 2, 0))
        
        # 调整大小
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    def _generate_entropy_plot(self):
        """高效生成熵曲线图 (使用 Agg 后端)"""
        # 清除并重绘
        self.ax.clear()
        self.ax.set_facecolor('white')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title('Policy Entropy', fontsize=10)
        self.ax.set_xlabel('Time Step', fontsize=8)
        self.ax.set_ylabel('Entropy (bits)', fontsize=8)
        self.ax.tick_params(axis='both', which='major', labelsize=7)
        
        # 动态Y轴
        if len(self.entropy_history) > 0:
            min_ent = -100 #max(0, min(self.entropy_history) * 0.9)
            max_ent = 0 #max(1.0, max(self.entropy_history) * 1.1)
            self.ax.set_ylim(min_ent, max_ent)
        
        # 绘图
        if self.time_steps:
            self.ax.plot(self.time_steps, self.entropy_history, 'b-', linewidth=1.5)
            self.ax.plot(
                self.time_steps[-1], 
                self.entropy_history[-1], 
                'ro', 
                markersize=6,
                markeredgecolor='darkred'
            )
        
        # === 关键修复：使用 buffer 提取 RGB ===
        self.fig.canvas.draw()
        # 获取渲染器缓冲区
        buf = self.fig.canvas.buffer_rgba()  # Agg 后端支持
        plot_img = np.asarray(buf)  # 转为 numpy array
        
        # 转换 RGBA → BGR (OpenCV 格式)
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        
        # 调整到目标尺寸
        return cv2.resize(plot_img, (300, 256), interpolation=cv2.INTER_AREA)

    def _combine_views(self, wrist, global_view, entropy_plot):
        """拼接三部分视图"""
        # 创建空白画布
        canvas = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 240
        
        # 放置腕部视角 (左)
        wrist_h, wrist_w = wrist.shape[:2]
        canvas[0:wrist_h, 0:wrist_w] = wrist
        
        # 放置全局视角 (中)
        global_h, global_w = global_view.shape[:2]
        canvas[0:global_h, wrist_w:wrist_w+global_w] = global_view
        
        # 放置熵曲线 (右)
        curve_h, curve_w = entropy_plot.shape[:2]
        start_x = wrist_w + global_w
        canvas[0:curve_h, start_x:start_x+curve_w] = entropy_plot
        
        # 添加分割线
        cv2.line(canvas, (wrist_w, 0), (wrist_w, self.frame_height), (180, 180, 180), 2)
        cv2.line(canvas, (wrist_w+global_w, 0), (wrist_w+global_w, self.frame_height), (180, 180, 180), 2)
        
        # 添加标题
        cv2.putText(canvas, "Wrist View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Global View", (wrist_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Entropy", (wrist_w + global_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return canvas

    def release(self):
        """释放资源"""
        self.video_writer.release()
        plt.close(self.fig)
        print(f"Video saved to: {self.video_path}")
        print(f"Total frames rendered: {self.render_count}")

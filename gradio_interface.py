import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import tempfile
import shutil
import time
from model import MultiTaskModel
from pathlib import Path
import matplotlib

# 设置matplotlib后端为非交互式
plt.switch_backend('Agg')

# 配置matplotlib中文字体支持
def setup_chinese_font():
    """配置matplotlib以支持中文显示"""
    import platform
    import matplotlib.font_manager as fm

    system = platform.system()

    # 尝试常见的中文字体
    chinese_fonts = []

    if system == 'Windows':
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC']

    # 查找可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print(f"已设置中文字体: {font}")
            return True

    # 如果没有找到预设字体，使用系统中任何包含CJK的字体
    for font_name in available_fonts:
        if any(keyword in font_name.lower() for keyword in ['cjk', 'chinese', 'zh', 'simhei', 'simsun']):
            matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"已设置中文字体: {font_name}")
            return True

    print("警告: 未找到合适的中文字体，中文可能无法正常显示")
    # 即使没找到中文字体，也设置unicode_minus避免负号问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    return False

# 初始化中文字体
setup_chinese_font()

class StrokeGradioInterface:
    """
    Gradio界面类，用于中风损伤检测和时间分类
    """
    def __init__(self, model_path='output/model_best.pt'):
        """
        初始化Gradio界面

        Args:
            model_path: 模型权重文件路径
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """
        初始化模型
        """
        try:
            # 加载模型
            self.model = MultiTaskModel(dropout_rate=0.0)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"模型已从 {self.model_path} 加载到 {self.device}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            self.model = None

    def preprocess_image(self, dwi_file, flair_file):
        """
        预处理上传的DWI和FLAIR图像

        Args:
            dwi_file: DWI图像文件
            flair_file: FLAIR图像文件

        Returns:
            torch.Tensor: 预处理后的2通道图像张量
            tuple: 原始图像（用于可视化）
        """
        import pydicom
        from PIL import Image

        try:
            # 读取DICOM文件
            dwi_ds = pydicom.dcmread(dwi_file.name)
            flair_ds = pydicom.dcmread(flair_file.name)

            dwi = dwi_ds.pixel_array.astype(np.float32)
            flair = flair_ds.pixel_array.astype(np.float32)

            # 处理多帧DICOM
            if len(dwi.shape) > 2:
                dwi = dwi[0]
            if len(flair.shape) > 2:
                flair = flair[0]

            # 保存原始图像用于可视化
            dwi_original = dwi.copy()
            flair_original = flair.copy()

            # 归一化到[0, 1]
            dwi = (dwi - np.min(dwi)) / (np.ptp(dwi) + 1e-5)
            flair = (flair - np.min(flair)) / (np.ptp(flair) + 1e-5)

            # 转换为PIL图像并调整大小
            dwi_img = Image.fromarray((dwi * 255).astype(np.uint8)).convert('L')
            flair_img = Image.fromarray((flair * 255).astype(np.uint8)).convert('L')

            dwi_img = dwi_img.resize((224, 224), resample=Image.BILINEAR)
            flair_img = flair_img.resize((224, 224), resample=Image.BILINEAR)

            # 转换为张量
            dwi_tensor = torch.from_numpy(np.array(dwi_img)).float() / 255.0
            flair_tensor = torch.from_numpy(np.array(flair_img)).float() / 255.0

            img_tensor = torch.stack([dwi_tensor, flair_tensor], dim=0).unsqueeze(0)  # (1, 2, 224, 224)

            return img_tensor, (dwi_original, flair_original)
        except Exception as e:
            print(f"预处理图像失败: {str(e)}")
            raise

    def predict(self, dwi_file, flair_file):
        """
        执行推理并生成可视化结果

        Args:
            dwi_file: DWI图像文件
            flair_file: FLAIR图像文件

        Returns:
            matplotlib.figure.Figure: 可视化结果
            dict: 预测结果
            float: 推理时间
        """
        if self.model is None:
            self.initialize_model()
            if self.model is None:
                raise Exception("模型加载失败，请检查模型文件路径")

        # 记录开始时间
        start_time = time.time()

        # 预处理图像
        img_tensor, (dwi_original, flair_original) = self.preprocess_image(dwi_file, flair_file)
        img_tensor = img_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            lesion_out, time_out = self.model(img_tensor)

            # 获取概率
            lesion_probs = torch.softmax(lesion_out, dim=1).cpu().numpy()[0]
            time_probs = torch.softmax(time_out, dim=1).cpu().numpy()[0]

            # 获取预测结果
            lesion_pred = int(torch.argmax(lesion_out, dim=1).cpu().numpy()[0])
            time_pred = int(torch.argmax(time_out, dim=1).cpu().numpy()[0])

        # 计算推理时间
        inference_time = time.time() - start_time

        results = {
            'lesion_pred': lesion_pred,
            'lesion_prob': float(lesion_probs[1]),  # 损伤存在的概率
            'lesion_confidence': float(max(lesion_probs)),
            'time_pred': time_pred,
            'time_prob': float(time_probs[1]),  # 晚期阶段(>=270 min)的概率
            'time_confidence': float(max(time_probs)),
            'originals': (dwi_original, flair_original),
            'inference_time': inference_time
        }

        # 生成可视化结果
        fig = self.visualize_results(dwi_file.name, flair_file.name, results)

        return fig, results, inference_time

    def preview_images(self, dwi_file, flair_file):
        """
        预览上传的DICOM图像（不进行推理）

        Args:
            dwi_file: DWI图像文件
            flair_file: FLAIR图像文件

        Returns:
            matplotlib.figure.Figure: 预览图像
        """
        if dwi_file is None or flair_file is None:
            return None

        try:
            import pydicom

            # 读取DICOM文件
            dwi_ds = pydicom.dcmread(dwi_file.name)
            flair_ds = pydicom.dcmread(flair_file.name)

            dwi = dwi_ds.pixel_array.astype(np.float32)
            flair = flair_ds.pixel_array.astype(np.float32)

            # 处理多帧DICOM
            if len(dwi.shape) > 2:
                dwi = dwi[0]
            if len(flair.shape) > 2:
                flair = flair[0]

            # 归一化用于显示
            dwi_display = (dwi - np.min(dwi)) / (np.ptp(dwi) + 1e-5)
            flair_display = (flair - np.min(flair)) / (np.ptp(flair) + 1e-5)

            # 创建预览图
            fig = plt.figure(figsize=(10, 4.5), facecolor='white')

            # DWI图像
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(dwi_display, cmap='gray')
            ax1.set_title('DWI序列', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
            ax1.axis('off')

            # 添加边框
            from matplotlib.patches import Rectangle
            rect1 = Rectangle((0, 0), dwi.shape[1], dwi.shape[0],
                            linewidth=3, edgecolor='#3498db', facecolor='none')
            ax1.add_patch(rect1)

            # FLAIR图像
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(flair_display, cmap='gray')
            ax2.set_title('FLAIR序列', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
            ax2.axis('off')

            # 添加边框
            rect2 = Rectangle((0, 0), flair.shape[1], flair.shape[0],
                            linewidth=3, edgecolor='#e67e22', facecolor='none')
            ax2.add_patch(rect2)

            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"预览图像失败: {str(e)}")
            return None

    def visualize_results(self, dwi_path, flair_path, results):
        """
        可视化推理结果（华丽版本）

        Args:
            dwi_path: DWI文件路径
            flair_path: FLAIR文件路径
            results: 预测结果

        Returns:
            matplotlib.figure.Figure: 可视化结果
        """
        dwi_original, flair_original = results['originals']

        # 归一化用于显示
        dwi_display = (dwi_original - np.min(dwi_original)) / (np.ptp(dwi_original) + 1e-5)
        flair_display = (flair_original - np.min(flair_original)) / (np.ptp(flair_original) + 1e-5)

        # 创建华丽的布局
        fig = plt.figure(figsize=(18, 9), facecolor='white')

        # 添加主标题和推理时间
        inference_time = results.get('inference_time', 0)
        fig.text(0.5, 0.99, '中风损伤智能检测与时间分类系统',
                ha='center', va='top', fontsize=25, fontweight='bold',
                color='#2c3e50')
        fig.text(0.5, 0.935, f'推理时间: {inference_time:.3f} 秒',
                ha='center', va='top', fontsize=20, color="#495353")

        # 创建子图
        gs = plt.GridSpec(2, 4, figure=fig, height_ratios=[2.2, 1],
                         hspace=0.35, wspace=0.30, top=0.91, bottom=0.06,
                         left=0.04, right=0.98)

        # 损伤检测结果
        lesion_label = "检测到损伤" if results['lesion_pred'] == 1 else "未检测到损伤"
        lesion_color = '#e74c3c' if results['lesion_pred'] == 1 else '#27ae60'
        lesion_bg = '#fde8e8' if results['lesion_pred'] == 1 else '#e8f8f5'

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.patches import FancyBboxPatch, Rectangle

        # DWI图像 - 使用炫彩colormap
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(dwi_display, cmap='viridis', interpolation='bilinear')
        ax1.set_title('DWI序列', fontsize=14, fontweight='bold',
                     color='#2c3e50', pad=12,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.9))
        ax1.axis('off')

        # 添加colorbar
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="4%", pad=0.1)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.ax.tick_params(labelsize=8)

        # 添加装饰性边框
        rect1 = FancyBboxPatch((0, 0), 1, 1, transform=ax1.transAxes,
                              boxstyle="round,pad=0.01", linewidth=3.5,
                              edgecolor=lesion_color, facecolor='none', alpha=0.95)
        ax1.add_patch(rect1)

        # FLAIR图像 - 使用plasma colormap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(flair_display, cmap='plasma', interpolation='bilinear')
        ax2.set_title('FLAIR序列', fontsize=14, fontweight='bold',
                     color='#2c3e50', pad=12,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.9))
        ax2.axis('off')

        # 添加colorbar
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="4%", pad=0.1)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.ax.tick_params(labelsize=8)

        # 添加装饰性边框
        rect2 = FancyBboxPatch((0, 0), 1, 1, transform=ax2.transAxes,
                              boxstyle="round,pad=0.01", linewidth=3.5,
                              edgecolor=lesion_color, facecolor='none', alpha=0.95)
        ax2.add_patch(rect2)

        # 损伤检测结果卡片
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        # 背景
        bg_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, transform=ax3.transAxes,
                               facecolor='#f8f9fa', alpha=1, zorder=0,
                               edgecolor='#dee2e6', linewidth=2.5)
        ax3.add_patch(bg_rect)

        y_pos = 0.88
        ax3.text(0.5, y_pos, '损伤检测', ha='center', va='top',
                fontsize=14, fontweight='bold', color='#2c3e50',
                transform=ax3.transAxes)

        y_pos -= 0.18
        result_box = plt.Rectangle((0.08, y_pos - 0.14), 0.84, 0.20,
                                   transform=ax3.transAxes,
                                   facecolor=lesion_bg, alpha=0.95, zorder=1,
                                   edgecolor=lesion_color, linewidth=2.5)
        ax3.add_patch(result_box)

        ax3.text(0.5, y_pos - 0.04, lesion_label, ha='center', va='center',
                fontsize=13, fontweight='bold', color=lesion_color,
                transform=ax3.transAxes)

        y_pos -= 0.14
        ax3.text(0.5, y_pos - 0.04, f'置信度: {results["lesion_confidence"]:.1%}',
                ha='center', va='center',
                fontsize=11, color='#34495e', fontweight='bold',
                transform=ax3.transAxes)

        # 时间分类结果卡片
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        if results['lesion_pred'] == 1:
            bg_rect2 = plt.Rectangle((0.05, 0.05), 0.9, 0.9, transform=ax4.transAxes,
                                   facecolor='#f8f9fa', alpha=1, zorder=0,
                                   edgecolor='#dee2e6', linewidth=2.5)
            ax4.add_patch(bg_rect2)

            time_stage = "晚期阶段\n(≥270 min)" if results['time_pred'] == 1 else "早期阶段\n(<270 min)"
            time_color = '#e67e22' if results['time_pred'] == 1 else '#3498db'
            time_bg = '#fef5e7' if results['time_pred'] == 1 else '#ebf5fb'

            y_pos = 0.88
            ax4.text(0.5, y_pos, '时间分类', ha='center', va='top',
                    fontsize=14, fontweight='bold', color='#2c3e50',
                    transform=ax4.transAxes)

            y_pos -= 0.18
            time_box = plt.Rectangle((0.08, y_pos - 0.14), 0.84, 0.20,
                                    transform=ax4.transAxes,
                                    facecolor=time_bg, alpha=0.95, zorder=1,
                                    edgecolor=time_color, linewidth=2.5)
            ax4.add_patch(time_box)

            ax4.text(0.5, y_pos - 0.04, time_stage, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=time_color,
                    transform=ax4.transAxes)

            y_pos -= 0.14
            ax4.text(0.5, y_pos - 0.04, f'置信度: {results["time_confidence"]:.1%}',
                    ha='center', va='center',
                    fontsize=11, color='#34495e', fontweight='bold',
                    transform=ax4.transAxes)

        # 底部概率柱状图
        labels = ['无损伤', '有损伤']
        values = [1 - results['lesion_prob'], results['lesion_prob']]
        colors = ['#27ae60', '#e74c3c']

        ax5 = fig.add_subplot(gs[1, :2])
        ax5.set_title('损伤检测概率分布', fontsize=13, fontweight='bold',
                     color='#2c3e50', pad=10)

        x_pos = np.arange(len(labels))
        bars = ax5.bar(x_pos, values, color=colors, alpha=0.88,
                      edgecolor='white', linewidth=2.5, width=0.55)

        # 添加渐变和阴影效果
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
            bar.set_edgecolor('#2c3e50')
            bar.set_linewidth(2.5)

            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            shadow = Rectangle((x + 0.02, y - 0.02), w, h,
                             facecolor='gray', alpha=0.25, zorder=0)
            ax5.add_patch(shadow)

        ax5.set_ylim([0, 1.2])
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax5.set_ylabel('概率', fontsize=11, fontweight='bold', color='#2c3e50')
        ax5.grid(axis='y', linestyle='--', alpha=0.35, linewidth=1.2)
        ax5.set_axisbelow(True)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['left'].set_linewidth(1.5)
        ax5.spines['bottom'].set_linewidth(1.5)

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.04,
                    f'{val:.1%}', ha='center', va='bottom',
                    fontsize=13, fontweight='bold', color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor=colors[i], linewidth=2, alpha=0.95))

        # 时间阶段概率柱状图
        if results['lesion_pred'] == 1:
            ax6 = fig.add_subplot(gs[1, 2:])
            ax6.set_title('时间阶段概率分布', fontsize=13, fontweight='bold',
                         color='#2c3e50', pad=10)

            labels_time = ['早期阶段', '晚期阶段']
            values_time = [1 - results['time_prob'], results['time_prob']]
            colors_time = ['#3498db', '#e67e22']

            x_pos_time = np.arange(len(labels_time))
            bars_time = ax6.bar(x_pos_time, values_time, color=colors_time, alpha=0.88,
                               edgecolor='white', linewidth=2.5, width=0.55)

            # 添加渐变和阴影效果
            for bar, color in zip(bars_time, colors_time):
                bar.set_facecolor(color)
                bar.set_edgecolor('#2c3e50')
                bar.set_linewidth(2.5)

                x, y = bar.get_xy()
                w, h = bar.get_width(), bar.get_height()
                shadow = Rectangle((x + 0.02, y - 0.02), w, h,
                                 facecolor='gray', alpha=0.25, zorder=0)
                ax6.add_patch(shadow)

            ax6.set_ylim([0, 1.2])
            ax6.set_xticks(x_pos_time)
            ax6.set_xticklabels(labels_time, fontsize=12, fontweight='bold')
            ax6.set_ylabel('概率', fontsize=11, fontweight='bold', color='#2c3e50')
            ax6.grid(axis='y', linestyle='--', alpha=0.35, linewidth=1.2)
            ax6.set_axisbelow(True)
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            ax6.spines['left'].set_linewidth(1.5)
            ax6.spines['bottom'].set_linewidth(1.5)

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars_time, values_time)):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.04,
                        f'{val:.1%}', ha='center', va='bottom',
                        fontsize=13, fontweight='bold', color=colors_time[i],
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                 edgecolor=colors_time[i], linewidth=2, alpha=0.95))

        return fig

    def create_interface(self):
        """
        创建Gradio界面（优化版本）
        """
        def process_files(dwi_file, flair_file):
            """处理上传的文件并返回结果"""
            try:
                fig, _, _ = self.predict(dwi_file, flair_file)
                return fig
            except Exception as e:
                import traceback
                print(f"处理文件时出错: {str(e)}")
                traceback.print_exc()
                return None

        def update_preview(dwi_file, flair_file):
            """更新预览图像"""
            if dwi_file is not None and flair_file is not None:
                preview_fig = self.preview_images(dwi_file, flair_file)
                return preview_fig
            return None

        # 自定义主题
        theme = gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=["Inter", "system-ui", "sans-serif"],
        ).set(
            button_primary_background_fill='linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            button_primary_background_fill_hover='linear-gradient(135deg, #764ba2 0%, #667eea 100%)',
        )

        # 创建界面
        with gr.Blocks(title="🧠 中风MRI智能分析系统", theme=theme, css="""
            .gradio-container {
                max-width: 100% !important;
            }
            .left-col {
                min-width: 450px !important;
            }
            .right-col {
                min-width: 900px !important;
            }
            .preview-plot {
                height: 400px !important;
            }
            .result-plot {
                height: 750px !important;
            }
            footer {
                display: none !important;
            }
        """) as interface:

            # 标题
            gr.HTML("""
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 12px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);'>
                <h1 style='color: white; margin: 0; font-size: 2.2em; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
                    🧠 多模态MRI卒中发病时间智能评估系统
                </h1>
                <p style='color: rgba(255,255,255,0.95); margin-top: 8px; font-size: 1.05em;'>
                    🏥 基于深度学习和类脑计算的医学影像智能诊断平台 | 损伤检测 · 时间分类 · 快速响应 · 精准分析
                </p>
            </div>
            """)

            # 主内容区域
            with gr.Row():
                # 左侧：输入和预览
                with gr.Column(scale=2, elem_classes=["left-col"]):
                    gr.Markdown("### 📁 上传 DICOM 文件")
                    dwi_input = gr.File(
                        label="🔵 DWI 序列",
                        type="filepath",
                        file_types=[".dcm"]
                    )
                    flair_input = gr.File(
                        label="🟠 FLAIR 序列",
                        type="filepath",
                        file_types=[".dcm"]
                    )

                    process_btn = gr.Button(
                        "🚀 开始智能分析",
                        variant="primary",
                        size="lg"
                    )

                    gr.Markdown("### 👁️ 图像预览")
                    preview_plot = gr.Plot(
                        label="",
                        elem_classes=["preview-plot"]
                    )

                # 右侧：分析结果
                with gr.Column(scale=3, elem_classes=["right-col"]):
                    gr.Markdown("### 📊 智能分析结果")
                    result_plot = gr.Plot(
                        label="",
                        elem_classes=["result-plot"]
                    )

            # 设置事件处理 - 自动预览
            dwi_input.change(
                fn=update_preview,
                inputs=[dwi_input, flair_input],
                outputs=[preview_plot]
            )
            flair_input.change(
                fn=update_preview,
                inputs=[dwi_input, flair_input],
                outputs=[preview_plot]
            )

            # 点击按钮分析
            process_btn.click(
                fn=process_files,
                inputs=[dwi_input, flair_input],
                outputs=[result_plot]
            )

            # 页脚
            gr.HTML("""
            <div style='margin-top: 20px; padding: 15px; text-align: center;
                        background-color: rgba(255,255,255,0.8); border-radius: 10px;'>
                <p style='color: #7f8c8d; margin: 3px 0; font-size: 0.95em;'>
                    <strong>⚕️ 医学影像AI分析平台</strong> |
                    采用先进的深度学习技术 | 准确、快速、可靠
                </p>
                <p style='color: #95a5a6; font-size: 0.85em; margin: 3px 0;'>
                    © 2025 中风MRI智能分析系统 | 仅供研究使用
                </p>
            </div>
            """)

        return interface

if __name__ == "__main__":
    # 创建并启动Gradio界面
    stroke_interface = StrokeGradioInterface()
    interface = stroke_interface.create_interface()

    # 启用队列功能
    interface.queue()

    # 启动界面（本地访问，如需公开链接可设置share=True）
    interface.launch(share=False, debug=False, server_name="127.0.0.1", server_port=7860)

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
        """
        if self.model is None:
            self.initialize_model()
            if self.model is None:
                raise Exception("模型加载失败，请检查模型文件路径")
        
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
        
        results = {
            'lesion_pred': lesion_pred,
            'lesion_prob': float(lesion_probs[1]),  # 损伤存在的概率
            'lesion_confidence': float(max(lesion_probs)),
            'time_pred': time_pred,
            'time_prob': float(time_probs[1]),  # 晚期阶段(>=270 min)的概率
            'time_confidence': float(max(time_probs)),
            'originals': (dwi_original, flair_original)
        }
        
        # 生成可视化结果
        fig = self.visualize_results(dwi_file.name, flair_file.name, results)
        
        return fig, results
    
    def visualize_results(self, dwi_path, flair_path, results):
        """
        可视化推理结果
        
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
        
        # 创建一个更美观的布局
        fig = plt.figure(figsize=(14, 8), facecolor='#f0f0f0')
        
        # 添加标题区域
        plt.suptitle('中风损伤检测与时间分类', fontsize=20, fontweight='bold', y=0.98)
        
        # 创建子图
        gs = plt.GridSpec(2, 3, figure=fig, height_ratios=[3, 1.5], hspace=0.3, wspace=0.2)
        
        # DWI图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(dwi_display, cmap='gray')
        ax1.set_title('DWI序列', fontsize=14, fontweight='bold', color='#333333')
        ax1.axis('off')
        
        # FLAIR图像
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(flair_display, cmap='gray')
        ax2.set_title('FLAIR序列', fontsize=14, fontweight='bold', color='#333333')
        ax2.axis('off')
        
        # 结果概览
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        # 损伤检测结果
        lesion_label = "检测到损伤" if results['lesion_pred'] == 1 else "未检测到损伤"
        lesion_color = '#d62728' if results['lesion_pred'] == 1 else '#2ca02c'
        
        # 添加基于损伤检测的彩色边框
        for ax in [ax1, ax2]:
            rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                linewidth=4, edgecolor=lesion_color, facecolor='none')
            ax.add_patch(rect)
        
        # 在右侧面板显示结果
        result_text = f"损伤检测结果:\n"
        result_text += f"  {lesion_label}\n"
        result_text += f"  置信度: {results['lesion_confidence']:.2%}\n\n"
        
        if results['lesion_pred'] == 1:
            time_stage = "晚期阶段 (≥270 分钟)" if results['time_pred'] == 1 else "早期阶段 (<270 分钟)"
            time_color = '#ff7f0e' if results['time_pred'] == 1 else '#1f77b4'
            result_text += f"时间分类结果:\n"
            result_text += f"  {time_stage}\n"
            result_text += f"  置信度: {results['time_confidence']:.2%}\n"
        
        # 创建文本框
        props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#dddddd', linewidth=1)
        ax3.text(0.5, 0.5, result_text, ha='center', va='center', fontsize=12, bbox=props, transform=ax3.transAxes)
        
        # 添加底部概率条
        ax4 = fig.add_subplot(gs[1, :])
        ax4.set_title('预测概率分布', fontsize=14, fontweight='bold', color='#333333')
        
        # 创建概率条数据
        labels = ['无损伤概率', '有损伤概率']
        values = [1 - results['lesion_prob'], results['lesion_prob']]
        colors = ['#2ca02c', '#d62728']
        
        # 绘制概率条
        bars = ax4.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('概率', fontsize=12)
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 如果检测到损伤，添加时间阶段概率
        if results['lesion_pred'] == 1:
            # 在下方添加时间阶段概率
            ax5 = ax4.twinx()
            labels_time = ['早期阶段', '晚期阶段']
            values_time = [1 - results['time_prob'], results['time_prob']]
            colors_time = ['#1f77b4', '#ff7f0e']
            
            # 调整位置，避免重叠
            x_pos = np.arange(len(labels_time)) + 0.3  # 偏移位置
            bars_time = ax5.bar(x_pos, values_time, width=0.3, color=colors_time, alpha=0.8, edgecolor='black', linewidth=1)
            ax5.set_ylim([0, 1])
            
            # 添加数值标签
            for bar in bars_time:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 添加图例
            ax4.legend(bars, labels, loc='upper left')
            ax5.legend(bars_time, labels_time, loc='upper right')
        else:
            ax4.legend(bars, labels, loc='upper center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig
    
    def create_interface(self):
        """
        创建Gradio界面
        """
        def process_files(dwi_file, flair_file):
            """处理上传的文件并返回结果"""
            try:
                start_time = time.time()
                fig, results = self.predict(dwi_file, flair_file)
                inference_time = time.time() - start_time
                
                # 准备文本结果
                result_text = f"## 推理结果\n\n"
                result_text += f"**损伤检测**: {'检测到损伤' if results['lesion_pred'] == 1 else '未检测到损伤'}\n"
                result_text += f"  - 置信度: {results['lesion_confidence']:.2%}\n\n"
                
                if results['lesion_pred'] == 1:
                    result_text += f"**时间分类**: {'晚期阶段 (≥270 分钟)' if results['time_pred'] == 1 else '早期阶段 (<270 分钟)'}\n"
                    result_text += f"  - 置信度: {results['time_confidence']:.2%}\n\n"
                
                result_text += f"**推理时间**: {inference_time:.2f} 秒\n"
                
                return fig, result_text
            except Exception as e:
                import traceback
                print(f"处理文件时出错: {str(e)}")
                traceback.print_exc()
                return None, f"处理失败: {str(e)}"
        
        # 自定义主题
        theme = gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.gray,
            font=["Inter", "sans-serif"],
            font_mono=["JetBrains Mono", "monospace"],
        )
        
        # 创建界面
        with gr.Blocks(title="中风MRI图像分析系统", theme=theme) as interface:
            gr.Markdown("""
            # 中风MRI图像分析系统
            上传DWI和FLAIR两种模态的MRI图像，系统将自动进行损伤检测和时间分类。
            
            **注意:** 请确保上传的是DICOM格式的图像文件。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    dwi_input = gr.File(label="DWI", type="filepath")
                    flair_input = gr.File(label="FLAIR", type="filepath")
                    process_btn = gr.Button("开始分析", variant="primary", size="large")
                
                with gr.Column(scale=2):
                    result_plot = gr.Plot(label="分析结果可视化")
                    result_text = gr.Markdown()
            
            # 设置按钮点击事件
            process_btn.click(
                fn=process_files,
                inputs=[dwi_input, flair_input],
                outputs=[result_plot, result_text]
            )
            
            # 添加页脚信息
            gr.Markdown("""
            --- 
            <div style='text-align: center; color: #666; font-size: 0.9em;'>
                <p>© 2025 中风MRI图像分析系统 | 使用先进的深度学习技术进行医学影像分析</p>
            </div>
            """)
        
        return interface

if __name__ == "__main__":
    # 创建并启动Gradio界面
    stroke_interface = StrokeGradioInterface()
    interface = stroke_interface.create_interface()

    # 启用队列功能
    interface.queue()

    # 启动界面，设置share=True以创建公开访问链接
    interface.launch(share=True, debug=False)
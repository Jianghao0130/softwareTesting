import tarfile
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2
import urllib.request
import sys
from tqdm import tqdm
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import warnings

# 忽略 PyTorch 的 meshgrid 警告
warnings.filterwarnings('ignore', message='torch.meshgrid: in an upcoming release')

def download_lfw():
    """下载 LFW 数据集"""
    LFW_URL = 'https://vis-www.cs.umass.edu/lfw/lfw.tgz'
    LFW_TGZ = '../lfw.tgz'
    
    if not os.path.exists(LFW_TGZ):
        print(f"正在下载 LFW 数据集到 {LFW_TGZ}...")
        try:
            # 显示下载进度
            def progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r下载进度: {percent}%")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(LFW_URL, LFW_TGZ, progress)
            print("\n数据集下载完成！")
        except Exception as e:
            print(f"\n下载失败: {e}")
            print(f"请手动从 {LFW_URL} 下载数据集并放置到 {LFW_TGZ}")
            sys.exit(1)
    else:
        print(f"发现已存在的数据集文件: {LFW_TGZ}")

# 在开始时检查并下载数据
download_lfw()

# 解压函数
def extract_tgz(file_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dest_path)
    print(f"已解压 {file_path} 到 {dest_path}")

# 解压 LFW 数据集
LFW_TGZ = '../lfw.tgz'
LFW_DEST = '../test_images/'

if not os.path.exists(LFW_DEST):
    extract_tgz(LFW_TGZ, LFW_DEST)

# 添加新的导入
from mim import download

# 下载模型
model_name = 'rtmdet-ins_l_8xb32-300e_coco'
try:
    # 创建 checkpoints 目录
    os.makedirs('../checkpoints', exist_ok=True)

    # 使用 mim 下载模型
    download('mmdet', [model_name], dest_root='../checkpoints')
    print(f'模型已下载到: ../checkpoints/')
except Exception as e:
    print(f'下载模型时出错: {e}')
    print('请手动下载模型文件并放置到 ../checkpoints/ 目录下')


# 配置文件路径为 RTMDet-Ins 模型
CONFIG_FILE = '../configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py'
CHECKPOINT_FILE = '../checkpoints/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'

# 修改输入和输出路径
ORIGINAL_INPUT_DIR = '../test_images/lfw/'
AUGMENTED_INPUT_DIR = '../test_images_augmented/'
OUTPUT_DIR = '../results/'

# 注册所有模块
register_all_modules()

# 创建结果保存目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化模型
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')

# 设置 COCO 类名
model.dataset_meta = {'classes': tuple([
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
])}

# 调整检测参数
CONF_THRESH = 0.3  # 降低置信度阈值以检测更多人脸
MAX_DETECTIONS = 20  # 增加最大检测数量

# 颜色配置
COLORS = {
    'person': (0, 255, 0),  # 绿色表示人
    'face': (255, 0, 0)     # 红色表示脸
}

# 测试模式配置
TEST_MODE = False  # 设置为 True 时只处理少量图片
TEST_SAMPLES_PER_PERSON = 2  # 每个人处理的图片数量
TEST_MAX_PERSONS = 3  # 测试时处理的人数

def process_image(img_path, output_subdir):
    """处理单张图片的函数"""
    # 推理
    result = inference_detector(model, img_path)
    
    # 读取图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 获取检测结果
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    
    # 过滤低置信度结果
    high_conf_indices = scores >= CONF_THRESH
    bboxes = bboxes[high_conf_indices]
    scores = scores[high_conf_indices]
    labels = labels[high_conf_indices]
    
    # 按置信度排序并限制检测数量
    sort_indices = scores.argsort()[::-1][:MAX_DETECTIONS]
    bboxes = bboxes[sort_indices]
    scores = scores[sort_indices]
    labels = labels[sort_indices]
    
    # 增强的可视化效果
    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = bbox.astype(int)
        class_name = model.dataset_meta['classes'][label]
        color = COLORS.get(class_name.lower(), (255, 0, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{class_name}: {score:.2f}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_width, text_height = text_size
        
        cv2.rectangle(img, (x1, y1 - text_height - 4), 
                    (x1 + text_width + 2, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 1, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存结果
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, os.path.basename(img_path))
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # 返回检测结果
    return result

class TestResults:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_images': {},
            'augmented_images': {},
            'metrics': {
                'per_person': {},
                'per_augmentation': {},
                'overall': {}
            }
        }
    
    def add_result(self, image_path, detections, is_augmented=False):
        """添加单张图片的检测结果"""
        person_name = os.path.basename(os.path.dirname(image_path))
        img_name = os.path.basename(image_path)
        
        result_dict = self.results['augmented_images' if is_augmented else 'original_images']
        
        if person_name not in result_dict:
            result_dict[person_name] = {}
        
        result_dict[person_name][img_name] = {
            'num_detections': len(detections.pred_instances),
            'confidence_scores': detections.pred_instances.scores.cpu().numpy().tolist(),
            'labels': detections.pred_instances.labels.cpu().numpy().tolist()
        }
    
    def calculate_metrics(self):
        """计算各种评估指标"""
        # 添加更多评估指标
        def calculate_confidence_stats(scores):
            if not scores:
                return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
            return {
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
        
        # 计算每个人的指标
        for person in set(list(self.results['original_images'].keys()) + 
                         list(self.results['augmented_images'].keys())):
            orig_detections = [
                result['num_detections']
                for result in self.results['original_images'].get(person, {}).values()
            ]
            aug_detections = [
                result['num_detections']
                for result in self.results['augmented_images'].get(person, {}).values()
            ]
            
            self.results['metrics']['per_person'][person] = {
                'avg_original_detections': np.mean(orig_detections) if orig_detections else 0,
                'avg_augmented_detections': np.mean(aug_detections) if aug_detections else 0,
                'detection_stability': (np.mean(aug_detections) / np.mean(orig_detections)
                                     if orig_detections and aug_detections else 0)
            }
        
        # 计算每种增强方法的指标
        aug_types = ['bright', 'dark', 'occluded', 'noise_low', 'noise_high',
                    'blur_low', 'blur_high', 'rotate_left', 'rotate_right']
        
        for aug_type in aug_types:
            aug_results = []
            for person_results in self.results['augmented_images'].values():
                for img_name, result in person_results.items():
                    if aug_type in img_name:
                        aug_results.append(result)
            
            if aug_results:
                self.results['metrics']['per_augmentation'][aug_type] = {
                    'avg_detections': np.mean([r['num_detections'] for r in aug_results]),
                    'avg_confidence': np.mean([np.mean(r['confidence_scores']) for r in aug_results])
                }
        
        # 计算总体指标
        all_orig = []
        all_aug = []
        for person_results in self.results['original_images'].values():
            all_orig.extend(person_results.values())
        for person_results in self.results['augmented_images'].values():
            all_aug.extend(person_results.values())
        
        self.results['metrics']['overall'] = {
            'total_original_images': len(all_orig),
            'total_augmented_images': len(all_aug),
            'avg_original_detections': np.mean([r['num_detections'] for r in all_orig]),
            'avg_augmented_detections': np.mean([r['num_detections'] for r in all_aug]),
            'overall_stability': (np.mean([r['num_detections'] for r in all_aug]) /
                                np.mean([r['num_detections'] for r in all_orig])
                                if all_orig else 0)
        }
        
        # 添加每个类别的检测统计
        class_stats = {}
        for result_dict in [self.results['original_images'], self.results['augmented_images']]:
            for person_results in result_dict.values():
                for result in person_results.values():
                    for label in result['labels']:
                        class_name = model.dataset_meta['classes'][label]
                        if class_name not in class_stats:
                            class_stats[class_name] = {'count': 0, 'confidence_scores': []}
                        class_stats[class_name]['count'] += 1
                        class_stats[class_name]['confidence_scores'].extend(result['confidence_scores'])
        
        self.results['metrics']['per_class'] = {
            class_name: {
                'detection_count': stats['count'],
                'confidence_stats': calculate_confidence_stats(stats['confidence_scores'])
            }
            for class_name, stats in class_stats.items()
        }
    
    def generate_visualizations(self, output_dir):
        """生成可视化图表"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用新版本的 seaborn 样式
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        # 添加类别检测分布图
        plt.figure(figsize=(15, 8))
        class_data = [(class_name, metrics['detection_count'])
                      for class_name, metrics in self.results['metrics']['per_class'].items()]
        class_df = pd.DataFrame(class_data, columns=['Class', 'Count'])
        class_df = class_df.sort_values('Count', ascending=False).head(10)  # 只显示前10个最常见的类别
        # 更新 seaborn 绘图方式
        sns.barplot(
            data=class_df,
            x='Class',
            y='Count',
            hue='Class',  # 使用 Class 作为颜色区分
            palette='husl',
            legend=False  # 不显示图例
        )
        plt.xticks(rotation=45)
        plt.title('Top 10 Most Detected Classes')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_classes.png'))
        plt.close()
        
        # 添加置信度分布图
        plt.figure(figsize=(12, 6))
        all_confidences = []
        for result_dict in [self.results['original_images'], self.results['augmented_images']]:
            for person_results in result_dict.values():
                for result in person_results.values():
                    all_confidences.extend(result['confidence_scores'])
        sns.histplot(all_confidences, bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        plt.close()
    
    def generate_report(self, output_path):
        """生成测试报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# AI 模型测试报告\n\n")
            
            # 基本信息
            f.write("## 测试基本信息\n")
            f.write(f"- 测试时间：{self.results['timestamp']}\n")
            f.write(f"- 原始图片数量：{self.results['metrics']['overall']['total_original_images']}\n")
            f.write(f"- 增强图片数量：{self.results['metrics']['overall']['total_augmented_images']}\n\n")
            
            # 总体性能
            f.write("## 总体性能指标\n")
            overall = self.results['metrics']['overall']
            f.write(f"- 原始图片平均检测数：{overall['avg_original_detections']:.2f}\n")
            f.write(f"- 增强图片平均检测数：{overall['avg_augmented_detections']:.2f}\n")
            f.write(f"- 整体检测稳定性：{overall['overall_stability']:.2f}\n\n")
            
            # 各种增强方法的性能
            f.write("## 不同增强方法的性能\n")
            for aug_type, metrics in self.results['metrics']['per_augmentation'].items():
                f.write(f"### {aug_type}\n")
                f.write(f"- 平均检测数：{metrics['avg_detections']:.2f}\n")
                f.write(f"- 平均置信度：{metrics['avg_confidence']:.2f}\n\n")
            
            # 优化个人检测性能部分
            f.write("## 个人检测性能统计\n")
            # 计算统计数据
            all_stabilities = [m['detection_stability'] for m in self.results['metrics']['per_person'].values()]
            all_orig_detections = [m['avg_original_detections'] for m in self.results['metrics']['per_person'].values()]
            all_aug_detections = [m['avg_augmented_detections'] for m in self.results['metrics']['per_person'].values()]
            
            # 计算总体统计
            f.write("### 总体统计\n")
            f.write(f"- 总样本数：{len(self.results['metrics']['per_person'])} 人\n")
            f.write(f"- 平均检测稳定性：{np.mean(all_stabilities):.2f} ± {np.std(all_stabilities):.2f}\n")
            f.write(f"- 原始图片平均检测数：{np.mean(all_orig_detections):.2f} ± {np.std(all_orig_detections):.2f}\n")
            f.write(f"- 增强图片平均检测数：{np.mean(all_aug_detections):.2f} ± {np.std(all_aug_detections):.2f}\n\n")
            
            # 找出典型案例
            f.write("### 典型案例分析\n")
            
            # 检测稳定性最高的3个人
            top_stable = sorted(
                self.results['metrics']['per_person'].items(),
                key=lambda x: x[1]['detection_stability'],
                reverse=True
            )[:3]
            f.write("#### 检测稳定性最高的案例\n")
            for person, metrics in top_stable:
                f.write(f"- {person}：稳定性 {metrics['detection_stability']:.2f}\n")
            f.write("\n")
            
            # 检测稳定性最低的3个人
            bottom_stable = sorted(
                self.results['metrics']['per_person'].items(),
                key=lambda x: x[1]['detection_stability']
            )[:3]
            f.write("#### 检测稳定性最低的案例\n")
            for person, metrics in bottom_stable:
                f.write(f"- {person}：稳定性 {metrics['detection_stability']:.2f}\n")
            f.write("\n")
            
            # 检测数量最多的3个人
            top_detections = sorted(
                self.results['metrics']['per_person'].items(),
                key=lambda x: x[1]['avg_original_detections'],
                reverse=True
            )[:3]
            f.write("#### 检测数量最多的案例\n")
            for person, metrics in top_detections:
                f.write(f"- {person}：原始 {metrics['avg_original_detections']:.1f}，增强后 {metrics['avg_augmented_detections']:.1f}\n")
            f.write("\n")
            
            # 添加类别分析
            f.write("## 类别检测分析\n")
            top_classes = sorted(
                self.results['metrics']['per_class'].items(),
                key=lambda x: x[1]['detection_count'],
                reverse=True
            )[:10]
            
            f.write("### 前10个最常检测到的类别\n")
            for class_name, metrics in top_classes:
                conf_stats = metrics['confidence_stats']
                f.write(f"#### {class_name}\n")
                f.write(f"- 检测次数：{metrics['detection_count']}\n")
                f.write(f"- 置信度范围：{conf_stats['min']:.2f} - {conf_stats['max']:.2f}\n")
                f.write(f"- 平均置信度：{conf_stats['mean']:.2f} ± {conf_stats['std']:.2f}\n\n")
            
            # 添加测试环境信息
            f.write("## 测试环境信息\n")
            f.write(f"- 模型：RTMDet\n")
            f.write(f"- 设备：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"- 置信度阈值：{CONF_THRESH}\n")
            f.write(f"- 最大检测数量：{MAX_DETECTIONS}\n\n")
    
    def save_results(self, output_path):
        """保存原始结果数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)

def process_directory(input_dir, test_results, is_augmented=False):
    """处理目录中的所有图片"""
    # 收集所有图片路径
    all_images = []
    person_count = 0
    
    for root, _, files in os.walk(input_dir):
        if TEST_MODE and person_count >= TEST_MAX_PERSONS:
            break
            
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            continue
            
        # 在测试模式下限制每个人的图片数量
        if TEST_MODE:
            image_files = image_files[:TEST_SAMPLES_PER_PERSON]
            person_count += 1
            
        for file in image_files:
            all_images.append((root, file))
    
    total_files = len(all_images)
    
    # 获取更友好的目录名显示
    dir_type = "原始图片" if input_dir == ORIGINAL_INPUT_DIR else "增强图片"
    with tqdm(total=total_files, desc=f"处理{dir_type}", ncols=100) as pbar:
        for root, file in all_images:
            img_path = os.path.join(root, file)
            person_name = os.path.basename(os.path.dirname(img_path))
            output_subdir = os.path.join(OUTPUT_DIR, person_name)
            
            # 处理图片并保存结果
            result = process_image(img_path, output_subdir)
            test_results.add_result(img_path, result, is_augmented)
            
            pbar.update(1)

def main():
    if TEST_MODE:
        print(f"\n运行测试模式:")
        print(f"- 处理 {TEST_MAX_PERSONS} 个人的图片")
        print(f"- 每人处理 {TEST_SAMPLES_PER_PERSON} 张图片")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建测试结果对象
    test_results = TestResults()
    
    print("\n开始处理图像...")
    
    # 处理原始图像和增强后的图像
    process_directory(ORIGINAL_INPUT_DIR, test_results, is_augmented=False)
    process_directory(AUGMENTED_INPUT_DIR, test_results, is_augmented=True)
    
    # 计算指标
    test_results.calculate_metrics()
    
    # 生成可视化
    test_results.generate_visualizations(os.path.join(OUTPUT_DIR, 'visualizations'))
    
    # 生成报告
    test_results.generate_report(os.path.join(OUTPUT_DIR, 'test_report.md'))
    
    # 保存原始结果
    test_results.save_results(os.path.join(OUTPUT_DIR, 'test_results.json'))
    
    print(f"\n所有图像处理完成。结果保存在: {OUTPUT_DIR}")
    print("- 可视化图表：visualizations/")
    print("- 测试报告：test_report.md")
    print("- 详细结果：test_results.json")

if __name__ == "__main__":
    main()

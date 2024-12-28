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
        # 计算每个人的指标
        for person in set(list(self.results['original_images'].keys()) + 
                         list(self.results['augmented_images'].keys())):
            orig_detections = [
                result['num_detections']
                for results in self.results['original_images'].get(person, {}).values()
            ]
            aug_detections = [
                result['num_detections']
                for results in self.results['augmented_images'].get(person, {}).values()
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
            aug_results = [
                result
                for person_results in self.results['augmented_images'].values()
                for img_name, result in person_results.items()
                if aug_type in img_name
            ]
            
            if aug_results:
                self.results['metrics']['per_augmentation'][aug_type] = {
                    'avg_detections': np.mean([r['num_detections'] for r in aug_results]),
                    'avg_confidence': np.mean([np.mean(r['confidence_scores']) for r in aug_results]),
                }
        
        # 计算总体指标
        all_orig = [r for p in self.results['original_images'].values() for r in p.values()]
        all_aug = [r for p in self.results['augmented_images'].values() for r in p.values()]
        
        self.results['metrics']['overall'] = {
            'total_original_images': len(all_orig),
            'total_augmented_images': len(all_aug),
            'avg_original_detections': np.mean([r['num_detections'] for r in all_orig]),
            'avg_augmented_detections': np.mean([r['num_detections'] for r in all_aug]),
            'overall_stability': (np.mean([r['num_detections'] for r in all_aug]) /
                                np.mean([r['num_detections'] for r in all_orig])
                                if all_orig else 0)
        }
    
    def generate_visualizations(self, output_dir):
        """生成可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 每个人的检测稳定性柱状图
        plt.figure(figsize=(15, 6))
        stability_data = [(person, metrics['detection_stability'])
                         for person, metrics in self.results['metrics']['per_person'].items()]
        stability_df = pd.DataFrame(stability_data, columns=['Person', 'Stability'])
        sns.barplot(data=stability_df, x='Person', y='Stability')
        plt.xticks(rotation=45)
        plt.title('Detection Stability by Person')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stability_by_person.png'))
        plt.close()
        
        # 2. 不同增强方法的平均置信度对比
        plt.figure(figsize=(12, 6))
        conf_data = [(aug_type, metrics['avg_confidence'])
                    for aug_type, metrics in self.results['metrics']['per_augmentation'].items()]
        conf_df = pd.DataFrame(conf_data, columns=['Augmentation', 'Confidence'])
        sns.barplot(data=conf_df, x='Augmentation', y='Confidence')
        plt.xticks(rotation=45)
        plt.title('Average Confidence by Augmentation Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_by_augmentation.png'))
        plt.close()
    
    def generate_report(self, output_path):
        """生成测试报告"""
        with open(output_path, 'w') as f:
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
            
            # 各个人的性能
            f.write("## 个人检测性能\n")
            for person, metrics in self.results['metrics']['per_person'].items():
                f.write(f"### {person}\n")
                f.write(f"- 原始图片平均检测数：{metrics['avg_original_detections']:.2f}\n")
                f.write(f"- 增强图片平均检测数：{metrics['avg_augmented_detections']:.2f}\n")
                f.write(f"- 检测稳定性：{metrics['detection_stability']:.2f}\n\n")
    
    def save_results(self, output_path):
        """保存原始结果数据"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)

def process_directory(input_dir, test_results, is_augmented=False):
    """处理目录中的所有图片"""
    total_files = sum(1 for root, _, files in os.walk(input_dir) 
                     for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    
    with tqdm(total=total_files, desc=f"处理 {os.path.basename(input_dir)} 中的图片") as pbar:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in {'.jpg', '.png', '.jpeg'}:
                    img_path = os.path.join(root, file)
                    person_name = os.path.basename(os.path.dirname(img_path))
                    output_subdir = os.path.join(OUTPUT_DIR, person_name)
                    
                    # 处理图片并保存结果
                    result = process_image(img_path, output_subdir)
                    test_results.add_result(img_path, result, is_augmented)
                    
                    pbar.update(1)

def main():
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

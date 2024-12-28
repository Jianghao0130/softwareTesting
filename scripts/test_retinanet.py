import tarfile
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2
import urllib.request
import sys
from tqdm import tqdm

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

def process_directory(input_dir):
    """处理目录中的所有图片"""
    # 获取总文件数
    total_files = sum(1 for root, _, files in os.walk(input_dir) 
                     for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    
    # 使用 tqdm 创建进度条
    with tqdm(total=total_files, desc=f"处理 {os.path.basename(input_dir)} 中的图片") as pbar:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in {'.jpg', '.png', '.jpeg'}:
                    # 获取人名
                    person_name = os.path.basename(os.path.dirname(os.path.join(root, file)))
                    output_subdir = os.path.join(OUTPUT_DIR, person_name)
                    
                    img_path = os.path.join(root, file)
                    process_image(img_path, output_subdir)
                    pbar.update(1)

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n开始处理图像...")
    
    # 处理原始图像和增强后的图像
    for input_dir in [ORIGINAL_INPUT_DIR, AUGMENTED_INPUT_DIR]:
        process_directory(input_dir)
    
    print(f"\n所有图像处理完成。结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

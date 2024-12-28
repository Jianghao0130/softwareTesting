import tarfile
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2
from mim import download

# 解压函数
def extract_tgz(file_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dest_path)
    print(f"Extracted {file_path} to {dest_path}")

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


# 更新配置文件路径为 RTMDet-Ins 模型
CONFIG_FILE = '../configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py'
CHECKPOINT_FILE = '../checkpoints/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'

# 输入和输出路径
INPUT_DIR = '../test_images/lfw/'
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

# 遍历所有子文件夹
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if os.path.splitext(file)[1].lower() in {'.jpg', '.png', '.jpeg'}:
            img_path = os.path.join(root, file)
            
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
                
                # 使用不同颜色区分不同类别
                color = COLORS.get(class_name.lower(), (255, 0, 0))
                
                # 绘制检测框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 改进的标签显示
                label_text = f"{class_name}: {score:.2f}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_width, text_height = text_size
                
                # 标签背景
                cv2.rectangle(img, (x1, y1 - text_height - 4), 
                            (x1 + text_width + 2, y1), color, -1)
                # 标签文本
                cv2.putText(img, label_text, (x1 + 1, y1 - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存结果
            output_path = os.path.join(OUTPUT_DIR, file)
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"已检测到 {len(bboxes)} 个目标在 {file} 中")

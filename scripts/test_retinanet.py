import tarfile
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2

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

# 配置文件路径
CONFIG_FILE = '../configs/retinanet/retinanet_r50_fpn_1x_coco.py'
CHECKPOINT_FILE = '../checkpoints/retinanet_resnet50_fpn_coco.pth'

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

            # 绘制检测结果
            bboxes = result.pred_instances.bboxes.cpu().numpy()
            scores = result.pred_instances.scores.cpu().numpy()
            labels = result.pred_instances.labels.cpu().numpy()
            class_names = model.dataset_meta['classes']

            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label_text = f"{class_names[label]}: {score:.2f}"
                cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            output_path = os.path.join(OUTPUT_DIR, file)
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Result saved to {output_path}")

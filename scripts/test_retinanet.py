import tarfile
import os

# 解压函数
def extract_tgz(file_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dest_path)
    print(f"Extracted {file_path} to {dest_path}")

# 解压 LFW 数据集
LFW_TGZ = '../lfw.tgz'  # 压缩包路径
LFW_DEST = '../test_images/'  # 解压目标路径

if not os.path.exists(LFW_DEST):  # 如果未解压，则解压
    extract_tgz(LFW_TGZ, LFW_DEST)


import os
import cv2
from mmdet.apis import init_detector, inference_detector

# 配置文件路径
CONFIG_FILE = '../configs/retinanet/retinanet_r50_fpn_1x_coco.py'
CHECKPOINT_FILE = '../checkpoints/retinanet_r50_fpn_coco.pth'

# 输入和输出路径
INPUT_DIR = '../test_images/lfw'
OUTPUT_DIR = '../results/'

# 创建结果保存目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化模型
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')

# 支持的图片格式
SUPPORTED_FORMATS = {'.jpg', '.png', '.jpeg'}

# 递归遍历所有子文件夹
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if os.path.splitext(file)[1].lower() in SUPPORTED_FORMATS:
            # 图片路径
            img_path = os.path.join(root, file)

            # 推理
            result = inference_detector(model, img_path)

            # 读取图片
            img = cv2.imread(img_path)

            # 在图片上绘制结果
            result_img = model.show_result(img, result, score_thr=0.3, wait_time=0, show=False)

            # 保存结果到 results 文件夹
            relative_path = os.path.relpath(img_path, INPUT_DIR)
            save_path = os.path.join(OUTPUT_DIR, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, result_img)

print("推理完成，结果已保存到 results 文件夹。")

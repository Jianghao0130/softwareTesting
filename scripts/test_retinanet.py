from mmdet.apis import init_detector, inference_detector
import os

# 配置文件和权重路径
CONFIG_FILE = 'configs/retinanet_r50_fpn_1x_coco.py'
CHECKPOINT_FILE = 'checkpoints/retinanet_resnet50_fpn_coco.pth.pth'

# 测试图片路径
TEST_IMAGE_FOLDER = 'test_images/'
RESULT_FOLDER = 'results/'

# 初始化模型
def init_model():
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')  # 使用 GPU
    return model

# 对单张图片进行推理
def test_single_image(model, img_path):
    result = inference_detector(model, img_path)
    output_path = os.path.join(RESULT_FOLDER, os.path.basename(img_path))
    model.show_result(img_path, result, out_file=output_path)
    print(f'Result saved to {output_path}')

# 主函数
if __name__ == "__main__":
    os.makedirs(RESULT_FOLDER, exist_ok=True)  # 确保结果文件夹存在
    model = init_model()

    # 对测试图片文件夹中的所有图片进行推理
    for img_file in os.listdir(TEST_IMAGE_FOLDER):
        img_path = os.path.join(TEST_IMAGE_FOLDER, img_file)
        test_single_image(model, img_path)

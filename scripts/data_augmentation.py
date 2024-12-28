from PIL import Image, ImageEnhance, ImageFilter
import os
import sys
import shutil
import urllib.request
import tarfile
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import multiprocessing
from functools import partial
from itertools import repeat

def download_lfw():
    """下载 LFW 数据集"""
    LFW_URL = 'https://vis-www.cs.umass.edu/lfw/lfw.tgz'
    LFW_TGZ = '../lfw.tgz'

    if not os.path.exists(LFW_TGZ):
        print(f"正在下载 LFW 数据集到 {LFW_TGZ}...")
        try:
            # 使用 tqdm 显示下载进度
            response = urllib.request.urlopen(LFW_URL)
            total_size = int(response.headers['Content-Length'])
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载进度") as pbar:
                urllib.request.urlretrieve(
                    LFW_URL, 
                    LFW_TGZ,
                    lambda count, block_size, total_size: pbar.update(block_size)
                )
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

# 测试图片路径
TEST_IMAGE_FOLDER = '../test_images/lfw'  # 修改为正确的 LFW 子文件夹路径
AUGMENTED_IMAGE_FOLDER = '../test_images_augmented/'

def check_permissions():
    """检查文件夹权限"""
    try:
        # 检查源文件夹是否存在且可读
        if not os.path.exists(TEST_IMAGE_FOLDER):
            print(f"错误: 源文件夹 {TEST_IMAGE_FOLDER} 不存在")
            return False
            
        # 尝试创建目标文件夹
        if os.path.exists(AUGMENTED_IMAGE_FOLDER):
            # 如果目标文件夹存在，先尝试删除
            try:
                shutil.rmtree(AUGMENTED_IMAGE_FOLDER)
                print(f"已清除旧的输出文件夹: {AUGMENTED_IMAGE_FOLDER}")
            except PermissionError:
                print(f"错误: 无法删除已存在的输出文件夹 {AUGMENTED_IMAGE_FOLDER}")
                return False
                
        try:
            os.makedirs(AUGMENTED_IMAGE_FOLDER, exist_ok=True)
            print(f"成功创建输出文件夹: {AUGMENTED_IMAGE_FOLDER}")
        except PermissionError:
            print(f"错误: 无法创建输出文件夹 {AUGMENTED_IMAGE_FOLDER}")
            return False
            
        return True
    except Exception as e:
        print(f"检查权限时出错: {e}")
        return False

# 模拟光线变化
def simulate_lighting(image_path, brightness=1.5):
    try:
        image = Image.open(image_path)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness)
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None

# 模拟遮挡
def simulate_occlusion(image_path):
    try:
        image = Image.open(image_path)
        overlay = Image.new('RGB', image.size, (0, 0, 0))  # 黑色遮挡
        mask = Image.new('L', image.size, 128)  # 半透明遮挡
        image.paste(overlay, (image.size[0] // 4, image.size[1] // 4), mask)
        return image
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None

# 添加高斯噪声
def simulate_noise(image_path, noise_factor=0.1):
    """添加高斯噪声"""
    image = Image.open(image_path)
    img_array = np.array(image)
    noise = np.random.normal(0, noise_factor, img_array.shape)
    noisy_img = img_array + noise
    return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

# 模拟模糊效果
def simulate_blur(image_path, radius=2):
    """模拟模糊效果"""
    image = Image.open(image_path)
    return image.filter(ImageFilter.GaussianBlur(radius))

# 模拟旋转
def simulate_rotation(image_path, angle=15):
    """模拟旋转"""
    image = Image.open(image_path)
    return image.rotate(angle)

def apply_all_augmentations(image_path):
    """对单张图片应用所有增强方法"""
    augmentations = {
        'bright': simulate_lighting(image_path, brightness=1.5),
        'dark': simulate_lighting(image_path, brightness=0.5),
        'occluded': simulate_occlusion(image_path),
        'noise_low': simulate_noise(image_path, noise_factor=0.05),
        'noise_high': simulate_noise(image_path, noise_factor=0.15),
        'blur_low': simulate_blur(image_path, radius=1),
        'blur_high': simulate_blur(image_path, radius=3),
        'rotate_left': simulate_rotation(image_path, angle=-15),
        'rotate_right': simulate_rotation(image_path, angle=15)
    }
    return augmentations

def process_single_image(args):
    """处理单张图片的所有增强"""
    img_path, person_name, output_dir = args
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 应用所有增强方法
        augmented_images = apply_all_augmentations(img_path)
        results = {
            'person_name': person_name,
            'original_count': 1,
            'augmented_count': 0,
            'success': True,
            'error': None
        }
        
        # 保存增强后的图片
        for aug_type, aug_image in augmented_images.items():
            if aug_image:
                output_path = os.path.join(output_dir, f'{aug_type}_{os.path.basename(img_path)}')
                aug_image.save(output_path)
                results['augmented_count'] += 1
                
        return results
        
    except Exception as e:
        return {
            'person_name': person_name,
            'original_count': 0,
            'augmented_count': 0,
            'success': False,
            'error': str(e)
        }

def augment_images():
    """使用多进程增强图片"""
    if not check_permissions():
        print("权限检查失败，程序退出")
        sys.exit(1)

    # 初始化增强信息
    augmentation_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_original_images': 0,
        'total_augmented_images': 0,
        'augmentation_types': [
            'brightness_increase', 'brightness_decrease', 'occlusion',
            'noise_low', 'noise_high', 'blur_low', 'blur_high',
            'rotation_left', 'rotation_right'
        ],
        'per_person_stats': {},
        'errors': []
    }

    # 收集所有需要处理的图片
    image_tasks = []
    for root, _, files in os.walk(TEST_IMAGE_FOLDER):
        person_name = os.path.basename(root)
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, img_file)
                output_dir = os.path.join(AUGMENTED_IMAGE_FOLDER, person_name)
                image_tasks.append((img_path, person_name, output_dir))

    # 设置进程池
    num_processes = max(1, multiprocessing.cpu_count() - 1)  # 保留一个CPU核心
    print(f"\n使用 {num_processes} 个进程进行并行处理...")

    # 使用进程池处理图片
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_single_image, image_tasks),
            total=len(image_tasks),
            desc="处理图片"
        ))

    # 处理结果
    for result in results:
        person_name = result['person_name']
        
        # 初始化个人统计信息
        if person_name not in augmentation_info['per_person_stats']:
            augmentation_info['per_person_stats'][person_name] = {
                'original_count': 0,
                'augmented_count': 0,
                'error_count': 0
            }
        
        # 更新统计信息
        if result['success']:
            augmentation_info['per_person_stats'][person_name]['original_count'] += result['original_count']
            augmentation_info['per_person_stats'][person_name]['augmented_count'] += result['augmented_count']
        else:
            augmentation_info['per_person_stats'][person_name]['error_count'] += 1
            augmentation_info['errors'].append({
                'person': person_name,
                'error': result['error']
            })

    # 计算总数
    augmentation_info['total_original_images'] = sum(
        stats['original_count'] for stats in augmentation_info['per_person_stats'].values()
    )
    augmentation_info['total_augmented_images'] = sum(
        stats['augmented_count'] for stats in augmentation_info['per_person_stats'].values()
    )
    augmentation_info['total_errors'] = len(augmentation_info['errors'])

    # 保存增强信息
    with open('../augmentation_info.json', 'w') as f:
        json.dump(augmentation_info, f, indent=4)

    # 打印统计信息
    print(f"\n数据增强完成:")
    print(f"原始图片总数: {augmentation_info['total_original_images']}")
    print(f"增强后图片总数: {augmentation_info['total_augmented_images']}")
    print(f"处理失败数量: {augmentation_info['total_errors']}")
    print(f"增强信息已保存到: augmentation_info.json")

    # 如果有错误，打印错误信息
    if augmentation_info['errors']:
        print("\n处理过程中的错误:")
        for error in augmentation_info['errors']:
            print(f"- {error['person']}: {error['error']}")

if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    augment_images()

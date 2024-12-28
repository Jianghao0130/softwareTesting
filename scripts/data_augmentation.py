from PIL import Image, ImageEnhance
import os
import sys
import shutil

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

# 扩增所有图片
def augment_images():
    if not check_permissions():
        print("权限检查失败，程序退出")
        sys.exit(1)

    success_count = 0
    error_count = 0
    
    # 递归遍历所有子文件夹
    for root, _, files in os.walk(TEST_IMAGE_FOLDER):
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, img_file)
                
                # 在输出目录中创建相同的目录结构
                rel_path = os.path.relpath(root, TEST_IMAGE_FOLDER)
                output_dir = os.path.join(AUGMENTED_IMAGE_FOLDER, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    # 光线变化
                    augmented_image = simulate_lighting(img_path)
                    if augmented_image:
                        augmented_image.save(os.path.join(output_dir, f'bright_{img_file}'))
                    
                    # 遮挡
                    occluded_image = simulate_occlusion(img_path)
                    if occluded_image:
                        occluded_image.save(os.path.join(output_dir, f'occluded_{img_file}'))
                    
                    success_count += 1
                    if success_count % 100 == 0:  # 每处理100张图片显示一次进度
                        print(f"已成功处理 {success_count} 张图片")
                        
                except Exception as e:
                    print(f"处理图片 {img_path} 时出错: {e}")
                    error_count += 1

    print(f"\n数据增强完成:")
    print(f"成功处理: {success_count} 张图片")
    print(f"处理失败: {error_count} 张图片")
    print(f"增强后的图片保存在: {AUGMENTED_IMAGE_FOLDER}")

if __name__ == "__main__":
    augment_images()

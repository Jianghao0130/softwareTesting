from PIL import Image, ImageEnhance
import os

# 测试图片路径
TEST_IMAGE_FOLDER = 'test_images/'
AUGMENTED_IMAGE_FOLDER = 'test_images_augmented/'

# 模拟光线变化
def simulate_lighting(image_path, brightness=1.5):
    image = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness)

# 模拟遮挡
def simulate_occlusion(image_path):
    image = Image.open(image_path)
    overlay = Image.new('RGB', image.size, (0, 0, 0))  # 黑色遮挡
    mask = Image.new('L', image.size, 128)  # 半透明遮挡
    image.paste(overlay, (image.size[0] // 4, image.size[1] // 4), mask)
    return image

# 扩增所有图片
def augment_images():
    os.makedirs(AUGMENTED_IMAGE_FOLDER, exist_ok=True)
    for img_file in os.listdir(TEST_IMAGE_FOLDER):
        img_path = os.path.join(TEST_IMAGE_FOLDER, img_file)
        # 光线变化
        augmented_image = simulate_lighting(img_path)
        augmented_image.save(os.path.join(AUGMENTED_IMAGE_FOLDER, f'bright_{img_file}'))

        # 遮挡
        occluded_image = simulate_occlusion(img_path)
        occluded_image.save(os.path.join(AUGMENTED_IMAGE_FOLDER, f'occluded_{img_file}'))
    print(f"Augmented images saved to {AUGMENTED_IMAGE_FOLDER}")

if __name__ == "__main__":
    augment_images()

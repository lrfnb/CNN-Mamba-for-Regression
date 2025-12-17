import os
import random
import numpy as np
from PIL import Image
import torch

from torchvision.transforms import transforms
import torchvision.transforms.functional as tf

from data.base_dataset import BaseDataset


class RemoteSensingResolutionDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.shuffle = True if opt.isTrain else False
        self.img_size = opt.load_size
        self.img_dir = opt.dataroot
        self.img_names = self.get_img_names()

        self.base_resolution = 0.1

        self.high_res_label_range = (0.01, 0.1)
        self.medium_res_label_range = (0.1, 0.4)
        self.low_res_label_range = (0.4, 1.0)

        self.rotation_angles = [0, 90, 180, 270]

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_img_names(self):
        img_names = [x for x in os.listdir(self.img_dir)
                     if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        if self.shuffle:
            random.shuffle(img_names)
        return img_names

    def __len__(self):
        return len(self.img_names)

    def generate_uniform_resolution_labels(self):
        high_res_label = np.random.uniform(
            self.high_res_label_range[0],
            self.high_res_label_range[1]
        )

        medium_res_label = np.random.uniform(
            self.medium_res_label_range[0],
            self.medium_res_label_range[1]
        )

        low_res_label = np.random.uniform(
            self.low_res_label_range[0],
            self.low_res_label_range[1]
        )

        return high_res_label, medium_res_label, low_res_label

    def calculate_sampling_factor_from_label(self, target_label):

        if target_label < self.base_resolution:
            factor = self.base_resolution / target_label
            mode = 'up'
        else:
            factor = target_label / self.base_resolution
            mode = 'down'

        return factor, mode

    def apply_simple_augmentation(self, img):
        if random.random() > 0.5:
            angle = random.choice(self.rotation_angles)
            if angle != 0:
                img = tf.rotate(img, angle)

        if random.random() > 0.5:
            img = tf.hflip(img)

        if random.random() > 0.5:
            img = tf.vflip(img)

        return img

    def process_image_with_sampling(self, img, factor, mode='up'):
        original_size = img.size

        if mode == 'up':
            new_size = (int(original_size[0] * factor), int(original_size[1] * factor))
            img_scaled = img.resize(new_size, Image.BICUBIC)
            img_resized = img_scaled.resize(original_size, Image.BICUBIC)
        else:
            new_size = (int(original_size[0] / factor), int(original_size[1] / factor))
            img_scaled = img.resize(new_size, Image.BICUBIC)
            img_resized = img_scaled.resize(original_size, Image.BICUBIC)

        return img_resized

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        original_img = Image.open(img_path).convert('RGB')

        if original_img.size != (self.img_size, self.img_size):
            original_img = original_img.resize((self.img_size, self.img_size), Image.BICUBIC)

        high_res_label, medium_res_label, low_res_label = self.generate_uniform_resolution_labels()

        high_res_factor, high_res_mode = self.calculate_sampling_factor_from_label(high_res_label)
        medium_res_factor, medium_res_mode = self.calculate_sampling_factor_from_label(medium_res_label)
        low_res_factor, low_res_mode = self.calculate_sampling_factor_from_label(low_res_label)

        high_res_img = self.process_image_with_sampling(original_img, high_res_factor, mode=high_res_mode)
        medium_res_img = self.process_image_with_sampling(original_img, medium_res_factor, mode=medium_res_mode)
        low_res_img = self.process_image_with_sampling(original_img, low_res_factor, mode=low_res_mode)

        high_res_img = self.apply_simple_augmentation(high_res_img)
        medium_res_img = self.apply_simple_augmentation(medium_res_img)
        low_res_img = self.apply_simple_augmentation(low_res_img)

        high_res_tensor = self.to_tensor(high_res_img)
        medium_res_tensor = self.to_tensor(medium_res_img)
        low_res_tensor = self.to_tensor(low_res_img)

        concatenated_images = torch.cat([
            high_res_tensor,
            medium_res_tensor,
            low_res_tensor
        ], dim=0)

        resolution_labels = torch.tensor([
            high_res_label,
            medium_res_label,
            low_res_label
        ], dtype=torch.float32)

        # print(f"=== Dataset __getitem__ Debug Info ===")
        # print(f"Image path: {img_path}")
        # print(f"Resolution labels: [{high_res_label:.6f}, {medium_res_label:.6f}, {low_res_label:.6f}]")
        # print(f"Sampling factors: high={high_res_factor:.3f}({high_res_mode}), medium={medium_res_factor:.3f}({medium_res_mode}), low={low_res_factor:.3f}({low_res_mode})")
        # print(f"Label ranges - High: {self.high_res_label_range}, Medium: {self.medium_res_label_range}, Low: {self.low_res_label_range}")
        # print(f"Concatenated images shape: {concatenated_images.shape}")
        # print(f"Resolution labels shape: {resolution_labels.shape}")
        # print("=" * 50)

        return {
            'images': concatenated_images,  # [9, H, W] 拼接后的图像
            'labels': resolution_labels,  # [3] 对应的分辨率标签
            'img_paths': img_path,  # 原始图像路径
            'high_res_factor': high_res_factor,  # 高分辨率采样倍率 (用于调试)
            'medium_res_factor': medium_res_factor,  # 中分辨率采样倍率 (用于调试)
            'low_res_factor': low_res_factor  # 低分辨率采样倍率 (用于调试)
        }


def test_distribution_uniformity():
    import matplotlib.pyplot as plt

    high_labels, medium_labels, low_labels = [], [], []

    for _ in range(10000):
        high = np.random.uniform(0.01, 0.1)
        medium = np.random.uniform(0.1, 0.4)
        low = np.random.uniform(0.4, 1.0)

        high_labels.append(high)
        medium_labels.append(medium)
        low_labels.append(low)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(high_labels, bins=50, alpha=0.7, color='blue')
    axes[0].set_title('High Resolution Labels (0.01-0.1)')
    axes[0].set_xlabel('Resolution Value')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(medium_labels, bins=50, alpha=0.7, color='green')
    axes[1].set_title('Medium Resolution Labels (0.1-0.4)')
    axes[1].set_xlabel('Resolution Value')
    axes[1].set_ylabel('Frequency')

    axes[2].hist(low_labels, bins=50, alpha=0.7, color='red')
    axes[2].set_title('Low Resolution Labels (0.4-1.0)')
    axes[2].set_xlabel('Resolution Value')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('resolution_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Distribution test completed. Check 'resolution_distribution.png' for results.")


# test_distribution_uniformity()

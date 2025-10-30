# datasets.py

import os
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import random
import cv2
from sklearn.model_selection import train_test_split

import custom_transforms as tr
import config

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MASK_EXTENSIONS = ('.png', '.tif', '.tiff', '.bmp')

def make_dataset(root, dataset_name, split='train', val_size=0.15, test_size=0.15, random_state=42):
    """
    Creates a list of (image_path, mask_path) tuples for a given dataset and split.
    Handles both pre-split datasets and flat datasets that need programmatic splitting.
    """
    dataset_config = config.DATASET_CONFIG[dataset_name]
    structure = dataset_config.get('structure')
    all_pairs = []

    if structure == 'TSRS_RSNA':
        split_root = os.path.join(root, split)
        if not os.path.exists(split_root):
            print(f"Warning: Directory not found for pre-split dataset: {split_root}")
            return []
        mask_dir = os.path.join(split_root + '_labels')
        if not os.path.exists(mask_dir):
            print(f"Warning: Mask directory not found: {mask_dir}")
            return []
        img_names = [os.path.splitext(f)[0] for f in os.listdir(split_root) if f.lower().endswith(IMAGE_EXTENSIONS)]
        for name in img_names:
            img_path = os.path.join(split_root, name + '.jpg')
            mask_path = os.path.join(mask_dir, name + '.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))
        return all_pairs

    elif structure == 'FLAT_SPLIT':
        if dataset_name == 'JSRT':
            image_dir = os.path.join(root, 'cxr')
            mask_dir = os.path.join(root, 'masks')
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.png')]
                for name in img_names:
                    all_pairs.append((os.path.join(image_dir, name + '.png'), os.path.join(mask_dir, name + '.png')))

        elif dataset_name == 'CVC-ClinicDB':
            image_dir = os.path.join(root, 'Original')
            mask_dir = os.path.join(root, 'Ground Truth')
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
                for name in img_names:
                    all_pairs.append((os.path.join(image_dir, name + '.tif'), os.path.join(mask_dir, name + '.tif')))

        elif dataset_name == 'DentalPanoramic':
            image_dir = os.path.join(root, 'images')
            mask_dir = os.path.join(root, 'segmentation_1')
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                img_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
                for name in img_names:
                    all_pairs.append((os.path.join(image_dir, name + '.jpg'), os.path.join(mask_dir, name + '.png')))

        if not all_pairs:
            print(f"Warning: Found 0 image-mask pairs for '{dataset_name}' at root '{root}'.")
            return []

        train_val_pairs, test_pairs = train_test_split(all_pairs, test_size=test_size, random_state=random_state)
        val_proportion = val_size / (1 - test_size)
        train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_proportion, random_state=random_state)

        if split == 'train': return train_pairs
        elif split == 'val': return val_pairs
        elif split == 'test': return test_pairs
        else: return []
    else:
        raise ValueError(f"Unknown dataset structure '{structure}' for dataset '{dataset_name}'.")


class ImageFolder(data.Dataset):
    def __init__(self, root, dataset_name, args, split='train', imgs=None):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.args = args

        if imgs is not None:
            self.imgs = imgs
        else:
            # --- THE ONLY CHANGE IS ON THIS LINE ---
            # Pass the 'split' variable to the make_dataset function
            self.imgs = make_dataset(self.root, self.dataset_name, self.split)
            # ----------------------------------------

        if not self.imgs:
            if imgs is None:
                 print(f"Warning: No images found for dataset '{self.dataset_name}', split '{self.split}' at root '{self.root}'.")
            self.imgs = []

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if dataset_name in ['JSRT', 'COVID19_Radiography']:
            self.mean = [0.5]
            self.std = [0.5]

        if self.split == 'train':
            self.composed_transforms = transforms.Compose([
                tr.RandomResizedCrop(size=(args.scale_h, args.scale_w), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                tr.RandomHorizontalFlip(),
                tr.ElasticTransform(alpha=35, sigma=5, p=0.4),
                tr.GridDistortion(num_steps=5, distort_limit=0.2, p=0.4),
                tr.RandomGaussianBlur(),
                tr.ColorJitter(brightness=0.2, contrast=0.2),
                tr.RandomAffine(degrees=7, translate=(0.05, 0.05), shear=5),
                tr.Normalize(mean=self.mean, std=self.std),
                tr.ToTensor()
            ])
        else:
            self.composed_transforms = transforms.Compose([
                tr.ProportionalResizePad(output_size=args.scale_h),
                tr.Normalize(mean=self.mean, std=self.std),
                tr.ToTensor()
            ])

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        try:
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                raise FileNotFoundError(f"OpenCV could not read image: {img_path}. File might be corrupted or path incorrect.")

            if img_cv.ndim == 3 and img_cv.shape[2] == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            elif img_cv.ndim == 2:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

            img = Image.fromarray(img_cv)

            mask_cv = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_cv is None:
                raise FileNotFoundError(f"OpenCV could not read mask: {gt_path}. File might be corrupted or path incorrect.")

            target = Image.fromarray(mask_cv, mode='L')
            label = self.convert_label(target)

        except (UnidentifiedImageError, FileNotFoundError, cv2.error, ValueError, Exception) as e:
            print(f"ERROR: Could not open/process image or mask for paths: {img_path}, {gt_path}. Error: {e}. Returning None for this sample.")
            return None

        sample = {'image': img, 'label': label}
        transformed_sample = self.composed_transforms(sample)

        if self.split != 'train':
            transformed_sample['name'] = os.path.basename(img_path)

        return transformed_sample

    def convert_label(self, label):
        label_np = np.array(label, dtype=np.uint8)
        if label_np.ndim == 3 and label_np.shape[2] == 1:
            label_np = label_np.squeeze(2)

        label_index = np.zeros_like(label_np, dtype=np.uint8)
        label_index[label_np > 0] = 1

        return Image.fromarray(label_index, mode='P')

    def __len__(self):
        return len(self.imgs)
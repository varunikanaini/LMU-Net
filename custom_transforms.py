import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from torchvision import transforms
import pywt
import random
from torchvision.transforms import functional as TF
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import cv2

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.8, 1.0), ratio=(3./4., 4./3.)):
        self.transform = transforms.RandomResizedCrop(size, scale, ratio, interpolation=transforms.InterpolationMode.LANCZOS)
        self.transform_label = transforms.RandomResizedCrop(size, scale, ratio, interpolation=transforms.InterpolationMode.NEAREST)

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        
        # Get parameters for the random crop
        i, j, h, w = self.transform.get_params(img, self.transform.scale, self.transform.ratio)
        
        # Apply the same crop to both image and label
        img = TF.resized_crop(img, i, j, h, w, self.transform.size, self.transform.interpolation)
        label = TF.resized_crop(label, i, j, h, w, self.transform.size, self.transform_label.interpolation)
        
        return {'image': img, 'label': label}
    
class ProportionalResizePad(object):
    """
    Resizes an image and its label to a target size while maintaining aspect ratio,
    padding the shorter side to create a square image.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        w, h = img.size
        long_side = max(w, h)
        
        # Calculate new dimensions while preserving aspect ratio
        new_w = int(w / long_side * self.output_size)
        new_h = int(h / long_side * self.output_size)

        # Resize using a high-quality filter for the image and nearest for the mask
        # Note: Image.LANCZOS is the modern replacement for the deprecated Image.ANTIALIAS
        img = img.resize((new_w, new_h), Image.LANCZOS) 
        label = label.resize((new_w, new_h), Image.NEAREST)

        # Calculate padding
        delta_w = self.output_size - new_w
        delta_h = self.output_size - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

        # Apply padding (fill=0 makes the padding black, which is correct for background)
        img = ImageOps.expand(img, padding, fill=0)
        label = ImageOps.expand(label, padding, fill=0)

        return {'image': img, 'label': label}

class GridDistortion(object):
    """
    Applies grid distortion on a PIL image and its corresponding mask.
    This augmentation is effective at simulating lens and perspective distortions.
    
    Args:
        num_steps (int): The number of grid steps on each side.
        distort_limit (float): The maximum distortion limit. 0.0 means no distortion.
        p (float): The probability of applying the transform.
    """
    def __init__(self, num_steps=5, distort_limit=0.3, p=0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p

    def __call__(self, sample):
        if np.random.rand() > self.p:
            return sample

        img, label = sample['image'], sample['label']
        
        # Convert PIL to numpy (OpenCV format)
        img_np = np.array(img)
        label_np = np.array(label)

        h, w = img_np.shape[:2]

        # Create the grid
        x_steps = np.linspace(0, w, self.num_steps + 1)
        y_steps = np.linspace(0, h, self.num_steps + 1)

        # Generate random distortions
        dx = np.random.uniform(-self.distort_limit, self.distort_limit, (self.num_steps + 1, self.num_steps + 1))
        dy = np.random.uniform(-self.distort_limit, self.distort_limit, (self.num_steps + 1, self.num_steps + 1))
        
        # Scale distortions by grid cell size
        dx *= w / self.num_steps
        dy *= h / self.num_steps

        # Create the distorted grid points
        xx, yy = np.meshgrid(x_steps, y_steps)
        xx_distorted = xx + dx
        yy_distorted = yy + dy

        # Build the map for cv2.remap
        map_x = np.zeros_like(img_np, dtype=np.float32)
        map_y = np.zeros_like(img_np, dtype=np.float32)

        # Interpolate the distorted grid to the full image size
        for i in range(self.num_steps):
            for j in range(self.num_steps):
                src_rect = np.array([
                    [yy[i, j], xx[i, j]],
                    [yy[i, j+1], xx[i, j+1]],
                    [yy[i+1, j], xx[i+1, j]],
                    [yy[i+1, j+1], xx[i+1, j+1]]
                ], dtype=np.float32)
                
                dst_rect = np.array([
                    [yy_distorted[i, j], xx_distorted[i, j]],
                    [yy_distorted[i, j+1], xx_distorted[i, j+1]],
                    [yy_distorted[i+1, j], xx_distorted[i+1, j]],
                    [yy_distorted[i+1, j+1], xx_distorted[i+1, j+1]]
                ], dtype=np.float32)
        
        # Generate random displacements for the grid
        dx = (np.random.rand(self.num_steps + 1, self.num_steps + 1) * 2 - 1) * self.distort_limit
        dy = (np.random.rand(self.num_steps + 1, self.num_steps + 1) * 2 - 1) * self.distort_limit

        # Resize the coarse displacement maps to the full image size
        map_dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_LINEAR)
        map_dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Scale the displacement maps
        map_dx *= w / self.num_steps
        map_dy *= h / self.num_steps

        # Create the final remapping grids
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + map_dx).astype(np.float32)
        map_y = (grid_y + map_dy).astype(np.float32)

        # Apply the remapping
        # Use BILINEAR for the image for smooth results
        transformed_img_np = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # Use NEAREST for the mask to preserve discrete label values
        transformed_label_np = cv2.remap(label_np, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Convert back to PIL
        img = Image.fromarray(transformed_img_np)
        label = Image.fromarray(transformed_label_np)
        
        return {'image': img, 'label': label}

class ElasticTransform(object):
    """
    Apply elastic deformation on a PIL image and its corresponding mask.
    Based on the excellent implementation from:
    https://github.com/albu/albumentations/blob/master/albumentations/augmentations/geometric/functional.py
    """
    def __init__(self, alpha, sigma, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if np.random.rand() > self.p:
            return sample

        img, label = sample['image'], sample['label']
        
        # Convert PIL to numpy
        img_np = np.array(img)
        label_np = np.array(label)

        # Get the 2D shape of the image
        shape = img_np.shape[:2]

        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        # Create coordinate grid
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Apply transformation to the label (order=0 for nearest neighbor interpolation)
        transformed_label_np = map_coordinates(label_np, indices, order=0, mode='reflect').reshape(shape)

        # Apply transformation to each channel of the image (order=1 for bilinear interpolation)
        if len(img_np.shape) == 3 and img_np.shape[2] > 1:
            channels = [map_coordinates(img_np[..., i], indices, order=1, mode='reflect').reshape(shape) for i in range(img_np.shape[2])]
            transformed_img_np = np.stack(channels, axis=-1)
        else: # Handle grayscale images
            transformed_img_np = map_coordinates(img_np, indices, order=1, mode='reflect').reshape(shape)
        
        # Convert back to PIL
        img = Image.fromarray(transformed_img_np.astype(np.uint8))
        label = Image.fromarray(transformed_label_np.astype(np.uint8))
        
        return {'image': img, 'label': label}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': label}

class FixedResize(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img.resize((self.w, self.h), Image.BILINEAR)
        label = label.resize((self.w, self.h), Image.NEAREST)
        return {'image': img, 'label': label}

class RandomCrop(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return {'image': img, 'label': label}

        if h < th or w < tw:
            img = img.resize((tw, th), Image.BILINEAR)
            label = label.resize((tw, th), Image.NEAREST)
            return {'image': img, 'label': label}

        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)

        img = img.crop((j, i, j + tw, i + th))
        label = label.crop((j, i, j + tw, i + th))
        return {'image': img, 'label': label}

class RandomGaussianBlur(object):
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if np.random.rand() < 0.5:
            radius = np.random.uniform(self.radius_range[0], self.radius_range[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return {'image': img, 'label': label}

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, self.mean, self.std)
        return {'image': img, 'label': label}

class ToTensor(object):
    """Converts the PIL label to a PyTorch LongTensor. Image is assumed to be already a Tensor."""
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label_np = np.array(label)
        # Binarize if necessary (assuming 0/255 for binary masks)
        if label_np.max() > 1:
            label_np = (label_np > 127).astype(np.int64)
        else:
            label_np = label_np.astype(np.int64)
        label = torch.from_numpy(label_np).long()
        return {'image': img, 'label': label}

class CenterAmplification(object):
    """
    Applies center-based amplification to small lesions, as described in STS-Net Method 4.
    Operates on PIL Images. This transform should ideally be applied AFTER FixedResize
    and BEFORE RandomCrop.
    """
    def __init__(self, min_lesion_area_pixels=576, expansion_factor=1.5, min_bbox_size=(32, 32)):
        self.min_lesion_area_pixels = min_lesion_area_pixels
        self.expansion_factor = expansion_factor
        self.min_bbox_size = min_bbox_size  # (h, w) for minimum expanded bbox size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label_np = np.array(label)

        lesion_coords = np.argwhere(label_np > 0)
        lesion_area = len(lesion_coords)

        if 0 < lesion_area < self.min_lesion_area_pixels:
            y_min, x_min = lesion_coords.min(axis=0)
            y_max, x_max = lesion_coords.max(axis=0)

            h_lesion, w_lesion = y_max - y_min + 1, x_max - x_min + 1
            center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2

            expanded_h = max(self.min_bbox_size[0], int(h_lesion * self.expansion_factor))
            expanded_w = max(self.min_bbox_size[1], int(w_lesion * self.expansion_factor))

            img_w, img_h = img.size

            cx1_raw = center_x - expanded_w // 2
            cy1_raw = center_y - expanded_h // 2

            cx2_raw = center_x + (expanded_w // 2) + (expanded_w % 2)
            cy2_raw = center_y + (expanded_h // 2) + (expanded_h % 2)

            crop_x_min = max(0, cx1_raw)
            crop_y_min = max(0, cy1_raw)
            crop_x_max = min(img_w, cx2_raw)
            crop_y_max = min(img_h, cy2_raw)

            current_crop_w = crop_x_max - crop_x_min
            current_crop_h = crop_y_max - crop_y_min

            if current_crop_w < expanded_w:
                if crop_x_min == 0:
                    crop_x_max = min(img_w, crop_x_min + expanded_w)
                else:
                    crop_x_min = max(0, crop_x_max - expanded_w)

            if current_crop_h < expanded_h:
                if crop_y_min == 0:
                    crop_y_max = min(img_h, crop_y_min + expanded_h)
                else:
                    crop_y_min = max(0, crop_y_max - expanded_h)

            crop_x_max = max(crop_x_min + self.min_bbox_size[1], crop_x_max)
            crop_y_max = max(crop_y_min + self.min_bbox_size[0], crop_y_max)
            crop_x_max = min(img_w, crop_x_max)
            crop_y_max = min(img_h, crop_y_max)
            crop_x_min = max(0, crop_x_max - self.min_bbox_size[1])
            crop_y_min = max(0, crop_y_max - self.min_bbox_size[0])

            img_cropped = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            label_cropped = label.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

            target_img_size = img.size
            img = img_cropped.resize(target_img_size, Image.BILINEAR)
            label = label_cropped.resize(target_img_size, Image.NEAREST)

        return {'image': img, 'label': label}

class HistogramEqualization(object):
    """
    Applies histogram equalization to the PIL image for contrast enhancement.
    If RGB, converts to YCbCr, equalizes Y channel, then converts back to RGB.
    """
    def __call__(self, sample):
        img = sample['image']
        if img.mode == 'RGB':
            img_ycbcr = img.convert('YCbCr')
            Y, Cb, Cr = img_ycbcr.split()
            Y_eq = ImageOps.equalize(Y)
            img_eq = Image.merge('YCbCr', (Y_eq, Cb, Cr)).convert('RGB')
        else:  # For grayscale images
            img_eq = ImageOps.equalize(img)
        sample['image'] = img_eq
        return sample

class WaveletContrastEnhancement(object):
    """
    Applies Discrete Wavelet Transform (DWT) based contrast enhancement.
    This version performs level 1 decomposition, scales detail coefficients,
    and reconstructs the image. Operates on grayscale for simplicity.
    """
    def __init__(self, wavelet='haar', level=1, detail_scale_factor=1.5):
        self.wavelet = wavelet
        self.level = level
        self.detail_scale_factor = detail_scale_factor

        if self.wavelet not in pywt.wavelist(kind='discrete'):
            print(f"Warning: Wavelet '{self.wavelet}' not found. Falling back to 'haar'.")
            self.wavelet = 'haar'

    def __call__(self, sample):
        img_pil = sample['image']

        if img_pil.mode == 'RGB':
            img_gray = img_pil.convert('L')
            original_mode = 'RGB'
        else:
            img_gray = img_pil
            original_mode = 'L'

        img_np = np.array(img_gray, dtype=np.float32) / 255.0

        # Perform 2D DWT
        coeffs_list = pywt.wavedec2(img_np, self.wavelet, mode='periodization', level=self.level)

        # Modify detail coefficients directly within the coeffs_list structure
        modified_coeffs_list = [coeffs_list[0]]  # Approximation coefficients (cA)
        for d_level in coeffs_list[1:]:  # Iterate through each level's detail coefficients tuple (cH, cV, cD)
            cH, cV, cD = d_level
            cH_e = cH * self.detail_scale_factor
            cV_e = cV * self.detail_scale_factor
            cD_e = cD * self.detail_scale_factor
            modified_coeffs_list.append((cH_e, cV_e, cD_e))  # Add modified tuple back to the list

        # Reconstruct the image from the modified coefficients list
        img_reconstructed = pywt.waverec2(modified_coeffs_list, self.wavelet, mode='periodization')

        img_reconstructed = np.clip(img_reconstructed, 0, 1)
        img_enhanced_pil = Image.fromarray((img_reconstructed * 255).astype(np.uint8))

        if original_mode == 'RGB':
            img_enhanced_pil = img_enhanced_pil.convert('RGB')

        sample['image'] = img_enhanced_pil
        return sample

class RandomCutout(object):
    """
    Randomly masks out one or more rectangular patches from an image and its label.
    Args:
        num_holes_range (tuple): Range (min, max) for number of patches to cut out.
        max_h_size (int): Maximum height of each cutout patch (pixels).
        max_w_size (int): Maximum width of each cutout patch (pixels).
        fill_value (int or tuple): Value to fill the cutout in the image (0 for black).
        p (float): Probability of applying cutout.
    """
    def __init__(self, num_holes_range=(1, 2), max_h_size=32, max_w_size=32, fill_value=0, p=0.3):
        self.num_holes_range = num_holes_range
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        img, label = sample['image'], sample['label']
        w, h = img.size
        num_holes = random.randint(self.num_holes_range[0], self.num_holes_range[1])

        for _ in range(num_holes):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            x2 = min(w, x1 + random.randint(1, self.max_w_size))
            y2 = min(h, y1 + random.randint(1, self.max_h_size))

            # Apply cutout to image
            draw_img = ImageDraw.Draw(img)
            draw_img.rectangle([x1, y1, x2, y2], fill=self.fill_value)

            # Apply cutout to label (set to background 0)
            draw_label = ImageDraw.Draw(label)
            draw_label.rectangle([x1, y1, x2, y2], fill=0)

        return {'image': img, 'label': label}

class RandomAffine(object):
    """
    Applies random affine transformations (rotation, translation, scale, shear) to image and label.
    Args:
        degrees (float): Range of rotation degrees (-degrees, +degrees).
        translate (tuple): Max translation as fraction of image size (dx, dy).
        scale (tuple): Range of scaling factors (min, max).
        shear (float): Range of shear angles in degrees (-shear, +shear).
        mask_fill_value (int): Fill value for areas outside the transformed mask.
    """
    def __init__(self, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, mask_fill_value=0):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.mask_fill_value = mask_fill_value

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        # Generate random affine parameters
        angle = random.uniform(-self.degrees, self.degrees)
        translate = (random.uniform(-self.translate[0], self.translate[0]) * img.size[0],
                     random.uniform(-self.translate[1], self.translate[1]) * img.size[1])
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(-self.shear, self.shear)

        # Apply affine transformation to image
        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR)

        # Apply affine transformation to label (use NEAREST to preserve label values)
        label = TF.affine(label, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST, fill=self.mask_fill_value)

        return {'image': img, 'label': label}

class ColorJitter(object):
    """
    Applies random adjustments to brightness, contrast, saturation, and hue of the image.
    Args:
        brightness (float): Max adjustment factor for brightness (0 to disable).
        contrast (float): Max adjustment factor for contrast (0 to disable).
        saturation (float): Max adjustment factor for saturation (0 to disable).
        hue (float): Max adjustment factor for hue (0 to disable).
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = TF.adjust_brightness(img, brightness_factor)
        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            img = TF.adjust_contrast(img, contrast_factor)
        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            img = TF.adjust_saturation(img, saturation_factor)
        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            img = TF.adjust_hue(img, hue_factor)
        return {'image': img, 'label': label} 
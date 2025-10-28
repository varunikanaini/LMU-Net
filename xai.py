# /kaggle/working/LMU-Net/xai.py

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import cv2
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader

# --- Setup Project Path ---
project_path = '/kaggle/working/LMU-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Custom Module Imports ---
import config
from light_lasa_unet import Light_LASA_Unet
from datasets import ImageFolder, make_dataset
from misc import check_mkdir, AvgMeter

# --- CAM Imports ---
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM,
    EigenCAM, EigenGradCAM, LayerCAM, FullGrad
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[-1]

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

# --- Helper Functions ---
def denormalize(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def get_args():
    parser = argparse.ArgumentParser(description='Generate CAM visualizations for a trained segmentation model')
    parser.add_argument('--exp-name', type=str, required=True, help='Name of the experiment folder in ./ckpt')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()), help='Name of the dataset as defined in config.py')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam', 'layercam', 'fullgrad'],
                        help='CAM visualization method')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to visualize the results for.')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to visualize from the test set.')
    parser.add_argument('--scale-h', type=int, default=512, help='Target height for image resizing.')
    parser.add_argument('--scale-w', type=int, default=512, help='Target width for image resizing.')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--target-class', type=int, default=1, help='The class index to generate explanations for (e.g., 1 for epiphysis).')


    args = parser.parse_args()
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path, args.num_classes = dataset_info['path'], dataset_info['num_classes']
    for k, v in config.DEFAULT_ARGS.items():
        if not hasattr(args, k): setattr(args, k, v)
    return args

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model = Light_LASA_Unet(num_classes=args.num_classes, lasa_kernels=args.lasa_kernels).to(device)
    checkpoint_path = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    # --- CAM Setup ---
    target_layers = [model.bottleneck_layer[-1]]
    cam_algorithm = {
        'gradcam': GradCAM, 'hirescam': HiResCAM, 'scorecam': ScoreCAM, 'gradcam++': GradCAMPlusPlus,
        'ablationcam': AblationCAM, 'xgradcam': XGradCAM, 'eigencam': EigenCAM,
        'eigengradcam': EigenGradCAM, 'layercam': LayerCAM, 'fullgrad': FullGrad
    }
    cam = cam_algorithm[args.method](model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # --- Load Dataset ---
    def custom_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch) if batch else None

    test_data_path = os.path.join(args.dataset_path, 'test')
    test_dataset = ImageFolder(root=test_data_path, dataset_name=args.dataset_name, args=args, split='test')
    if len(test_dataset) == 0:
        print(f"Error: No images found in the test directory: {test_data_path}")
        sys.exit(1)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate_fn
    )

    # --- Run Visualization ---
    output_dir = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", "xai_visualizations", args.method)
    check_mkdir(output_dir)
    print(f"Saving visualizations to: {output_dir}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if args.dataset_name in ['JSRT', 'COVID19_Radiography']:
        mean, std = [0.5], [0.5]

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader, desc=f"Generating {args.method.upper()} maps")):
            if i >= args.num_images:
                break
            if sample is None:
                continue

            image_tensor = sample['image'].to(device)
            image_name = sample.get('name', [f'image_{i}'])[0]
            
            # --- Correctly handle single-channel vs multi-channel images ---
            rgb_img_tensor = image_tensor.clone().squeeze(0).cpu()
            if rgb_img_tensor.shape[0] == 1: # Grayscale
                rgb_img_tensor = rgb_img_tensor.repeat(3, 1, 1) # Convert to 3 channels for visualization
            
            rgb_img_denorm = denormalize(rgb_img_tensor, mean, std)
            rgb_img_np = np.transpose(rgb_img_denorm.numpy(), (1, 2, 0))
            rgb_img_np = np.clip(rgb_img_np, 0, 1)

            # --- Generate CAM ---
            targets = [SemanticSegmentationTarget(args.target_class,
                                                 (sample['label'].squeeze().numpy() == args.target_class).astype(np.float32))]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
            
            # --- Plotting ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(rgb_img_np)
            axes[0].set_title(f"Original: {os.path.basename(image_name)}")
            axes[0].axis('off')

            axes[1].imshow(cam_image)
            axes[1].set_title(f"{args.method.upper()} Overlay")
            axes[1].axis('off')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}_{args.method}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"\n{args.method.upper()} visualization complete.")

if __name__ == '__main__':
    main()
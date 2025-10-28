# /kaggle/working/LMU-Net/xai.py (Final Version with Correct Targeting Logic)

import torch
import numpy as np
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
from datasets import ImageFolder
from misc import check_mkdir

# --- CAM Imports ---
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM,
    EigenCAM, EigenGradCAM, LayerCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Helper Classes ---

class SegmentationClassTarget:
    """
    Correct target for segmentation. The goal is to see which pixels
    contributed most to the overall activation of a particular class.
    We do this by summing the output logits for that class across all pixels.
    """
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        # model_output has shape (B, C, H, W) or (C, H, W)
        # We want to sum the activations for our target category across the spatial dimensions.
        if len(model_output.shape) == 4:
            return model_output[0, self.category, :, :].sum()
        return model_output[self.category, :, :].sum()

# --- Helper Functions ---
def denormalize(tensor, mean, std):
    if tensor.shape[0] == 1 and len(mean) == 3: mean, std = [mean[0]], [std[0]]
    elif tensor.shape[0] == 3 and len(mean) == 1: mean, std = mean * 3, std * 3
    for t, m, s in zip(tensor, mean, std): t.mul_(s).add_(m)
    return tensor

def get_args():
    parser = argparse.ArgumentParser(description='Generate CAM visualizations for a trained segmentation model')
    parser.add_argument('--exp-name', type=str, required=True, help='Name of the experiment folder in ./ckpt')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam', 'layercam'],
                        help='CAM visualization method')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num-images', type=int, default=20)
    parser.add_argument('--scale-h', type=int, default=512)
    parser.add_argument('--scale-w', type=int, default=512)
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
    model = Light_LASA_Unet(num_classes=args.num_classes, lasa_kernels=args.lasa_kernels)
    checkpoint_path = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", 'best_checkpoint.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}"); sys.exit(1)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}"); sys.exit(1)

    model.to(device).eval()

    # --- CAM Setup ---
    # Using the more specific target layer from your successful script for better localization
    target_layers = [model.bottleneck_layer[-1].conv[0]]

    cam_algorithm = {
        'gradcam': GradCAM, 'hirescam': HiResCAM, 'scorecam': ScoreCAM, 'gradcam++': GradCAMPlusPlus,
        'ablationcam': AblationCAM, 'xgradcam': XGradCAM, 'eigencam': EigenCAM,
        'eigengradcam': EigenGradCAM, 'layercam': LayerCAM
    }
    
    # We pass the raw model. The CAM library is smart enough to handle the tuple output
    # by default, just like in your successful script.
    cam_class = cam_algorithm[args.method]
    cam = cam_class(model=model, target_layers=target_layers)

    # --- Load Dataset ---
    test_data_path = os.path.join(args.dataset_path, 'test')
    test_dataset = ImageFolder(root=test_data_path, dataset_name=args.dataset_name, args=args, split='test')
    if len(test_dataset) == 0:
        print(f"Error: No images found in the test directory: {test_data_path}"); sys.exit(1)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x is not None])
    )

    # --- Run Visualization ---
    output_dir = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", "xai_visualizations", args.method)
    check_mkdir(output_dir)
    print(f"Saving visualizations to: {output_dir}")

    for i, sample in enumerate(tqdm(test_loader, desc=f"Generating {args.method.upper()} maps")):
        if i >= args.num_images: break
        if sample is None: continue

        input_tensor = sample['image'].to(device)
        image_name = sample.get('name', [f'image_{i}'])[0]

        mean = [0.5] if args.dataset_name in ['JSRT', 'COVID19_Radiography'] else [0.485, 0.456, 0.406]
        std = [0.5] if args.dataset_name in ['JSRT', 'COVID19_Radiography'] else [0.229, 0.224, 0.225]
        rgb_img_tensor = input_tensor.clone().squeeze(0).cpu()
        if rgb_img_tensor.shape[0] == 1: rgb_img_tensor = rgb_img_tensor.repeat(3, 1, 1)
        rgb_img_np = np.clip(np.transpose(denormalize(rgb_img_tensor, mean, std).numpy(), (1, 2, 0)), 0, 1)
        
        # Using the correct, simpler target class
        targets = [SegmentationClassTarget(args.target_class)]

        call_kwargs = {'input_tensor': input_tensor, 'targets': targets}
        if args.method in ['scorecam', 'ablationcam', 'layercam']: call_kwargs['show_progress'] = False
        if args.method == 'hirescam': call_kwargs['eigen_smooth'], call_kwargs['aug_smooth'] = False, False
        else: call_kwargs['eigen_smooth'], call_kwargs['aug_smooth'] = True, True
            
        grayscale_cam = cam(**call_kwargs)[0, :]

        cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_img_np); axes[0].set_title(f"Original: {os.path.basename(image_name)}"); axes[0].axis('off')
        axes[1].imshow(cam_image); axes[1].set_title(f"{args.method.upper()} Overlay"); axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}_{args.method}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\n{args.method.upper()} visualization complete.")

if __name__ == '__main__':
    main()
# /kaggle/working/ARAA-Net/grad_cam_explain.py (Corrected)

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
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Custom Module Imports ---
import config
from light_lasa_unet import Light_LASA_Unet
from datasets import ImageFolder
from misc import check_mkdir

# --- Grad-CAM Imports ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- NEW: Custom Target for Segmentation Models ---
class SegmentationClassTarget:
    """
    A target class for Grad-CAM on segmentation models.
    This tells Grad-CAM to use the activations of a specific class
    across the entire spatial map as the target for backpropagation.
    """
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        # The model returns a tuple of outputs (due to deep supervision).
        # We must select the final, main output for our explanation.
        final_output = model_output[-1]
        
        # We are working with a batch size of 1, so we select the first element.
        # Then, we select the channel corresponding to our target category.
        # The result is a 2D map of activations for our class.
        return final_output[0, self.category, :, :]

# ... (Helper 'denormalize' and 'get_args' functions remain the same) ...
def denormalize(tensor, mean, std):
    if tensor.shape[0] == 1:
        mean, std = [mean[0]], [std[0]]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def get_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM Explanations for a Trained Model')
    parser.add_argument('--exp-name', type=str, required=True, help='Name of the experiment folder in ./ckpt')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--fold', type=int, default=0, help='Fold number to generate explanations for.')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to generate explanations for.')
    parser.add_argument('--scale-h', type=int, default=512, help='Target height for image resizing.')
    parser.add_argument('--scale-w', type=int, default=512, help='Target width for image resizing.')
    parser.add_argument('--target-class', type=int, default=1, help='The class index to generate explanations for (e.g., 1 for epiphysis).')
    args = parser.parse_args()
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path, args.num_classes = dataset_info['path'], dataset_info['num_classes']
    return args

# --- Core XAI Logic (Corrected) ---
def generate_grad_cam_explanations(model, loader, device, args):
    target_layer = model.bottleneck_layer[-1][0]
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    output_dir = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", "grad_cam_explanations")
    check_mkdir(output_dir)
    print(f"Saving Grad-CAM visualizations to: {output_dir}")

    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    if args.dataset_name in ['JSRT', 'COVID19_Radiography']:
        mean, std = [0.5], [0.5]

    for i, sample in enumerate(tqdm(loader, desc="Generating Grad-CAM")):
        if i >= args.num_images: break
        if sample is None: continue

        input_tensor = sample['image'].to(device)
        image_name = sample.get('name', [f'image_{i}'])[0]

        # --- FIX: Use our new, simpler target class ---
        targets = [SegmentationClassTarget(args.target_class)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0, :]
        
        original_img = denormalize(input_tensor.clone().squeeze(0).cpu(), mean, std)
        original_img = np.transpose(original_img.numpy(), (1, 2, 0))
        original_img = np.clip(original_img, 0, 1)

        visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow((original_img * 255).astype(np.uint8))
        axes[0].set_title(f"Original: {image_name}"); axes[0].axis('off')
        axes[1].imshow(visualization)
        axes[1].set_title("Grad-CAM Overlay"); axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_grad_cam.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print("\nGrad-CAM generation complete.")

# --- Main Execution (Unchanged) ---
if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Light_LASA_Unet(num_classes=args.num_classes, lasa_kernels=config.DEFAULT_ARGS.get('lasa_kernels')).to(device)
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
    
    test_data_path = os.path.join(args.dataset_path, 'test')
    test_dataset = ImageFolder(root=test_data_path, dataset_name=args.dataset_name, args=args, split='test')
    
    if len(test_dataset) == 0:
        print(f"Error: No images found in the test directory: {test_data_path}")
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    generate_grad_cam_explanations(model, test_loader, device, args)
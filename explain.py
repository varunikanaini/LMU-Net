# /kaggle/working/ARAA-Net/explain.py

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
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Custom Module Imports ---
import config
from light_lasa_unet import Light_LASA_Unet
from datasets import ImageFolder
from misc import check_mkdir

# --- Global list to store attention maps from the hook ---
attention_maps = []

def get_attention_map_hook(module, input, output):
    """
    Forward hook to capture the attention map from the LASA module.
    The attention map is calculated as: output / (input + epsilon)
    """
    # input is a tuple containing the tensor, so we take the first element
    input_tensor = input[0]
    # Add a small epsilon to avoid division by zero
    attention_map = output / (input_tensor + 1e-8)
    attention_maps.append(attention_map.detach())

# --- Helper Functions ---
def denormalize(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    # Handle both grayscale (1 channel) and RGB (3 channels)
    if tensor.shape[0] == 1: # Grayscale
        mean = [mean[0]]
        std = [std[0]]
    
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Generate XAI Attention Maps for a Trained Model')
    parser.add_argument('--exp-name', type=str, required=True, help='Name of the experiment folder in ./ckpt')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()), help='Name of the dataset as defined in config.py')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2', choices=['mobilenet_v2'], help='Backbone used for the model.')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to generate explanations for.')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to generate explanations for.')
    parser.add_argument('--scale-h', type=int, default=256, help='Target height for image resizing.')
    parser.add_argument('--scale-w', type=int, default=256, help='Target width for image resizing.')
    parser.add_argument('--num-workers', type=int, default=2)

    args = parser.parse_args()
    
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path, args.num_classes = dataset_info['path'], dataset_info['num_classes']
    
    for k, v in config.DEFAULT_ARGS.items():
        if not hasattr(args, k): setattr(args, k, v)
            
    return args

# --- Core XAI Logic ---
def generate_explanations(model, loader, device, args):
    model.eval()
    
    output_dir = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", "explanations")
    check_mkdir(output_dir)
    print(f"Saving XAI visualizations to: {output_dir}")

    # --- Register the hook on the LASA module ---
    hook_handle = model.lasa_module.register_forward_hook(get_attention_map_hook)
    
    # Define mean and std for denormalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if args.dataset_name in ['JSRT', 'COVID19_Radiography']:
        mean, std = [0.5], [0.5]

    with torch.no_grad():
        for i, sample in enumerate(tqdm(loader, desc="Generating Explanations")):
            if i >= args.num_images:
                break
            if sample is None:
                continue

            image_tensor = sample['image'].to(device)
            image_name = sample.get('name', [f'image_{i}'])[0]

            # Clear the global list before running inference
            attention_maps.clear()
            
            # --- Run Inference ---
            # The hook will automatically populate the `attention_maps` list
            _ = model(image_tensor)

            if not attention_maps:
                print(f"Warning: Could not capture attention map for {image_name}. Skipping.")
                continue

            # --- Process Captured Attention Map ---
            captured_map = attention_maps[0].squeeze(0).cpu().numpy()
            
            # Average across channels to get a single 2D map
            attn_map_2d = np.mean(captured_map, axis=0)
            
            # Resize map to original image dimensions
            attn_map_resized = cv2.resize(attn_map_2d, (args.scale_w, args.scale_h), interpolation=cv2.INTER_LINEAR)
            
            # Normalize to 0-1 range
            attn_map_normalized = (attn_map_resized - np.min(attn_map_resized)) / (np.max(attn_map_resized) - np.min(attn_map_resized))
            
            # Apply a colormap to create a heatmap
            heatmap = (plt.get_cmap('jet')(attn_map_normalized)[:, :, :3] * 255).astype(np.uint8)
            heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

            # --- Prepare Original Image ---
            original_img_tensor = denormalize(image_tensor.clone().squeeze(0).cpu(), mean, std)
            original_img_np = np.transpose(original_img_tensor.numpy(), (1, 2, 0))
            original_img_np = np.clip(original_img_np * 255, 0, 255).astype(np.uint8)
            original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

            # --- Create Overlay ---
            overlay = cv2.addWeighted(original_img_bgr, 0.6, heatmap_bgr, 0.4, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # --- Plotting ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(original_img_np)
            axes[0].set_title(f"Original: {image_name}")
            axes[0].axis('off')

            axes[1].imshow(overlay_rgb)
            axes[1].set_title("Attention Map Overlay")
            axes[1].axis('off')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_explanation.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    # --- IMPORTANT: Remove the hook when done ---
    hook_handle.remove()
    print("\nExplanation generation complete.")

# --- Main Execution ---
if __name__ == '__main__':
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
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    # --- Run XAI Generation ---
    generate_explanations(model, test_loader, device, args)
# /kaggle/working/LMU-Net/visualize.py

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

project_path = '/kaggle/working/LMU-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import config
from light_lasa_unet import Light_LASA_Unet
from datasets import ImageFolder
from misc import check_mkdir

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def apply_color_map(mask_np, color_map):
    height, width = mask_np.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in enumerate(color_map):
        if class_idx < color_map.shape[0]:
            colored_mask[mask_np == class_idx] = color
    return colored_mask

def get_args():
    parser = argparse.ArgumentParser(description='Visualize Segmentation Model Results')
    parser.add_argument('--exp-name', type=str, required=True, help='Name of the experiment folder in ./ckpt')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--backbone', type=str, default='vgg16', choices=list(config.BACKBONE_CHANNELS.keys()) + ['mobilenet_v2'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num-images', type=int, default=20, help='Max number of images to visualize.')
    # --- ADDED ARGUMENT TO SELECT SPLIT ---
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to visualize.')
    parser.add_argument('--scale-h', type=int, default=256)
    parser.add_argument('--scale-w', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path, args.num_classes = dataset_info['path'], dataset_info['num_classes']
    for k, v in config.DEFAULT_ARGS.items():
        if not hasattr(args, k): setattr(args, k, v)
    return args

def visualize_and_save(model, loader, device, args, color_map):
    model.eval()
    # Output directory now includes the split name
    output_dir = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", f"visualizations_{args.split}")
    check_mkdir(output_dir)
    print(f"Saving visualizations for '{args.split}' split to: {output_dir}")

    mean, std = ([0.5], [0.5]) if args.dataset_name in ['JSRT', 'MontgomeryCounty'] else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    with torch.no_grad():
        for i, sample in enumerate(tqdm(loader, desc="Generating Visualizations")):
            if i >= args.num_images: break
            if sample is None: continue
            
            image_tensor, label_tensor = sample['image'].to(device), sample['label']
            image_name = sample.get('name', [f'image_{i}.png'])[0]
            
            prediction = model(image_tensor)[-1].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            original_img_np = np.transpose(denormalize(image_tensor.clone().squeeze(0).cpu(), mean, std).numpy(), (1, 2, 0))
            original_img_np = np.clip(original_img_np * 255, 0, 255).astype(np.uint8)
            if len(mean) == 1: original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2RGB)
            
            gt_colored = apply_color_map(label_tensor.squeeze(0).numpy().astype(np.uint8), color_map)
            pred_colored = apply_color_map(prediction, color_map)
            overlay = cv2.addWeighted(original_img_np, 0.6, pred_colored, 0.4, 0)
            
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            titles = [f"Original: {image_name}", "Ground Truth", "Prediction", "Prediction Overlay"]
            images = [original_img_np, gt_colored, pred_colored, overlay]
            for ax, img, title in zip(axes, images, titles):
                ax.imshow(img); ax.set_title(title); ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_vis.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
    print("\nVisualization complete.")

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    COLOR_MAP = np.array([[0, 0, 0], [255, 0, 0]]) # Assuming binary: Black=BG, Red=Foreground
    
    model = Light_LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone).to(device)
    checkpoint_path = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}"); sys.exit(1)
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Model loaded successfully from {checkpoint_path}")

    def custom_collate_fn(batch):
        return torch.utils.data.dataloader.default_collate([item for item in batch if item is not None])

    # --- UPDATED: Loads data based on the --split argument ---
    print(f"Loading '{args.split}' split from dataset '{args.dataset_name}'...")
    dataset = ImageFolder(root=args.dataset_path, dataset_name=args.dataset_name, args=args, split=args.split)
    
    if len(dataset) == 0:
        print(f"Error: No images found for the '{args.split}' split in {args.dataset_path}"); sys.exit(1)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)

    visualize_and_save(model, loader, device, args, COLOR_MAP)
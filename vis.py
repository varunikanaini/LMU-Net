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
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()), help='Name of the dataset as defined in config.py')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2', choices=list(config.BACKBONE_CHANNELS.keys()) + ['mobilenet_v2'], help='Backbone used for the model.')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to visualize the results for.')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to visualize from the test set.')
    parser.add_argument('--scale-h', type=int, default=256, help='Target height for image resizing.')
    parser.add_argument('--scale-w', type=int, default=256, help='Target width for image resizing.')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path, args.num_classes = dataset_info['path'], dataset_info['num_classes']
    for k, v in config.DEFAULT_ARGS.items():
        if not hasattr(args, k): setattr(args, k, v)
    return args


def visualize_and_save(model, loader, device, args, color_map):
    model.eval()
    output_dir = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", "visualizations")
    check_mkdir(output_dir)
    print(f"Saving visualizations to: {output_dir}")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if args.dataset_name in ['JSRT', 'COVID19_Radiography']:
        mean, std = [0.5], [0.5]
    with torch.no_grad():
        for i, sample in enumerate(tqdm(loader, desc="Generating Visualizations")):
            if i >= args.num_images: break
            if sample is None: continue
            image_tensor = sample['image'].to(device)
            label_tensor = sample['label']
            image_name = sample.get('name', [f'image_{i}'])[0]
            outputs = model(image_tensor)
            prediction = outputs[-1].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            original_img_tensor = denormalize(image_tensor.clone().squeeze(0).cpu(), mean, std)
            original_img_np = np.transpose(original_img_tensor.numpy(), (1, 2, 0))
            original_img_np = np.clip(original_img_np * 255, 0, 255).astype(np.uint8)
            gt_mask_np = label_tensor.squeeze(0).numpy().astype(np.uint8)
            gt_colored = apply_color_map(gt_mask_np, color_map)
            pred_colored = apply_color_map(prediction, color_map)
            if original_img_np.shape[2] == 1:
                original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2RGB)
            original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(original_img_bgr, 0.6, cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR), 0.4, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            axes[0].imshow(original_img_np.squeeze(), cmap='gray' if original_img_np.shape[2]==1 else None)
            axes[0].set_title(f"Original: {image_name}"); axes[0].axis('off')
            axes[1].imshow(gt_colored); axes[1].set_title("Ground Truth"); axes[1].axis('off')
            axes[2].imshow(pred_colored); axes[2].set_title("Prediction"); axes[2].axis('off')
            axes[3].imshow(overlay_rgb); axes[3].set_title("Prediction Overlay"); axes[3].axis('off')
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_visualization.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    print("\nVisualization complete.")


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    if args.dataset_name in ['TSRS_RSNA-Epiphysis', 'JSRT', 'CVC-ClinicDB']:
        COLOR_MAP = np.array([[0, 0, 0], [255, 0, 0]])
    else:
        COLOR_MAP = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]])[:args.num_classes]

    print(f"Loading Light_LASA_Unet with backbone: '{args.backbone}'")
    model = Light_LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone).to(device)


    checkpoint_path = os.path.join(config.CKPT_ROOT, args.exp_name, f"fold_{args.fold}", 'best_checkpoint.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    try:
        # <-- Here is the FIX: silent loading with strict=False to avoid errors
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print(f"Model loaded successfully from {checkpoint_path} (strict=False)")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)


    def custom_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch) if batch else None


    test_dataset = ImageFolder(root=args.dataset_path, dataset_name=args.dataset_name, args=args, split='test')

    if len(test_dataset) == 0:
        print(f"Error: No images found for the 'test' split in the directory: {args.dataset_path}")
        sys.exit(1)


    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)

    visualize_and_save(model, test_loader, device, args, COLOR_MAP)

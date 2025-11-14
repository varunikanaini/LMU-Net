import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from torch.utils.data import DataLoader

# Update with your model import
from light_lasa_unet import Light_LASA_Unet  # or your model class
from datasets import ImageFolder  # your dataset class
from misc import check_mkdir  # util function for folder creation

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def apply_color_map(mask_np, color_map):
    h, w = mask_np.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(color_map):
        colored_mask[mask_np == idx] = color
    return colored_mask

def enhance_mask(gt_mask):
    # Example enhancement: edge highlighting
    edges = cv2.Canny(gt_mask.astype(np.uint8) * 255, 50, 150)
    enhanced = np.stack([gt_mask * 255, edges, edges], axis=-1)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

def visualize_and_save(model, loader, device, args, color_map):
    model.eval()
    output_dir = os.path.join(args.output_dir, "visualizations")
    check_mkdir(output_dir)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    with torch.no_grad():
        for idx, sample in enumerate(loader):
            if idx >= args.num_images:
                break

            image_tensor = sample['image'].to(device)
            label_tensor = sample['label']
            image_name = sample.get('name', [f'image_{idx}'])[0]

            outputs = model(image_tensor)
            pred = outputs[-1].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Denormalize image
            img_denorm = denormalize(image_tensor.clone().squeeze(0).cpu(), mean, std)
            img_np = np.transpose(img_denorm.numpy(), (1, 2, 0))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            # Original GT mask and enhanced mask
            gt_mask_np = label_tensor.squeeze(0).numpy().astype(np.uint8)
            gt_colored = apply_color_map(gt_mask_np, color_map)
            gt_enhanced = enhance_mask(gt_mask_np)

            # Prediction colored
            pred_colored = apply_color_map(pred, color_map)

            if img_np.shape[2] == 1:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            overlay = cv2.addWeighted(img_np, 0.6, pred_colored, 0.4, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            fig, axes = plt.subplots(1, 5, figsize=(24, 5))
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[1].imshow(gt_colored)
            axes[1].set_title('GT Mask')
            axes[2].imshow(gt_enhanced)
            axes[2].set_title('Enhanced GT')
            axes[3].imshow(pred_colored)
            axes[3].set_title('Prediction')
            axes[4].imshow(overlay_rgb)
            axes[4].set_title('Overlay Prediction')
            for ax in axes:
                ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_vis.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved visualization: {save_path}")

def get_args():
    parser = argparse.ArgumentParser(description="Visualize Segmentation Results")
    parser.add_argument('--exp-name', type=str, required=True, help='Experiment folder')
    parser.add_argument('--dataset-name', type=str, required=True, help='Dataset name')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2', help='Model backbone')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to visualize')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory to save visuals')
    args = parser.parse_args()

    # Add your dataset config here or import
    import config
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path = dataset_info['path']
    args.num_classes = dataset_info['num_classes']

    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup color map - adjust or extend as per dataset
    COLOR_MAP = np.array([
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Class 1 - red
        [0, 255, 0],      # Class 2 - green
        [0, 0, 255],      # Class 3 - blue
        [255, 255, 0],    # Class 4 - yellow
        [255, 0, 255],    # Class 5 - magenta
        [0, 255, 255]     # Class 6 - cyan
    ])[:args.num_classes]

    # Load model
    model = Light_LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone).to(device)

    checkpoint_path = os.path.join('ckpt', args.exp_name, f'fold_{args.fold}', 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Create dataset and loader
    test_dataset = ImageFolder(root=args.dataset_path, dataset_name=args.dataset_name, split='test', args=args)
    if len(test_dataset) == 0:
        raise RuntimeError(f"No images found in test split at {args.dataset_path}")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    visualize_and_save(model, test_loader, device, args, COLOR_MAP)

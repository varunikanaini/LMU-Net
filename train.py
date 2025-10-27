# /kaggle/working/ARAA-Net/train.py

import sys, os, logging, argparse, torch, numpy as np, json
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.model_selection import KFold

project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path: sys.path.insert(0, project_path)

import config
from light_lasa_unet import Light_LASA_Unet
from datasets import ImageFolder, make_dataset
from seg_utils import ConfusionMatrix
from misc import AvgMeter, check_mkdir

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2): super(FocalLoss, self).__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, i, t):
        ce = F.cross_entropy(i, t, reduction='none'); pt = torch.exp(-ce)
        fl = self.alpha * (1 - pt)**self.gamma * ce; return fl.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super(DiceLoss, self).__init__(); self.smooth = smooth
    def forward(self, i, t):
        p, to = F.softmax(i, dim=1)[:, 1], (t == 1).float()
        inter = (p * to).sum()
        return 1 - ((2. * inter + self.smooth) / (p.sum() + to.sum() + self.smooth))

def freeze_backbone(model):
    for n, p in model.named_parameters():
        if 'encoder' in n: p.requires_grad = False
    logging.info("--- Encoder FROZEN ---")

def unfreeze_backbone(model):
    for n, p in model.named_parameters():
        if 'encoder' in n: p.requires_grad = True
    logging.info("--- Encoder UN-FROZEN ---")

def create_boundary_mask(labels):
    labels_np = labels.clone().cpu().numpy()
    boundary_masks = []
    for i in range(labels_np.shape[0]):
        single_mask = labels_np[i, 0, :, :]
        dilated = binary_dilation(single_mask, iterations=1)
        eroded = binary_erosion(single_mask, iterations=1)
        boundary = np.logical_xor(dilated, eroded)
        boundary_masks.append(boundary)
    return torch.from_numpy(np.array(boundary_masks)).float().unsqueeze(1).to(labels.device)

# --- Argument Parsing --- 
def get_args():
    parser = argparse.ArgumentParser(description='Train Segmentation Models')
    parser.add_argument('--dataset-name', type=str, required=True, choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--backbone', type=str, default='vgg19', choices=list(config.BACKBONE_CHANNELS.keys()))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lasa-kernels', type=int, nargs='+', default=[1, 3, 5, 7])
    parser.add_argument('--deep-supervision-weights', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--focal-loss-weight', type=float, default=0.5)
    parser.add_argument('--dice-loss-weight', type=float, default=1.5)
    parser.add_argument('--scheduler-type', type=str, default='CosineAnnealingWarmRestarts', choices=['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'])
    parser.add_argument('--scheduler-T0', type=int, default=15)
    parser.add_argument('--scheduler-T-mult', type=int, default=2)
    parser.add_argument('--scheduler-patience', type=int, default=10)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--fine-tune-epochs', type=int, default=40)
    parser.add_argument('--boundary-loss-weight', type=float, default=1.5, help='Weight for the dedicated boundary loss.')
    parser.add_argument('--scale-h', type=int, default=None, help='Target height for image resizing.')
    parser.add_argument('--scale-w', type=int, default=None, help='Target width for image resizing.')
    parser.add_argument('--k-folds', type=int, default=1, help='Number of folds for cross-validation. Default is 1 (no k-fold).')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for KFold split for reproducibility.')
    args = parser.parse_args()
    dataset_info = config.DATASET_CONFIG[args.dataset_name]
    args.dataset_path, args.num_classes = dataset_info['path'], dataset_info['num_classes']
    if args.scale_h is None or args.scale_w is None:
        res_h, res_w = config.get_backbone_resolution(args.backbone)
        if args.scale_h is None: args.scale_h = res_h
        if args.scale_w is None: args.scale_w = res_w
    for k, v in config.DEFAULT_ARGS.items():
        if not hasattr(args, k): setattr(args, k, v)
    return args

# --- Logging and Collate --- 
def setup_logging(log_dir, filename='training.log'):
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(os.path.join(log_dir, filename)), logging.StreamHandler()])

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


# --- Evaluation Function ---
def evaluate_model(net, data_loader, device, focal_loss_fn, dice_loss_fn, args, mode="Validating"):
    net.eval()
    confmat, loss_recorder = ConfusionMatrix(args.num_classes), AvgMeter()
    with torch.no_grad():
        for data in tqdm(data_loader, desc=mode, leave=False):
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(inputs)
            total_loss = 0
            for i, pred in enumerate(outputs):
                f_l, d_l = focal_loss_fn(pred, labels.long()), dice_loss_fn(pred, labels.long())
                total_loss += args.deep_supervision_weights[i] * ((args.focal_loss_weight * f_l) + (args.dice_loss_weight * d_l))
            loss_recorder.update(total_loss.item(), inputs.size(0))
            confmat.update(labels.flatten(), outputs[-1].argmax(1).flatten())
            
    # Unpack all metrics from the confusion matrix
    oa, _, iou, fwiou, dice = confmat.compute()
    miou = iou.mean().item()
    oa = oa.item()  # convert to float

    # Log all the metrics
    logging.info(f"--- {mode} Summary --- Loss: {loss_recorder.avg:.4f}, OA: {oa:.4f}, mIoU: {miou:.4f}, FW-IoU: {fwiou.item():.4f}, Dice: {dice:.4f}")

    if mode.startswith("Validating"):  # Make it more robust for "Validating Fold X"
        net.train()
        
    return oa, miou, fwiou.item(), dice

# --- Testing Function ---
def test(args):
    device = torch.device("cuda")
    exp_name = f"{args.backbone}_FreezeTune_LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    base_exp_path = os.path.join(config.CKPT_ROOT, exp_name)
    
    setup_logging(base_exp_path, 'main_training_log.log')
    logging.info("\n" + "="*50 + "\n" + " " * 20 + "STARTING TESTING" + "\n" + "="*50)

    test_ds = ImageFolder(os.path.join(args.dataset_path, 'test'), args.dataset_name, args, 'test')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    
    if args.backbone == 'mobilenet_v2':
        net = Light_LASA_Unet(num_classes=args.num_classes, lasa_kernels=args.lasa_kernels).to(device)
    else:
        print("Error: Testing currently only supports 'mobilenet_v2' backbone.")
        
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)

    all_fold_metrics = {'oa': [], 'miou': [], 'fwiou': [], 'dice': []}
    num_folds_to_test = args.k_folds if args.k_folds > 1 else 1

    for fold_idx in range(num_folds_to_test):
        fold_exp_path = os.path.join(base_exp_path, f"fold_{fold_idx}")
        ckpt_path = os.path.join(fold_exp_path, 'best_checkpoint.pth')

        if not os.path.exists(ckpt_path):
            logging.warning(f"Checkpoint for fold {fold_idx} not found at {ckpt_path}. Skipping.")
            continue

        logging.info(f"--- Loading model for Fold {fold_idx} from {ckpt_path} ---")
        net.load_state_dict(torch.load(ckpt_path, map_location=device))

        oa, miou, fwiou, dice = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, args, mode=f"Testing Fold {fold_idx}")
        
        all_fold_metrics['oa'].append(oa)
        all_fold_metrics['miou'].append(miou)
        all_fold_metrics['fwiou'].append(fwiou)
        all_fold_metrics['dice'].append(dice)

    logging.info("\n" + "="*50 + "\n" + " " * 15 + "FINAL TESTING SUMMARY" + "\n" + "="*50)
    
    if not all_fold_metrics['miou']:
        logging.error("No models were tested. Cannot compute final metrics.")
        return

    logging.info(f"Metrics calculated over {len(all_fold_metrics['miou'])} tested fold(s).")
    logging.info(f"Overall Accuracy (OA): {np.mean(all_fold_metrics['oa']):.4f} ± {np.std(all_fold_metrics['oa']):.4f}")
    logging.info(f"Mean IoU (mIoU):     {np.mean(all_fold_metrics['miou']):.4f} ± {np.std(all_fold_metrics['miou']):.4f}")
    logging.info(f"FW-IoU:              {np.mean(all_fold_metrics['fwiou']):.4f} ± {np.std(all_fold_metrics['fwiou']):.4f}")
    logging.info(f"Dice Score:          {np.mean(all_fold_metrics['dice']):.4f} ± {np.std(all_fold_metrics['dice']):.4f}")
    logging.info("="*50)

def train_fold(args, fold_idx, train_imgs, val_imgs):
    device = torch.device("cuda")
    exp_name = f"{args.backbone}_FreezeTune_LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    fold_exp_path = os.path.join(config.CKPT_ROOT, exp_name, f"fold_{fold_idx}")
    check_mkdir(fold_exp_path)
    setup_logging(fold_exp_path, f'fold_{fold_idx}_training.log')

    logging.info(f"===== Starting Fold {fold_idx}/{args.k_folds if args.k_folds > 1 else 1} =====")
    
    train_ds = ImageFolder(root=None, dataset_name=args.dataset_name, args=args, split='train', imgs=train_imgs)
    val_ds = ImageFolder(root=None, dataset_name=args.dataset_name, args=args, split='val', imgs=val_imgs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)

    if args.backbone == 'mobilenet_v2':
        net = Light_LASA_Unet(num_classes=args.num_classes, lasa_kernels=args.lasa_kernels).to(device)
    else:
        print("Error: Training currently only supports 'mobilenet_v2' backbone.")
    
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    dice_loss_fn = DiceLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2, eta_min=1e-6) if args.scheduler_type == 'CosineAnnealingWarmRestarts' else optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args.scheduler_patience)
    
    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    best_ckpt_path = os.path.join(fold_exp_path, 'best_checkpoint.pth')
    latest_ckpt_path = os.path.join(fold_exp_path, 'latest_checkpoint.pth')
    
    if args.resume and os.path.exists(latest_ckpt_path):
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming fold {fold_idx} from epoch {start_epoch}. Best mIoU: {best_mIoU:.4f}.")
        except Exception as e:
            logging.error(f"Could not resume fold {fold_idx}: {e}. Starting this fold from scratch.")
            start_epoch = 0

    if start_epoch < args.fine_tune_epochs: freeze_backbone(net)
    else: unfreeze_backbone(net)

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.fine_tune_epochs:
            unfreeze_backbone(net)
            new_lr = args.lr / 5.0
            optimizer = optim.Adam(net.parameters(), lr=new_lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2, eta_min=1e-6)
            logging.info(f"--- Switched to Phase 2. New LR: {new_lr} ---")
            
        net.train()
        loss_recorder = AvgMeter()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Fold {fold_idx})")
        
        for data in train_iterator:
            if data is None: continue
            inputs, labels = data['image'].to(device), data['label'].to(device)
            labels_float_unsqueezed = labels.unsqueeze(1).float()
            optimizer.zero_grad(set_to_none=True)
            outputs = net(inputs)
            main_loss = 0
            for head_idx, pred_output in enumerate(outputs):
                f_loss = focal_loss_fn(pred_output, labels.long())
                d_loss = dice_loss_fn(pred_output, labels.long())
                main_loss += args.deep_supervision_weights[head_idx] * ((args.focal_loss_weight * f_loss) + (args.dice_loss_weight * d_loss))
            boundary_mask = create_boundary_mask(labels_float_unsqueezed)
            final_pred_logits = outputs[-1][:, 1, :, :].unsqueeze(1)
            boundary_bce_loss = F.binary_cross_entropy_with_logits(final_pred_logits, boundary_mask, reduction='none')
            boundary_loss = (boundary_bce_loss * boundary_mask).mean()
            total_loss = main_loss + (args.boundary_loss_weight * boundary_loss)
            total_loss.backward()
            optimizer.step()
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
        
        _, current_mIoU, _, _ = evaluate_model(net, val_loader, device, focal_loss_fn, dice_loss_fn, args, mode=f"Validating Fold {fold_idx}")
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(current_mIoU)
        else: scheduler.step()
        
        if current_mIoU > best_mIoU:
            best_mIoU, patience_counter = current_mIoU, 0
            torch.save(net.state_dict(), best_ckpt_path)
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Model saved for fold {fold_idx}.")
        else:
            patience_counter += 1
            logging.info(f"⚠️ mIoU did not improve for {patience_counter} epoch(s). Best: {best_mIoU:.4f}")
        
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_mIoU': best_mIoU, 'patience_counter': patience_counter}, latest_ckpt_path)

        if patience_counter >= args.patience:
            logging.info(f"Early stopping triggered for fold {fold_idx}.")
            break
            
    logging.info(f"===== Finished Fold {fold_idx} ===== Best Val mIoU: {best_mIoU:.4f}")
    return best_mIoU

def main():
    args = get_args()
    
    if args.test_only:
        test(args)
        return

    exp_name = f"{args.backbone}_FreezeTune_LASA_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    base_exp_path = os.path.join(config.CKPT_ROOT, exp_name)
    check_mkdir(base_exp_path)
    
    setup_logging(base_exp_path, 'main_training_log.log')
    logging.info(f"Starting experiment: '{exp_name}'\nArguments: {vars(args)}")

    if args.k_folds > 1:
        train_path, val_path = os.path.join(args.dataset_path, 'train'), os.path.join(args.dataset_path, 'val')
        all_imgs = np.array(make_dataset(train_path, args.dataset_name) + make_dataset(val_path, args.dataset_name))
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
        kfold_state_path = os.path.join(base_exp_path, 'kfold_state.json')
        start_fold, all_fold_metrics = 0, {}

        if args.resume and os.path.exists(kfold_state_path):
            with open(kfold_state_path, 'r') as f:
                state = json.load(f)
                start_fold = state.get('next_fold_to_run', 0)
                all_fold_metrics = state.get('all_fold_metrics', {})
            logging.info(f"Resuming k-fold training. Starting from fold {start_fold}.")

        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_imgs)):
            if fold_idx < start_fold: continue
            with open(kfold_state_path, 'w') as f: json.dump({'next_fold_to_run': fold_idx, 'all_fold_metrics': all_fold_metrics}, f)
            fold_train_imgs, fold_val_imgs = all_imgs[train_indices].tolist(), all_imgs[val_indices].tolist()
            best_fold_mIoU = train_fold(args, fold_idx, fold_train_imgs, fold_val_imgs)
            all_fold_metrics[f'fold_{fold_idx}'] = best_fold_mIoU
            with open(kfold_state_path, 'w') as f: json.dump({'next_fold_to_run': fold_idx + 1, 'all_fold_metrics': all_fold_metrics}, f)

        logging.info("===== K-Fold Training Finished =====")
        mean_mIoU = np.mean(list(all_fold_metrics.values())); std_mIoU = np.std(list(all_fold_metrics.values()))
        logging.info(f"Metrics across {args.k_folds} folds: {all_fold_metrics}")
        logging.info(f"Average Validation mIoU: {mean_mIoU:.4f} ± {std_mIoU:.4f}")
    else:
        logging.info("Running a single train/validation split (k_folds=1).")
        train_ds = ImageFolder(os.path.join(args.dataset_path, 'train'), args.dataset_name, args, 'train')
        val_ds = ImageFolder(os.path.join(args.dataset_path, 'val'), args.dataset_name, args, 'val')
        train_fold(args, 0, train_ds.imgs, val_ds.imgs)

if __name__ == '__main__':
    main()
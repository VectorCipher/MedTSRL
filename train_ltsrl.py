import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json
import random
import time
from glob import glob
import numpy as np

from dataset import ManipDensityDataset
from dataset_wrapper import DatasetWrapper, collate_keep_boxes_with_index
from train_utils import compute_loss, eval_loader, iou as box_iou
from manip_density_detect import density_to_boxes
from transunet_density import TransUNetDensity
from agents import TutorPPO, StateManager

def build_or_load_fixed_splits(data_dir, img_dir, save_dir, val_ratio=0.10, test_ratio=0.05):
    split_json = os.path.join(save_dir, "split_fixed_density.json")
    
    def list_ids():
        imgs = sorted(glob(os.path.join(img_dir, "*.png")))
        return [os.path.splitext(os.path.basename(p))[0] for p in imgs]
        
    ids = list_ids()
    if os.path.exists(split_json):
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        fixed_val = set(split["val"])
        fixed_test = set(split["test"])
        train = [i for i in ids if i not in (fixed_val | fixed_test)]
        val   = [i for i in ids if i in fixed_val]
        test  = [i for i in ids if i in fixed_test]
        return train, val, test

    ids_shuf = ids.copy()
    random.shuffle(ids_shuf)
    n = len(ids_shuf)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))

    test = ids_shuf[:n_test]
    val  = ids_shuf[n_test:n_test+n_val]
    train = ids_shuf[n_test+n_val:]
    
    os.makedirs(save_dir, exist_ok=True)
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump({"val": val, "test": test, "created_at": time.strftime("%F %T")}, f, indent=2)
    return train, val, test

def calculate_dice(pred_msk, gt_msk):
    pred = (torch.sigmoid(pred_msk) > 0.5).float()
    intersection = (pred * gt_msk).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + gt_msk.sum(dim=(2,3))
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.squeeze(1)

def calculate_batch_iou_and_recall(pred_den, gt_boxes_list, thr=0.28, min_area=20):
    pd = pred_den.detach().cpu().numpy()
    ious = []
    recalls = []
    for i in range(pd.shape[0]):
        d = pd[i,0]
        pred_boxes = density_to_boxes(d, thr=thr, min_area=min_area)
        gt_boxes = gt_boxes_list[i]
        
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            ious.append(1.0)
            recalls.append(1.0)
        elif len(gt_boxes) == 0 or len(pred_boxes) == 0:
            ious.append(0.0)
            recalls.append(0.0)
        else:
            best_iou = 0.0
            matched_gt = set()
            for p in pred_boxes:
                for j, g in enumerate(gt_boxes):
                    v = box_iou(p, g)
                    if v > best_iou: best_iou = v
                    if v >= 0.5: matched_gt.add(j)
            ious.append(best_iou)
            recalls.append(len(matched_gt) / len(gt_boxes))
    return torch.tensor(ious, dtype=torch.float32), torch.tensor(recalls, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="L-TSRL TransUNet Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Epochs to train student normally before RL")
    parser.add_argument("--bc_epochs", type=int, default=5, help="Epochs to train Tutor using Behavioral Cloning")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--resume", type=str, default="", help="Path to a .pth checkpoint to resume training from")
    parser.add_argument("--alpha", type=float, default=1.0, help="Reward weight for Dice")
    parser.add_argument("--beta", type=float, default=1.0, help="Reward weight for IoU")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Data Prep
    img_dir = os.path.join(args.data_dir, "images")
    den_dir = os.path.join(args.data_dir, "density")
    msk_dir = os.path.join(args.data_dir, "masks")
    lbl_dir = os.path.join(args.data_dir, "labels")

    train_ids, val_ids, test_ids = build_or_load_fixed_splits(args.data_dir, img_dir, args.save_dir)
    
    # Original Datasets
    train_ds_orig = ManipDensityDataset(train_ids, img_dir, den_dir, msk_dir, lbl_dir, augment=True)
    val_ds   = ManipDensityDataset(val_ids, img_dir, den_dir, msk_dir, lbl_dir, augment=False)
    
    # Wrapped Dataset for RL
    train_ds = DatasetWrapper(train_ds_orig)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_keep_boxes_with_index)
    
    # 2. Models
    student_model = TransUNetDensity().to(device)
    
    if args.resume and os.path.exists(args.resume):
        print(f"Loading pretrained TransUNet weights from: {args.resume}")
        student_model.load_state_dict(torch.load(args.resume, map_location=device))
    elif args.resume:
        print(f"Warning: --resume path '{args.resume}' not found. Starting from scratch.")
        
    student_opt = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # State Dim: 128 (GAP features) + 1 (Dice) + 1 (IoU) + 1 (LocLoss) + 1 (DenLoss) + 1 (Confidence) = 133
    state_dim = 133
    tutor_model = TutorPPO(state_dim=state_dim, action_dim=1).to(device)
    state_manager = StateManager(num_samples=len(train_ds_orig), device=device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = 0.0

    print("Starting L-TSRL Training...")
    
    for epoch in range(1, args.epochs + 1):
        student_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        is_warmup_stage = epoch <= args.warmup_epochs
        is_bc_stage = args.warmup_epochs < epoch <= (args.warmup_epochs + args.bc_epochs)
        is_rl_stage = epoch > (args.warmup_epochs + args.bc_epochs)
        
        for batch in pbar:
            imgs = batch["img"].to(device, non_blocking=True)
            gt_den = batch["den"].to(device, non_blocking=True)
            gt_msk = batch["msk"].to(device, non_blocking=True)
            gt_boxes_list = batch["boxes"]
            indices = batch["indices"]

            # --- Forward 1 (Before Update) ---
            with torch.no_grad():
                student_model.eval()
                init_pred_den, init_pred_msk_logit, init_feat = student_model(imgs)
                
                # Compute metrics
                init_dice = calculate_dice(init_pred_msk_logit, gt_msk)
                init_iou, init_recall = calculate_batch_iou_and_recall(init_pred_den, gt_boxes_list)
                init_iou = init_iou.to(device)
                init_recall = init_recall.to(device)
                
                # Confidence = Max density value in the predicted density map
                init_conf = init_pred_den.view(imgs.shape[0], -1).max(dim=1)[0]
                
                # Per-sample loss (unreduced)
                loss_den_sample = F.smooth_l1_loss(init_pred_den, gt_den, reduction='none').mean(dim=(1,2,3))
                loss_msk_sample = F.binary_cross_entropy_with_logits(init_pred_msk_logit, gt_msk, reduction='none').mean(dim=(1,2,3))
                loss_tv_sample = (init_pred_den[:,:,1:,:]-init_pred_den[:,:,:-1,:]).abs().mean(dim=(1,2,3)) + \
                                 (init_pred_den[:,:,:,1:]-init_pred_den[:,:,:,:-1]).abs().mean(dim=(1,2,3))
                init_loc_loss_sample = loss_den_sample + 1.0 * loss_msk_sample + 0.05 * loss_tv_sample

                # Extract GAP features for state: [B, 128, 32, 32] -> [B, 128]
                gap_feat = F.adaptive_avg_pool2d(init_feat, (1, 1)).flatten(1)

                # Compose State
                historical_states = state_manager.get_states(indices)
                state_s = torch.cat([
                    gap_feat,
                    historical_states['dice'].unsqueeze(1),
                    historical_states['iou'].unsqueeze(1),
                    historical_states['loc_loss'].unsqueeze(1),
                    historical_states['den_loss'].unsqueeze(1),
                    historical_states['confidence'].unsqueeze(1)
                ], dim=1)

            # --- Tutor Action / BC Training ---
            bc_loss_val = None
            if is_rl_stage:
                weights_np, _ = tutor_model.select_action(state_s)
                weights = torch.from_numpy(weights_np).to(device).float()
            else:
                # Warmup and BC: w = 1.0
                weights = torch.ones(imgs.shape[0], device=device).float()

            if is_bc_stage:
                # Calculate expert weights
                difficulty = 0.4 * (1.0 - historical_states['iou']) + \
                             0.4 * (1.0 - historical_states['dice']) + \
                             0.2 * historical_states['loc_loss']
                expert_weights = torch.clamp(difficulty, 0.0, 1.0).to(device)
                
                # Update Tutor Actor via BC
                bc_loss_val = tutor_model.bc_update(state_s, expert_weights)

            # --- Student Update ---
            student_model.train()
            student_opt.zero_grad()
            
            pred_den, pred_msk_logit, _ = student_model(imgs)
            
            # Compute localization loss with standard train_utils components, but per sample
            # (In practice we use standard reduction then weight, or weight per sample)
            l_den = F.smooth_l1_loss(pred_den, gt_den, reduction='none').mean(dim=(1,2,3))
            l_msk = F.binary_cross_entropy_with_logits(pred_msk_logit, gt_msk, reduction='none').mean(dim=(1,2,3))
            
            # Simplified TV Loss per sample
            l_tv = (pred_den[:,:,1:,:]-pred_den[:,:,:-1,:]).abs().mean(dim=(1,2,3)) + \
                   (pred_den[:,:,:,1:]-pred_den[:,:,:,:-1]).abs().mean(dim=(1,2,3))
            
            # Total localization loss per sample
            loc_loss_sample = l_den + 1.0 * l_msk + 0.05 * l_tv
            
            # Weighted loss
            weighted_loss = (loc_loss_sample * weights.squeeze()).mean()
            weighted_loss.backward()
            student_opt.step()

            # --- Forward 2 & Tutor Update (RL Stage) ---
            if is_rl_stage:
                with torch.no_grad():
                    student_model.eval()
                    new_pred_den, new_pred_msk_logit, _ = student_model(imgs)
                    
                    new_dice = calculate_dice(new_pred_msk_logit, gt_msk)
                    new_iou, new_recall = calculate_batch_iou_and_recall(new_pred_den, gt_boxes_list)
                    new_iou = new_iou.to(device)
                    new_recall = new_recall.to(device)
                    
                    # Compute Reward
                    delta_dice = new_dice - init_dice
                    delta_iou = new_iou - init_iou
                    delta_recall = new_recall - init_recall
                    rewards = 0.5 * delta_iou + 0.3 * delta_dice + 0.2 * delta_recall
                    
                    tutor_model.buffer.rewards.extend(rewards.cpu().numpy())
                    tutor_model.buffer.is_terminals.extend([True] * len(rewards))

            # --- Update State Manager ---
            state_manager.update_states(
                indices, 
                current_dice=init_dice, 
                current_iou=init_iou, 
                current_loc_loss=init_loc_loss_sample,
                current_den_loss=loss_den_sample, 
                current_confidence=init_conf
            )
            
            if is_bc_stage and bc_loss_val is not None:
                pbar.set_postfix(w_loss=weighted_loss.item(), bc_loss=bc_loss_val)
            else:
                pbar.set_postfix(w_loss=weighted_loss.item())

        # End of Epoch
        if is_rl_stage:
            print(f"Updating Tutor PPO Policy...")
            tutor_model.update()

        # Validate
        from dataset import collate_keep_boxes
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_keep_boxes)
        val_metrics = eval_loader(student_model, val_loader, device, thr=0.28, min_area=20)
        f1 = val_metrics['F1@0.5']
        print(f"\nEpoch {epoch} Validation: F1@0.5: {f1:.4f}")
        
        # Save Best Model
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(args.save_dir, 'ltsrl_transunet_best.pth')
            torch.save(student_model.state_dict(), save_path)
            tutor_model.save(os.path.join(args.save_dir, 'ltsrl_tutor_best.pth'))
            print(f"--> Saved new best model to {save_path} (F1: {best_f1:.4f})")

    print("L-TSRL Training complete.")

if __name__ == "__main__":
    main()

import torch
import os

class StateManager:
    """
    Manages the dynamic state of each sample throughout the training process
    for the Localization-Aware Tutor-Student Reinforcement Learning (L-TSRL).
    
    Tracks:
    - Dice Score (EMA)
    - IoU (EMA)
    - Localization Loss (EMA)
    - Density Loss (EMA)
    - Detection Probability (Confidence)
    """

    def __init__(self, num_samples, alpha=0.1, device='cpu'):
        self.num_samples = num_samples
        self.alpha = alpha
        self.device = device
        
        # Initialize EMA values. 
        # Losses start relatively high, metrics start low
        self.ema_dice = torch.zeros((num_samples,), device=device)
        self.ema_iou = torch.zeros((num_samples,), device=device)
        self.ema_loc_loss = torch.full((num_samples,), 1.0, device=device)
        self.ema_den_loss = torch.full((num_samples,), 1.0, device=device)
        self.ema_confidence = torch.zeros((num_samples,), device=device)

        print(f"L-TSRL StateManager initialized for {num_samples} samples on device {device}.")

    def get_states(self, indices):
        """
        Retrieve the current states for a batch of indices.
        """
        indices = indices.to(self.device)
        return {
            'dice': self.ema_dice[indices],
            'iou': self.ema_iou[indices],
            'loc_loss': self.ema_loc_loss[indices],
            'den_loss': self.ema_den_loss[indices],
            'confidence': self.ema_confidence[indices]
        }

    def update_states(self, indices, current_dice, current_iou, current_loc_loss, current_den_loss, current_confidence):
        """
        Update the EMA states for a batch of indices based on current metrics.
        """
        indices = indices.to(self.device)
        
        # Ensure tensors are detached and on correct device
        current_dice = current_dice.detach().to(self.device)
        current_iou = current_iou.detach().to(self.device)
        current_loc_loss = current_loc_loss.detach().to(self.device)
        current_den_loss = current_den_loss.detach().to(self.device)
        current_confidence = current_confidence.detach().to(self.device)
        
        # Update EMA: new_ema = alpha * current + (1 - alpha) * old_ema
        self.ema_dice[indices] = self.alpha * current_dice + (1 - self.alpha) * self.ema_dice[indices]
        self.ema_iou[indices] = self.alpha * current_iou + (1 - self.alpha) * self.ema_iou[indices]
        self.ema_loc_loss[indices] = self.alpha * current_loc_loss + (1 - self.alpha) * self.ema_loc_loss[indices]
        self.ema_den_loss[indices] = self.alpha * current_den_loss + (1 - self.alpha) * self.ema_den_loss[indices]
        self.ema_confidence[indices] = self.alpha * current_confidence + (1 - self.alpha) * self.ema_confidence[indices]

    def save_states(self, checkpoint_dir, epoch):
        state_path = os.path.join(checkpoint_dir, f'state_manager_epoch_{epoch}.pth')
        torch.save({
            'ema_dice': self.ema_dice,
            'ema_iou': self.ema_iou,
            'ema_loc_loss': self.ema_loc_loss,
            'ema_den_loss': self.ema_den_loss,
            'ema_confidence': self.ema_confidence
        }, state_path)
        print(f"StateManager states saved to {state_path}")

    def load_states(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Warning: StateManager checkpoint not found at {checkpoint_path}. Starting with fresh states.")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.ema_dice = checkpoint['ema_dice'].to(self.device)
        self.ema_iou = checkpoint['ema_iou'].to(self.device)
        self.ema_loc_loss = checkpoint['ema_loc_loss'].to(self.device)
        self.ema_den_loss = checkpoint['ema_den_loss'].to(self.device)
        self.ema_confidence = checkpoint['ema_confidence'].to(self.device)
        print(f"StateManager states loaded from {checkpoint_path}")

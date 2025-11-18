"""
ControlNet Trainer

Implements training loop for ControlNet adapter with all loss components.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional
from tqdm import tqdm
import wandb

from ..models.stable_diffusion_wrapper import StableDiffusionControlNetWrapper
from .losses import TotalLoss, SyntheticEnhancementLoss
from .dataset import VideoAdDataset


class ControlNetTrainer:
    """Trainer for ControlNet adapter."""

    def __init__(
        self,
        model: StableDiffusionControlNetWrapper,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cuda",
        output_dir: str = "./outputs",
        lambda_recon: float = 1.0,
        lambda_lpips: float = 1.0,
        lambda_bg: float = 0.5,
        use_recon_loss: bool = False,  # Set True for full reconstruction training
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        log_wandb: bool = False,
        project_name: str = "attention-keyword-alignment",
    ):
        """
        Initialize trainer.

        Args:
            model: SD+ControlNet wrapper
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            device: Device to train on
            output_dir: Directory to save checkpoints
            lambda_recon: Weight for reconstruction loss
            lambda_lpips: Weight for LPIPS loss
            lambda_bg: Weight for background preservation
            use_recon_loss: Whether to use full reconstruction loss
            gradient_accumulation_steps: Steps to accumulate gradients
            mixed_precision: Whether to use mixed precision training
            log_wandb: Whether to log to Weights & Biases
            project_name: W&B project name
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = output_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.log_wandb = log_wandb

        os.makedirs(output_dir, exist_ok=True)

        # Loss function
        self.criterion = TotalLoss(
            lambda_recon=lambda_recon,
            lambda_lpips=lambda_lpips,
            lambda_bg=lambda_bg,
            device=device,
            use_recon_loss=use_recon_loss,
        ).to(device)

        # Optimizer
        trainable_params = model.get_trainable_parameters()
        self.optimizer = AdamW(trainable_params, lr=learning_rate)

        # Scheduler
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Initialize W&B
        if log_wandb:
            wandb.init(project=project_name, config={
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'lambda_recon': lambda_recon,
                'lambda_lpips': lambda_lpips,
                'lambda_bg': lambda_bg,
            })

        self.global_step = 0
        self.epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'diffusion': 0.0,
            'recon_total': 0.0,
            'background': 0.0,
        }
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            image = batch['image'].to(self.device)
            control_tensor = batch['control'].to(self.device)  # Fixed: 'control' not 'control_tensor'
            background_mask = batch['background_mask'].to(self.device)
            keyword_text = batch['keyword'][0]  # Fixed: 'keyword' not 'keyword_text'

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                # Training step
                outputs = self.model.training_step(
                    image=image,
                    control_tensor=control_tensor,
                    text_prompt=keyword_text,
                )

                noise_pred = outputs['noise_pred']
                noise = outputs['noise']

                # Compute loss
                losses = self.criterion(
                    noise_pred=noise_pred,
                    noise_target=noise,
                    pred_image=None,  # Not computing full reconstruction by default
                    target_image=image,
                    background_mask=background_mask,
                )

                loss = losses['total'] / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()

            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'lr': self.scheduler.get_last_lr()[0],
            })

            # Log to W&B
            if self.log_wandb and self.global_step % 10 == 0:
                log_dict = {f'train/{k}': v.item() for k, v in losses.items()}
                log_dict['train/learning_rate'] = self.scheduler.get_last_lr()[0]
                wandb.log(log_dict, step=self.global_step)

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation pass."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()

        val_losses = {
            'total': 0.0,
            'diffusion': 0.0,
            'recon_total': 0.0,
            'background': 0.0,
        }
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            # Move to device
            image = batch['image'].to(self.device)
            control_tensor = batch['control'].to(self.device)  # Fixed: 'control' not 'control_tensor'
            background_mask = batch['background_mask'].to(self.device)
            keyword_text = batch['keyword'][0]  # Fixed: 'keyword' not 'keyword_text'

            # Forward pass
            outputs = self.model.training_step(
                image=image,
                control_tensor=control_tensor,
                text_prompt=keyword_text,
            )

            # Compute loss
            losses = self.criterion(
                noise_pred=outputs['noise_pred'],
                noise_target=outputs['noise'],
                pred_image=None,
                target_image=image,
                background_mask=background_mask,
            )

            # Accumulate losses
            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()

            num_batches += 1

        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Output directory: {self.output_dir}")

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Print results
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"Train Loss: {train_losses['total']:.4f}")
            if val_losses:
                print(f"Val Loss: {val_losses['total']:.4f}")

            # Log to W&B
            if self.log_wandb:
                log_dict = {f'val/{k}': v for k, v in val_losses.items()}
                log_dict['epoch'] = epoch
                wandb.log(log_dict, step=self.global_step)

            # Save checkpoint
            if val_losses and val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_checkpoint(os.path.join(self.output_dir, 'best_model.pt'))
                print("Saved best model!")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    os.path.join(self.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                )

        print("\nTraining complete!")
        if self.log_wandb:
            wandb.finish()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.controlnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.controlnet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from epoch {self.epoch}")

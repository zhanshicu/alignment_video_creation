"""
Example: Training ControlNet Adapter

This script demonstrates how to train the ControlNet adapter
on video advertisement data.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import StableDiffusionControlNetWrapper
from src.training import ControlNetTrainer, VideoAdDataModule
from src.models.controlnet_adapter import ControlNetAdapter


def main():
    # Load configuration
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("ControlNet Training")
    print("=" * 50)

    # Setup
    device = "cuda"
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Initialize model
    print("\n[1/3] Initializing SD+ControlNet model...")

    controlnet_config = config['model']['controlnet']
    model = StableDiffusionControlNetWrapper(
        sd_model_name=config['model']['sd_model_name'],
        controlnet_config=controlnet_config,
        device=device,
        use_lora=config['model']['use_lora'],
    )

    print("Model initialized successfully")
    print(f"  - SD Model: {config['model']['sd_model_name']}")
    print(f"  - ControlNet channels: {controlnet_config['control_channels']}")
    print(f"  - Using LoRA: {config['model']['use_lora']}")

    # Step 2: Setup data
    print("\n[2/3] Setting up data loaders...")

    # Define train/val splits
    train_videos = ["video1", "video2", "video3"]  # Replace with your video IDs
    val_videos = ["video4"]

    data_module = VideoAdDataModule(
        data_root=config['data']['data_root'],
        keywords_file=config['data']['keywords_file'],
        train_videos=train_videos,
        val_videos=val_videos,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        image_size=config['data']['image_size'],
        include_raw_maps=config['data']['include_raw_maps'],
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")

    # Step 3: Train
    print("\n[3/3] Starting training...")

    trainer = ControlNetTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['num_epochs'],
        device=device,
        output_dir=output_dir,
        lambda_recon=config['training']['lambda_recon'],
        lambda_lpips=config['training']['lambda_lpips'],
        lambda_bg=config['training']['lambda_bg'],
        use_recon_loss=config['training']['use_recon_loss'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'],
        log_wandb=config['training']['log_wandb'],
        project_name=config['training']['project_name'],
    )

    # Start training
    trainer.train()

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model.pt')}")
    print("=" * 50)


if __name__ == "__main__":
    main()

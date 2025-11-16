#!/usr/bin/env python3
"""
Example script showing proper dataset preparation.

This demonstrates:
1. Getting valid video IDs from alignment_score.csv (not keywords.csv!)
2. Splitting into train/val sets
3. Creating data loaders
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import (
    get_valid_video_ids,
    split_train_val_videos,
    print_dataset_statistics,
    VideoSceneDataModule,
)


def main():
    # Configuration
    alignment_score_file = 'data/alignment_score.csv'
    keywords_file = 'data/keywords.csv'
    screenshots_dir = 'data/screenshots_tiktok'
    keyword_masks_dir = 'data/keyword_masks'

    # Step 1: Get valid video IDs
    # IMPORTANT: This uses alignment_score.csv as the source of truth
    print("Step 1: Getting valid video IDs...")
    print("="*60)
    valid_video_ids = get_valid_video_ids(
        alignment_score_file=alignment_score_file,
        keywords_file=keywords_file
    )
    print(f"\nFound {len(valid_video_ids)} valid videos")
    print(f"First 5 video IDs: {valid_video_ids[:5]}")

    # Step 2: Split into train/val
    print("\n" + "="*60)
    print("Step 2: Splitting into train/validation sets...")
    print("="*60)
    train_videos, val_videos = split_train_val_videos(
        video_ids=valid_video_ids,
        val_ratio=0.2,
        random_seed=42
    )

    # Step 3: Print statistics
    print_dataset_statistics(
        alignment_score_file=alignment_score_file,
        train_videos=train_videos,
        val_videos=val_videos
    )

    # Step 4: Create data module
    print("Step 4: Creating data module...")
    print("="*60)
    data_module = VideoSceneDataModule(
        alignment_score_file=alignment_score_file,
        keywords_file=keywords_file,
        train_videos=train_videos,
        val_videos=val_videos,
        screenshots_dir=screenshots_dir,
        keyword_masks_dir=keyword_masks_dir,
        batch_size=4,
        num_workers=4,
        image_size=(512, 512),
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"\n✓ Data module created successfully")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Step 5: Test loading a batch
    print("\n" + "="*60)
    print("Step 5: Testing batch loading...")
    print("="*60)
    batch = next(iter(train_loader))

    print(f"\nBatch contents:")
    print(f"  image: {batch['image'].shape}")
    print(f"  control: {batch['control'].shape}")
    print(f"  keyword_mask: {batch['keyword_mask'].shape}")
    print(f"  alignment_score: {batch['alignment_score'][:3].tolist()}")
    print(f"  keywords: {batch['keyword'][:2]}")
    print(f"  video_ids: {batch['video_id'][:2]}")
    print(f"  scene_numbers: {batch['scene_number'][:3].tolist()}")

    print("\n✓ Dataset preparation complete!")


if __name__ == '__main__':
    main()

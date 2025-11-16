"""
Updated Dataset for Video Advertisement Training

Uses alignment_score.csv instead of raw attention heatmaps.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Callable


class VideoSceneDataset(Dataset):
    """
    Dataset for video advertisement scenes with alignment scores.

    Uses pre-computed alignment scores instead of raw attention heatmaps.
    """

    def __init__(
        self,
        alignment_score_file: str,
        keywords_file: str,
        screenshots_dir: str = "data/screenshots_tiktok",
        keyword_masks_dir: str = "data/keyword_masks",
        video_ids: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            alignment_score_file: Path to alignment_score.csv
            keywords_file: Path to keywords.csv
            screenshots_dir: Directory containing scene screenshots
            keyword_masks_dir: Directory containing keyword masks (from CLIPSeg)
            video_ids: Optional list of video IDs to include (None = all videos)
            image_size: Target image size (H, W)
            transform: Optional transform to apply
        """
        self.screenshots_dir = screenshots_dir
        self.keyword_masks_dir = keyword_masks_dir
        self.image_size = image_size
        self.transform = transform

        # Load alignment scores
        self.alignment_df = pd.read_csv(alignment_score_file)
        self.alignment_df.columns = self.alignment_df.columns.str.strip()

        # Load keywords
        self.keywords_df = pd.read_csv(keywords_file)
        self.keywords_df.columns = self.keywords_df.columns.str.strip()

        # Create keyword mapping
        if '_id' in self.keywords_df.columns:
            video_id_col = '_id'
        elif 'video_id' in self.keywords_df.columns:
            video_id_col = 'video_id'
        else:
            raise ValueError(f"Could not find video ID column in keywords.csv")

        if 'keyword_list[0]' in self.keywords_df.columns:
            keyword_col = 'keyword_list[0]'
        elif 'keyword' in self.keywords_df.columns:
            keyword_col = 'keyword'
        else:
            raise ValueError(f"Could not find keyword column in keywords.csv")

        self.keywords = dict(zip(
            self.keywords_df[video_id_col].astype(str),
            self.keywords_df[keyword_col]
        ))

        # Filter by video_ids if provided
        if video_ids is not None:
            video_ids_str = [str(vid) for vid in video_ids]
            self.alignment_df = self.alignment_df[
                self.alignment_df['video id'].astype(str).isin(video_ids_str)
            ]

        # Filter out scenes without keywords
        self.alignment_df['video_id_str'] = self.alignment_df['video id'].astype(str)
        self.alignment_df['has_keyword'] = self.alignment_df['video_id_str'].isin(self.keywords.keys())
        self.alignment_df = self.alignment_df[self.alignment_df['has_keyword']]

        # Reset index
        self.alignment_df = self.alignment_df.reset_index(drop=True)

        print(f"Dataset initialized with {len(self.alignment_df)} scenes from {self.alignment_df['video id'].nunique()} videos")

    def __len__(self) -> int:
        return len(self.alignment_df)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image and resize."""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask and resize."""
        mask = Image.open(path).convert('L')
        mask = mask.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        mask = np.array(mask).astype(np.float32) / 255.0
        return mask

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dict containing:
                - 'image': Frame tensor (3, H, W) in [-1, 1]
                - 'control': Control tensor (2, H, W)
                - 'keyword_mask': Binary keyword mask (1, H, W)
                - 'alignment_score': Scalar alignment score
                - 'keyword': Text description
                - 'video_id': Video ID
                - 'scene_number': Scene number
        """
        row = self.alignment_df.iloc[idx]

        video_id = str(row['video id'])
        scene_number = int(row['Scene Number'])
        alignment_score = float(row['attention_proportion'])
        keyword = self.keywords.get(video_id, "product")

        # Construct file paths
        # Assuming screenshots are named like: {video_id}/scene_{scene_number}.png
        screenshot_path = os.path.join(
            self.screenshots_dir,
            video_id,
            f"scene_{scene_number}.png"
        )

        # Alternative naming: scene_{scene_number:02d}.png
        if not os.path.exists(screenshot_path):
            screenshot_path = os.path.join(
                self.screenshots_dir,
                video_id,
                f"scene_{scene_number:02d}.png"
            )

        # Load screenshot
        if not os.path.exists(screenshot_path):
            # If screenshot doesn't exist, create a dummy black image
            # (This is for testing when data is not available)
            image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        else:
            image = self._load_image(screenshot_path)

        # Convert to [-1, 1]
        image = (image * 2.0) - 1.0

        # Load keyword mask
        mask_path = os.path.join(
            self.keyword_masks_dir,
            video_id,
            f"scene_{scene_number}.png"
        )

        # Alternative naming
        if not os.path.exists(mask_path):
            mask_path = os.path.join(
                self.keyword_masks_dir,
                video_id,
                f"scene_{scene_number:02d}.png"
            )

        if not os.path.exists(mask_path):
            # If mask doesn't exist, create a dummy mask
            # (This will be generated by the preprocessing script)
            keyword_mask = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float32)
        else:
            keyword_mask = self._load_mask(mask_path)
            # Binarize
            keyword_mask = (keyword_mask > 0.5).astype(np.float32)

        # Create background mask
        background_mask = 1.0 - keyword_mask

        # Build control tensor
        # Control channel 0: keyword mask (M_t)
        # Control channel 1: alignment map (S_t = M_t * alignment_score)
        alignment_map = keyword_mask * alignment_score

        control_tensor = np.stack([
            keyword_mask,
            alignment_map
        ], axis=0)

        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        control_tensor = torch.from_numpy(control_tensor).float()
        keyword_mask = torch.from_numpy(keyword_mask).unsqueeze(0).float()
        background_mask = torch.from_numpy(background_mask).unsqueeze(0).float()

        # Apply optional transform
        if self.transform is not None:
            image = self.transform(image)

        return {
            'image': image,
            'control': control_tensor,
            'keyword_mask': keyword_mask,
            'background_mask': background_mask,
            'alignment_score': alignment_score,
            'keyword': keyword,
            'video_id': video_id,
            'scene_number': scene_number,
        }


class VideoSceneDataModule:
    """Data module for organizing train/val splits."""

    def __init__(
        self,
        alignment_score_file: str,
        keywords_file: str,
        train_videos: List[str],
        val_videos: List[str],
        screenshots_dir: str = "data/screenshots_tiktok",
        keyword_masks_dir: str = "data/keyword_masks",
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (512, 512),
    ):
        """
        Initialize data module.

        Args:
            alignment_score_file: Path to alignment_score.csv
            keywords_file: Path to keywords.csv
            train_videos: List of video IDs for training
            val_videos: List of video IDs for validation
            screenshots_dir: Directory containing screenshots
            keyword_masks_dir: Directory containing keyword masks
            batch_size: Batch size
            num_workers: Number of data loading workers
            image_size: Target image size (H, W)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create datasets
        print("Creating training dataset...")
        self.train_dataset = VideoSceneDataset(
            alignment_score_file=alignment_score_file,
            keywords_file=keywords_file,
            screenshots_dir=screenshots_dir,
            keyword_masks_dir=keyword_masks_dir,
            video_ids=train_videos,
            image_size=image_size,
        )

        print("Creating validation dataset...")
        self.val_dataset = VideoSceneDataset(
            alignment_score_file=alignment_score_file,
            keywords_file=keywords_file,
            screenshots_dir=screenshots_dir,
            keyword_masks_dir=keyword_masks_dir,
            video_ids=val_videos,
            image_size=image_size,
        )

    def train_dataloader(self):
        """Get training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

"""
Dataset for Video Advertisement Training

Loads video frames with attention and keyword heatmaps.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Callable
import json


class VideoAdDataset(Dataset):
    """
    Dataset for video advertisement frames with attention and keyword heatmaps.
    """

    def __init__(
        self,
        data_root: str,
        video_ids: List[str],
        keywords: Dict[str, str],
        attention_dir: str = "attention_heatmaps",
        keyword_dir: str = "keyword_heatmaps",
        frames_dir: str = "frames",
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
        include_raw_maps: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            data_root: Root directory containing all data
            video_ids: List of video IDs to include
            keywords: Dict mapping video_id to keyword/product description
            attention_dir: Subdirectory containing attention heatmaps
            keyword_dir: Subdirectory containing keyword heatmaps
            frames_dir: Subdirectory containing video frames
            image_size: Target image size (H, W)
            transform: Optional transform to apply
            include_raw_maps: Whether to include raw A_t and K_t in control tensor
        """
        self.data_root = data_root
        self.video_ids = video_ids
        self.keywords = keywords
        self.attention_dir = attention_dir
        self.keyword_dir = keyword_dir
        self.frames_dir = frames_dir
        self.image_size = image_size
        self.transform = transform
        self.include_raw_maps = include_raw_maps

        # Build file list
        self.samples = self._build_sample_list()

    def _build_sample_list(self) -> List[Dict]:
        """Build list of all valid samples."""
        samples = []

        for video_id in self.video_ids:
            # Get all frames for this video
            frame_dir = os.path.join(self.data_root, self.frames_dir, video_id)
            if not os.path.exists(frame_dir):
                continue

            frame_files = sorted([
                f for f in os.listdir(frame_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])

            for frame_file in frame_files:
                frame_name = os.path.splitext(frame_file)[0]

                # Check for corresponding heatmaps
                attention_path = os.path.join(
                    self.data_root, self.attention_dir, video_id, f"{frame_name}.png"
                )
                keyword_path = os.path.join(
                    self.data_root, self.keyword_dir, video_id, f"{frame_name}.png"
                )

                if os.path.exists(attention_path) and os.path.exists(keyword_path):
                    samples.append({
                        'video_id': video_id,
                        'frame_name': frame_name,
                        'frame_path': os.path.join(frame_dir, frame_file),
                        'attention_path': attention_path,
                        'keyword_path': keyword_path,
                        'keyword_text': self.keywords.get(video_id, "product"),
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image and resize."""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        return img

    def _load_heatmap(self, path: str) -> np.ndarray:
        """Load heatmap and resize."""
        heatmap = Image.open(path).convert('L')
        heatmap = heatmap.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        heatmap = np.array(heatmap).astype(np.float32) / 255.0
        return heatmap

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict containing:
                - 'image': Frame tensor (3, H, W) in [-1, 1]
                - 'control_tensor': Control tensor (C, H, W) where C=2 or 4
                - 'attention_map': Attention heatmap (H, W)
                - 'keyword_map': Keyword heatmap (H, W)
                - 'keyword_mask': Binary/soft keyword mask (H, W)
                - 'background_mask': Background mask (H, W)
                - 'keyword_text': Text description
        """
        sample_info = self.samples[idx]

        # Load image
        image = self._load_image(sample_info['frame_path'])
        # Convert to [-1, 1]
        image = (image * 2.0) - 1.0

        # Load heatmaps
        attention_map = self._load_heatmap(sample_info['attention_path'])
        keyword_map = self._load_heatmap(sample_info['keyword_path'])

        # Create keyword mask (binarize keyword_map)
        keyword_mask = (keyword_map > 0.5).astype(np.float32)

        # Create background mask
        background_mask = 1.0 - keyword_mask

        # Build control tensor
        # Compute alignment map
        alignment_map = attention_map * keyword_map
        # Normalize
        if alignment_map.max() > 0:
            alignment_map = alignment_map / (alignment_map.max() + 1e-8)

        if self.include_raw_maps:
            control_tensor = np.stack([
                keyword_mask,
                alignment_map,
                attention_map,
                keyword_map
            ], axis=0)
        else:
            control_tensor = np.stack([
                keyword_mask,
                alignment_map
            ], axis=0)

        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        control_tensor = torch.from_numpy(control_tensor).float()
        attention_map = torch.from_numpy(attention_map).float()
        keyword_map = torch.from_numpy(keyword_map).float()
        keyword_mask = torch.from_numpy(keyword_mask).unsqueeze(0).float()
        background_mask = torch.from_numpy(background_mask).unsqueeze(0).float()

        # Apply optional transform
        if self.transform is not None:
            image = self.transform(image)

        return {
            'image': image,
            'control': control_tensor,  # Changed from 'control_tensor' to match notebook
            'control_tensor': control_tensor,  # Keep for backward compatibility
            'attention_map': attention_map,
            'keyword_map': keyword_map,
            'keyword_mask': keyword_mask,
            'background_mask': background_mask,
            'keyword': sample_info['keyword_text'],  # Changed from 'keyword_text' to match notebook
            'keyword_text': sample_info['keyword_text'],  # Keep for backward compatibility
            'video_id': sample_info['video_id'],
            'frame_name': sample_info['frame_name'],
        }


class VideoAdDataModule:
    """Data module for organizing train/val splits."""

    def __init__(
        self,
        data_root: str,
        keywords_file: str,
        train_videos: List[str],
        val_videos: List[str],
        batch_size: int = 4,
        num_workers: int = 4,
        **dataset_kwargs
    ):
        """
        Initialize data module.

        Args:
            data_root: Root data directory
            keywords_file: Path to JSON file with video_id -> keyword mapping
            train_videos: List of video IDs for training
            val_videos: List of video IDs for validation
            batch_size: Batch size
            num_workers: Number of data loading workers
            **dataset_kwargs: Additional arguments for VideoAdDataset
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load keywords
        with open(keywords_file, 'r') as f:
            self.keywords = json.load(f)

        # Create datasets
        self.train_dataset = VideoAdDataset(
            data_root=data_root,
            video_ids=train_videos,
            keywords=self.keywords,
            **dataset_kwargs
        )

        self.val_dataset = VideoAdDataset(
            data_root=data_root,
            video_ids=val_videos,
            keywords=self.keywords,
            **dataset_kwargs
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

"""
Attention/Meaning Heatmap Generator

Implements the 4-stage pipeline:
1. Circular patch extraction
2. LLaVA semantic scoring
3. Gaussian smoothing + gamma correction
"""

import os
import csv
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm


class AttentionHeatmapGenerator:
    """Generates attention heatmaps from video frames using semantic richness."""

    def __init__(
        self,
        patch_degrees: List[int] = [3, 7],
        overlap: float = 0.1,
        fov: float = 90.0,
        sigma: float = 5.0,
        gamma: float = 3.0,
    ):
        """
        Initialize the attention heatmap generator.

        Args:
            patch_degrees: List of patch sizes in degrees of visual angle
            overlap: Overlap ratio between adjacent patches (0-1)
            fov: Field of view in degrees
            sigma: Gaussian smoothing parameter
            gamma: Gamma correction exponent for enhancing contrast
        """
        self.patch_degrees = patch_degrees
        self.overlap = overlap
        self.fov = fov
        self.sigma = sigma
        self.gamma = gamma

    def create_circular_mask(
        self,
        h: int,
        w: int,
        center: Tuple[int, int],
        radius: int
    ) -> np.ndarray:
        """Create a circular mask."""
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
        return mask

    def degree_to_pixel(self, degree: float, h: int, w: int) -> int:
        """Convert degree of visual angle to pixel radius."""
        return int((degree / self.fov) * min(h, w))

    def extract_circular_patches(
        self,
        image: np.ndarray,
        scene_name: str,
        save_dir_patch: str,
        save_dir_meta: str
    ) -> List[Dict]:
        """
        Extract circular patches from an image.

        Args:
            image: Input image as numpy array (H, W, 3)
            scene_name: Name identifier for the scene
            save_dir_patch: Directory to save patch images
            save_dir_meta: Directory to save metadata

        Returns:
            List of metadata dictionaries for each patch
        """
        os.makedirs(save_dir_patch, exist_ok=True)
        os.makedirs(save_dir_meta, exist_ok=True)

        h, w = image.shape[:2]
        patch_count = 0
        metadata = []

        for degree in self.patch_degrees:
            radius = self.degree_to_pixel(degree, h, w)
            step = int(radius * (1 - self.overlap))

            for y in range(0, h, step):
                for x in range(0, w, step):
                    mask = self.create_circular_mask(h, w, (x, y), radius)
                    patch = np.zeros_like(image)
                    patch[mask] = image[mask]

                    # Find bounding box
                    coords = np.argwhere(mask)
                    if len(coords) == 0:
                        continue
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)

                    # Crop and save patch
                    cropped_patch = patch[y_min:y_max+1, x_min:x_max+1]
                    patch_image = Image.fromarray(cropped_patch)
                    patch_filename = f'patch_{patch_count}_deg_{degree}.png'
                    patch_image.save(os.path.join(save_dir_patch, patch_filename))

                    # Store metadata
                    metadata.append({
                        'filename': patch_filename,
                        'center': (x, y),
                        'radius': radius,
                        'bbox': (x_min, y_min, x_max, y_max),
                        'degree': degree,
                    })

                    patch_count += 1

        # Save metadata
        np.save(os.path.join(save_dir_meta, f'{scene_name}_metadata.npy'), metadata)
        return metadata

    def save_patch_info_to_csv(
        self,
        metadata: List[Dict],
        scene_name: str,
        output_csv: str
    ):
        """Save patch metadata to CSV for LLaVA inference."""
        file_exists = os.path.isfile(output_csv)

        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    'scene', 'filename', 'center_x', 'center_y',
                    'radius', 'bbox_x_min', 'bbox_y_min',
                    'bbox_x_max', 'bbox_y_max', 'degree'
                ])

            for data in metadata:
                writer.writerow([
                    scene_name,
                    data['filename'],
                    data['center'][0], data['center'][1],
                    data['radius'],
                    data['bbox'][0], data['bbox'][1],
                    data['bbox'][2], data['bbox'][3],
                    data['degree']
                ])

    def likert_to_numeric(self, likert_label: str) -> int:
        """Convert Likert scale label to numeric score."""
        likert_scale = {
            'very low': 1, 'Very low': 1,
            'low': 2, 'Low': 2,
            'somewhat low': 3, 'Somewhat low': 3,
            'somewhat high': 4, 'Somewhat high': 4,
            'high': 5, 'High': 5,
            'very high': 6, 'Very high': 6
        }
        return likert_scale.get(likert_label, 3)  # Default to middle value

    def construct_heatmap_from_scores(
        self,
        scores_csv: str,
        original_shape: Tuple[int, int],
        scene_filter: Optional[str] = None
    ) -> np.ndarray:
        """
        Construct attention heatmap from patch scores.

        Args:
            scores_csv: Path to CSV with patch scores from LLaVA
            original_shape: (H, W) of the original image
            scene_filter: Optional scene name to filter

        Returns:
            Smoothed attention heatmap (H, W)
        """
        import pandas as pd

        # Load scores
        data = pd.read_csv(scores_csv)
        if scene_filter:
            data = data[data['scene'] == scene_filter]

        # Initialize maps
        meaning_map = np.zeros(original_shape[:2], dtype=np.float32)
        count_map = np.zeros(original_shape[:2], dtype=np.float32)

        # Accumulate scores
        for _, row in data.iterrows():
            # Get numeric score
            if 'likert_label_predicted' in row:
                score = self.likert_to_numeric(row['likert_label_predicted'])
            elif 'mean_score' in row:
                score = row['mean_score']
            else:
                continue

            # Get patch information
            x_min, y_min = int(row['bbox_x_min']), int(row['bbox_y_min'])
            x_max, y_max = int(row['bbox_x_max']), int(row['bbox_y_max'])
            center = (int(row['center_x']), int(row['center_y']))
            radius = int(row['radius'])

            # Create mask
            h, w = original_shape[:2]
            mask = self.create_circular_mask(h, w, center, radius)
            mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]

            # Accumulate
            meaning_map[y_min:y_max+1, x_min:x_max+1][mask_cropped] += score
            count_map[y_min:y_max+1, x_min:x_max+1][mask_cropped] += 1

        # Average
        count_map[count_map == 0] = 1
        heatmap = meaning_map / count_map

        # Smooth and apply gamma correction
        heatmap = gaussian_filter(heatmap, sigma=self.sigma)
        heatmap = np.power(heatmap, self.gamma)

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def process_video_frames(
        self,
        frames: List[np.ndarray],
        frame_names: List[str],
        output_dir: str,
        video_id: str
    ) -> Tuple[str, str]:
        """
        Process all frames of a video and extract patches.

        Args:
            frames: List of video frames as numpy arrays
            frame_names: List of frame identifiers
            output_dir: Base output directory
            video_id: Video identifier

        Returns:
            Tuple of (patch_dir, metadata_dir)
        """
        patch_dir = os.path.join(output_dir, 'patches', video_id)
        metadata_dir = os.path.join(output_dir, 'metadata', video_id)
        csv_path = os.path.join(output_dir, f'{video_id}_patch_info.csv')

        os.makedirs(patch_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        for frame, frame_name in tqdm(zip(frames, frame_names),
                                       total=len(frames),
                                       desc="Extracting patches"):
            scene_patch_dir = os.path.join(patch_dir, frame_name)
            metadata = self.extract_circular_patches(
                frame, frame_name, scene_patch_dir, metadata_dir
            )
            self.save_patch_info_to_csv(metadata, frame_name, csv_path)

        return patch_dir, metadata_dir, csv_path

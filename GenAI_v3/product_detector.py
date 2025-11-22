"""
Automatic Main Product Detection

Detects the main product/content in a video by analyzing what appears
consistently across frames (episodic memory approach).

No keyword required - automatically finds the most prominent recurring element.

Methods:
1. Sample frames across video
2. Segment all objects using SAM (Segment Anything Model)
3. Track which segments appear consistently (DINO features)
4. Main product = most frequent, prominent, consistent element
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional
from pathlib import Path


class MainProductDetector:
    """
    Automatically detect the main product/content in a video.

    Uses:
    - SAM (Segment Anything Model) for segmentation
    - DINO features for tracking consistency across frames
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize detector.

        Args:
            device: "cuda" or "cpu"
        """
        self.device = device
        print("Loading SAM and DINO models...")
        self._load_models()
        print("✓ Models loaded")

    def _load_models(self):
        """Load SAM and DINO models."""
        # Load SAM (Segment Anything Model)
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

            # Use SAM ViT-H (best quality) or ViT-B (faster)
            sam_checkpoint = self._download_sam_checkpoint()
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            sam.to(self.device)

            self.mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=1000,  # Filter tiny segments
            )
            print("  ✓ SAM loaded")

        except ImportError:
            print("  ⚠ SAM not available, using fallback saliency detection")
            self.mask_generator = None

        # Load DINO for feature extraction (tracking consistency)
        try:
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.dino.to(self.device)
            self.dino.eval()
            print("  ✓ DINOv2 loaded")

        except Exception as e:
            print(f"  ⚠ DINOv2 not available: {e}")
            self.dino = None

    def _download_sam_checkpoint(self) -> str:
        """Download SAM checkpoint if not exists."""
        import os
        import urllib.request

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / "sam_vit_h_4b8939.pth"

        if not checkpoint_path.exists():
            print("  Downloading SAM checkpoint (~2.5GB)...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            urllib.request.urlretrieve(url, checkpoint_path)
            print("  ✓ Downloaded")

        return str(checkpoint_path)

    def detect_main_product(
        self,
        video_frames: List[np.ndarray],
        num_sample_frames: int = 5,
    ) -> np.ndarray:
        """
        Detect the main product/content across video frames.

        Args:
            video_frames: List of video frames (RGB numpy arrays)
            num_sample_frames: Number of frames to sample for analysis

        Returns:
            Binary mask of the main product (H, W)
        """
        print(f"Detecting main product across {len(video_frames)} frames...")

        # Sample frames evenly across video
        sample_indices = np.linspace(0, len(video_frames) - 1, num_sample_frames, dtype=int)
        sample_frames = [video_frames[i] for i in sample_indices]
        print(f"  Sampled {num_sample_frames} frames: {list(sample_indices)}")

        if self.mask_generator is not None:
            # Use SAM for precise segmentation
            main_mask = self._detect_with_sam(sample_frames)
        else:
            # Fallback: Use saliency detection
            main_mask = self._detect_with_saliency(sample_frames)

        return main_mask

    def _detect_with_sam(self, frames: List[np.ndarray]) -> np.ndarray:
        """Detect main product using SAM + consistency tracking."""
        all_segments = []
        all_features = []

        # Get all segments from each frame
        for i, frame in enumerate(frames):
            print(f"  Processing frame {i+1}/{len(frames)}...")

            # Generate masks using SAM
            masks = self.mask_generator.generate(frame)

            # Sort by area (larger = more likely main product)
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)

            # Keep top N segments
            top_masks = masks[:10]

            for mask_data in top_masks:
                mask = mask_data['segmentation']
                bbox = mask_data['bbox']  # x, y, w, h

                # Extract DINO features for this segment
                if self.dino is not None:
                    features = self._extract_dino_features(frame, mask)
                else:
                    features = self._extract_color_histogram(frame, mask)

                all_segments.append({
                    'mask': mask,
                    'area': mask_data['area'],
                    'frame_idx': i,
                    'features': features,
                    'bbox': bbox,
                })

        # Find most consistent segment across frames
        main_segment = self._find_most_consistent(all_segments, len(frames))

        if main_segment is not None:
            print(f"  ✓ Found main product (appeared in {main_segment['consistency']:.0%} of frames)")
            return main_segment['mask'].astype(np.float32)
        else:
            # Fallback: Use center-weighted saliency
            print("  ⚠ No consistent segment found, using center saliency")
            return self._detect_with_saliency(frames)

    def _extract_dino_features(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract DINO features for a masked region."""
        # Get bounding box of mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Crop and resize
        crop = frame[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop).resize((224, 224))

        # Convert to tensor
        crop_tensor = torch.from_numpy(np.array(crop_pil)).permute(2, 0, 1).float() / 255.0
        crop_tensor = crop_tensor.unsqueeze(0).to(self.device)

        # Normalize for DINO
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        crop_tensor = (crop_tensor - mean) / std

        # Extract features
        with torch.no_grad():
            features = self.dino(crop_tensor)

        return features.cpu().numpy().flatten()

    def _extract_color_histogram(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback: Extract color histogram as features."""
        masked_pixels = frame[mask > 0]
        if len(masked_pixels) == 0:
            return np.zeros(64)

        # Compute color histogram
        hist_r = np.histogram(masked_pixels[:, 0], bins=16, range=(0, 255))[0]
        hist_g = np.histogram(masked_pixels[:, 1], bins=16, range=(0, 255))[0]
        hist_b = np.histogram(masked_pixels[:, 2], bins=16, range=(0, 255))[0]

        # Normalize
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)

        return hist

    def _find_most_consistent(
        self,
        segments: List[dict],
        num_frames: int,
        similarity_threshold: float = 0.7,
    ) -> Optional[dict]:
        """Find segment that appears most consistently across frames."""
        if len(segments) == 0:
            return None

        # Group segments by similarity
        clusters = []

        for seg in segments:
            found_cluster = False

            for cluster in clusters:
                # Check similarity with cluster representative
                sim = self._compute_similarity(seg['features'], cluster['features'])
                if sim > similarity_threshold:
                    cluster['segments'].append(seg)
                    cluster['frame_indices'].add(seg['frame_idx'])
                    found_cluster = True
                    break

            if not found_cluster:
                # Create new cluster
                clusters.append({
                    'features': seg['features'],
                    'segments': [seg],
                    'frame_indices': {seg['frame_idx']},
                })

        # Find cluster that appears in most frames
        best_cluster = None
        best_consistency = 0

        for cluster in clusters:
            consistency = len(cluster['frame_indices']) / num_frames

            # Prefer larger segments
            avg_area = np.mean([s['area'] for s in cluster['segments']])

            # Score = consistency × area_factor
            score = consistency * (1 + np.log10(avg_area + 1) / 10)

            if score > best_consistency:
                best_consistency = score
                best_cluster = cluster

        if best_cluster is None:
            return None

        # Return the largest mask from the best cluster
        best_segment = max(best_cluster['segments'], key=lambda x: x['area'])
        best_segment['consistency'] = len(best_cluster['frame_indices']) / num_frames

        return best_segment

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors."""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        return np.dot(feat1, feat2) / (norm1 * norm2)

    def _detect_with_saliency(self, frames: List[np.ndarray]) -> np.ndarray:
        """Fallback: Detect main content using saliency detection."""
        print("  Using saliency detection (fallback)...")

        # Accumulate saliency across frames
        h, w = frames[0].shape[:2]
        accumulated_saliency = np.zeros((h, w), dtype=np.float32)

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        for frame in frames:
            # Compute saliency map
            success, saliency_map = saliency.computeSaliency(frame)

            if success:
                # Resize to match
                saliency_map = cv2.resize(saliency_map, (w, h))
                accumulated_saliency += saliency_map

        # Normalize
        accumulated_saliency = accumulated_saliency / len(frames)

        # Add center prior (main product often in center)
        center_prior = self._create_center_prior(h, w)
        combined = accumulated_saliency * 0.7 + center_prior * 0.3

        # Threshold to binary mask
        threshold = np.percentile(combined, 70)
        mask = (combined > threshold).astype(np.float32)

        # Clean up mask
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _create_center_prior(self, h: int, w: int, sigma: float = 0.3) -> np.ndarray:
        """Create Gaussian center prior."""
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        prior = np.exp(-((x - cx)**2 / (2 * (sigma * w)**2) +
                         (y - cy)**2 / (2 * (sigma * h)**2)))
        return prior

    def detect_for_frame(
        self,
        frame: np.ndarray,
        reference_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Detect main product in a single frame.

        Args:
            frame: Single frame (RGB numpy array)
            reference_features: Features of main product from episodic memory
                               (if None, uses saliency)

        Returns:
            Binary mask of main product
        """
        if self.mask_generator is None or reference_features is None:
            return self._detect_with_saliency([frame])

        # Generate all masks
        masks = self.mask_generator.generate(frame)

        # Find mask most similar to reference
        best_mask = None
        best_sim = -1

        for mask_data in masks:
            mask = mask_data['segmentation']

            if self.dino is not None:
                features = self._extract_dino_features(frame, mask)
            else:
                features = self._extract_color_histogram(frame, mask)

            sim = self._compute_similarity(features, reference_features)

            if sim > best_sim:
                best_sim = sim
                best_mask = mask

        if best_mask is not None and best_sim > 0.5:
            return best_mask.astype(np.float32)
        else:
            return self._detect_with_saliency([frame])


def install_sam():
    """Helper to install SAM if not available."""
    import subprocess
    print("Installing Segment Anything Model...")
    subprocess.check_call([
        "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"
    ])
    print("✓ SAM installed. Please restart.")

"""
Experimental Variant Generator for Attention Heatmaps

Creates different video variants with manipulated attention-keyword alignment
using actual attention heatmaps (not scalar scores).
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm


class VideoVariantGeneratorV3:
    """
    Generates experimental variants with controlled alignment manipulation.

    Uses actual attention heatmaps and keyword masks to create variants.

    Creates 7 variants:
    1. baseline: Original attention heatmaps
    2. early_boost: Boost alignment in first 33% of scenes
    3. middle_boost: Boost alignment in middle 33% of scenes
    4. late_boost: Boost alignment in last 33% of scenes
    5. full_boost: Boost alignment in all scenes
    6. reduction: Reduce alignment in middle 33% of scenes
    7. placebo: Modify non-keyword regions only
    """

    def __init__(
        self,
        valid_scenes_file: str = "data/valid_scenes.csv",
        boost_alpha: float = 1.5,
        reduction_alpha: float = 0.5,
        image_size: Tuple[int, int] = (512, 512),
    ):
        """
        Initialize variant generator.

        Args:
            valid_scenes_file: Path to valid_scenes.csv
            boost_alpha: Multiplication factor for boosting (default: 1.5)
            reduction_alpha: Multiplication factor for reduction (default: 0.5)
            image_size: Target image size (H, W)
        """
        self.boost_alpha = boost_alpha
        self.reduction_alpha = reduction_alpha
        self.image_size = image_size

        # Load valid scenes
        self.scenes_df = pd.read_csv(valid_scenes_file)
        self.scenes_df.columns = self.scenes_df.columns.str.strip()

        print(f"Loaded {len(self.scenes_df)} valid scenes from {self.scenes_df['video_id'].nunique()} videos")

    def _load_heatmap(self, path: str) -> np.ndarray:
        """Load grayscale heatmap and resize."""
        heatmap = Image.open(path).convert('L')
        heatmap = heatmap.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        heatmap = np.array(heatmap).astype(np.float32) / 255.0
        return heatmap

    def _save_heatmap(self, heatmap: np.ndarray, path: str):
        """Save heatmap to file."""
        # Convert to uint8
        heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(heatmap_uint8, mode='L')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)

    def modify_attention_heatmap(
        self,
        attention_heatmap: np.ndarray,
        keyword_mask: np.ndarray,
        alpha: float,
        modify_keyword_region: bool = True
    ) -> np.ndarray:
        """
        Modify attention heatmap based on keyword mask.

        Args:
            attention_heatmap: Original attention heatmap (H, W)
            keyword_mask: Binary keyword mask (H, W)
            alpha: Scaling factor (>1 for boost, <1 for reduction)
            modify_keyword_region: If True, modify keyword region; else modify background

        Returns:
            Modified attention heatmap (H, W) normalized to [0, 1]
        """
        modified = attention_heatmap.copy()

        if modify_keyword_region:
            # Modify attention in keyword region
            modified = modified * (1 - keyword_mask) + (modified * keyword_mask * alpha)
        else:
            # Modify attention in background region (placebo)
            background_mask = 1 - keyword_mask
            modified = modified * (1 - background_mask) + (modified * background_mask * alpha)

        # Renormalize to [0, 1]
        modified = np.clip(modified, 0, 1)

        # Preserve overall attention distribution by normalizing
        if modified.sum() > 0:
            modified = modified / modified.sum() * attention_heatmap.sum()
            modified = np.clip(modified, 0, 1)

        return modified

    def define_temporal_windows(
        self,
        scenes_df: pd.DataFrame,
        window_type: str = "thirds"
    ) -> Dict[str, Tuple[int, int]]:
        """
        Define temporal windows for experimental manipulation.

        Args:
            scenes_df: DataFrame containing scenes for a video
            window_type: Type of windowing ('thirds', 'halves')

        Returns:
            Dict mapping window name to (start_idx, end_idx) in DataFrame
        """
        total_scenes = len(scenes_df)

        if window_type == "thirds":
            third = total_scenes // 3
            return {
                "early": (0, third),
                "middle": (third, 2 * third),
                "late": (2 * third, total_scenes),
            }
        elif window_type == "halves":
            half = total_scenes // 2
            return {
                "first_half": (0, half),
                "second_half": (half, total_scenes),
            }
        else:
            return self.define_temporal_windows(scenes_df, "thirds")

    def create_variant_for_video(
        self,
        video_id: str,
        variant_type: str,
        output_dir: str
    ) -> pd.DataFrame:
        """
        Create a variant for a single video.

        Args:
            video_id: Video identifier
            variant_type: One of: 'baseline', 'early_boost', 'middle_boost',
                          'late_boost', 'full_boost', 'reduction', 'placebo'
            output_dir: Directory to save modified attention heatmaps

        Returns:
            DataFrame with paths to modified attention heatmaps
        """
        # Get scenes for this video
        video_scenes = self.scenes_df[
            self.scenes_df['video_id'].astype(str) == str(video_id)
        ].copy()

        if len(video_scenes) == 0:
            raise ValueError(f"No scenes found for video {video_id}")

        # Sort by scene number
        video_scenes = video_scenes.sort_values('scene_number').reset_index(drop=True)

        # Define temporal windows
        windows = self.define_temporal_windows(video_scenes)

        # Create output directory for this variant
        variant_dir = os.path.join(output_dir, str(video_id), variant_type, "attention_heatmap")
        os.makedirs(variant_dir, exist_ok=True)

        # Process each scene
        modified_heatmap_paths = []

        for idx, row in video_scenes.iterrows():
            # Load attention heatmap and keyword mask
            attention_heatmap = self._load_heatmap(row['attention_heatmap_path'])
            keyword_mask = self._load_heatmap(row['keyword_mask_path'])
            keyword_mask = (keyword_mask > 0.5).astype(np.float32)

            # Determine if this scene should be modified
            should_modify = False
            alpha = 1.0
            modify_keyword = True

            if variant_type == 'baseline':
                # No modification
                modified_heatmap = attention_heatmap

            elif variant_type == 'early_boost':
                start_idx, end_idx = windows['early']
                if start_idx <= idx < end_idx:
                    alpha = self.boost_alpha
                    modified_heatmap = self.modify_attention_heatmap(
                        attention_heatmap, keyword_mask, alpha, modify_keyword_region=True
                    )
                else:
                    modified_heatmap = attention_heatmap

            elif variant_type == 'middle_boost':
                start_idx, end_idx = windows['middle']
                if start_idx <= idx < end_idx:
                    alpha = self.boost_alpha
                    modified_heatmap = self.modify_attention_heatmap(
                        attention_heatmap, keyword_mask, alpha, modify_keyword_region=True
                    )
                else:
                    modified_heatmap = attention_heatmap

            elif variant_type == 'late_boost':
                start_idx, end_idx = windows['late']
                if start_idx <= idx < end_idx:
                    alpha = self.boost_alpha
                    modified_heatmap = self.modify_attention_heatmap(
                        attention_heatmap, keyword_mask, alpha, modify_keyword_region=True
                    )
                else:
                    modified_heatmap = attention_heatmap

            elif variant_type == 'full_boost':
                alpha = self.boost_alpha
                modified_heatmap = self.modify_attention_heatmap(
                    attention_heatmap, keyword_mask, alpha, modify_keyword_region=True
                )

            elif variant_type == 'reduction':
                start_idx, end_idx = windows['middle']
                if start_idx <= idx < end_idx:
                    alpha = self.reduction_alpha
                    modified_heatmap = self.modify_attention_heatmap(
                        attention_heatmap, keyword_mask, alpha, modify_keyword_region=True
                    )
                else:
                    modified_heatmap = attention_heatmap

            elif variant_type == 'placebo':
                # Modify non-keyword regions only
                modified_heatmap = self.modify_attention_heatmap(
                    attention_heatmap, keyword_mask, alpha=1.2, modify_keyword_region=False
                )

            else:
                raise ValueError(f"Unknown variant type: {variant_type}")

            # Save modified heatmap
            filename = row['filename']
            output_path = os.path.join(variant_dir, filename)
            self._save_heatmap(modified_heatmap, output_path)
            modified_heatmap_paths.append(output_path)

        # Add modified paths to DataFrame
        video_scenes['modified_attention_heatmap_path'] = modified_heatmap_paths
        video_scenes['variant'] = variant_type

        return video_scenes

    def create_all_variants_for_video(
        self,
        video_id: str,
        output_dir: str = "outputs/variants"
    ) -> Dict[str, pd.DataFrame]:
        """
        Create all 7 experimental variants for a video.

        Args:
            video_id: Video identifier
            output_dir: Directory to save variants

        Returns:
            Dict mapping variant name to DataFrame with modified scenes
        """
        variant_types = [
            'baseline',
            'early_boost',
            'middle_boost',
            'late_boost',
            'full_boost',
            'reduction',
            'placebo'
        ]

        variants = {}
        for variant_type in tqdm(variant_types, desc=f"Creating variants for {video_id}"):
            variants[variant_type] = self.create_variant_for_video(
                video_id, variant_type, output_dir
            )

        return variants

    def compute_variant_statistics(
        self,
        variants: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Compute statistics for each variant.

        Args:
            variants: Dict of variant DataFrames

        Returns:
            Dict of statistics for each variant
        """
        stats = {}

        for variant_name, df in variants.items():
            # Load and compute statistics from modified heatmaps
            alignment_scores = []

            for _, row in df.iterrows():
                # Load modified attention heatmap
                if 'modified_attention_heatmap_path' in row and pd.notna(row['modified_attention_heatmap_path']):
                    attention_path = row['modified_attention_heatmap_path']
                else:
                    attention_path = row['attention_heatmap_path']

                # Load keyword mask
                keyword_mask = self._load_heatmap(row['keyword_mask_path'])
                keyword_mask = (keyword_mask > 0.5).astype(np.float32)

                # Load attention heatmap
                attention_heatmap = self._load_heatmap(attention_path)

                # Compute alignment score
                alignment_score = float((keyword_mask * attention_heatmap).mean())
                alignment_scores.append(alignment_score)

            stats[variant_name] = {
                'mean_alignment': np.mean(alignment_scores),
                'std_alignment': np.std(alignment_scores),
                'min_alignment': np.min(alignment_scores),
                'max_alignment': np.max(alignment_scores),
                'num_scenes': len(df),
                'temporal_profile': alignment_scores,
            }

        return stats

    def generate_variants_for_all_videos(
        self,
        output_dir: str = "outputs/variants",
        max_videos: Optional[int] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate all variants for all videos.

        Args:
            output_dir: Directory to save variant specifications
            max_videos: Maximum number of videos to process (for testing)

        Returns:
            Dict mapping video_id to dict of variants
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get unique videos
        video_ids = self.scenes_df['video_id'].unique().tolist()

        if max_videos is not None:
            video_ids = video_ids[:max_videos]

        print(f"Generating variants for {len(video_ids)} videos...")

        all_variants = {}
        for video_id in tqdm(video_ids, desc="Processing videos"):
            try:
                variants = self.create_all_variants_for_video(video_id, output_dir)
                all_variants[video_id] = variants

                # Save variant specifications
                video_output_dir = os.path.join(output_dir, str(video_id))
                os.makedirs(video_output_dir, exist_ok=True)

                for variant_name, variant_df in variants.items():
                    variant_df.to_csv(
                        os.path.join(video_output_dir, f"{variant_name}_scenes.csv"),
                        index=False
                    )

                # Save statistics
                stats = self.compute_variant_statistics(variants)
                with open(os.path.join(video_output_dir, "statistics.json"), 'w') as f:
                    json.dump(stats, f, indent=2)

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue

        print(f"\n✓ Generated variants for {len(all_variants)} videos")
        print(f"  Output saved to: {output_dir}")

        return all_variants

    def save_variant_manifest(
        self,
        all_variants: Dict[str, Dict[str, pd.DataFrame]],
        output_path: str
    ):
        """
        Save a manifest of all generated variants.

        Args:
            all_variants: Dict of all variants
            output_path: Path to save manifest JSON
        """
        manifest = {
            'num_videos': len(all_variants),
            'num_variants_per_video': 7,
            'variant_types': [
                'baseline',
                'early_boost',
                'middle_boost',
                'late_boost',
                'full_boost',
                'reduction',
                'placebo'
            ],
            'boost_alpha': self.boost_alpha,
            'reduction_alpha': self.reduction_alpha,
            'videos': {}
        }

        for video_id, variants in all_variants.items():
            baseline_df = variants['baseline']
            keyword = baseline_df.iloc[0]['keyword'] if len(baseline_df) > 0 else "unknown"
            num_scenes = len(baseline_df)

            manifest['videos'][str(video_id)] = {
                'keyword': keyword,
                'num_scenes': num_scenes,
                'variants': list(variants.keys())
            }

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Saved manifest to: {output_path}")


def visualize_variant_comparison(
    variants: Dict[str, pd.DataFrame],
    video_id: str,
    keyword: str,
    generator: VideoVariantGeneratorV3
):
    """
    Visualize alignment profiles for all variants.

    Args:
        variants: Dict of variant DataFrames
        video_id: Video identifier
        keyword: Product keyword
        generator: Variant generator instance
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Alignment scores over scenes
    for variant_name, df in variants.items():
        alignment_scores = []
        scene_numbers = []

        for _, row in df.iterrows():
            # Load heatmaps
            if 'modified_attention_heatmap_path' in row and pd.notna(row['modified_attention_heatmap_path']):
                attention_path = row['modified_attention_heatmap_path']
            else:
                attention_path = row['attention_heatmap_path']

            keyword_mask = generator._load_heatmap(row['keyword_mask_path'])
            keyword_mask = (keyword_mask > 0.5).astype(np.float32)
            attention_heatmap = generator._load_heatmap(attention_path)

            alignment_score = float((keyword_mask * attention_heatmap).mean())
            alignment_scores.append(alignment_score)
            scene_numbers.append(row['scene_number'])

        ax1.plot(scene_numbers, alignment_scores, marker='o', label=variant_name, linewidth=2)

    ax1.set_xlabel('Scene Number', fontsize=12)
    ax1.set_ylabel('Alignment Score', fontsize=12)
    ax1.set_title(f'Alignment Profiles: Video {video_id}\n(Keyword: {keyword})', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean alignment comparison
    variant_names = []
    mean_alignments = []
    std_alignments = []

    for variant_name, df in variants.items():
        alignment_scores = []

        for _, row in df.iterrows():
            if 'modified_attention_heatmap_path' in row and pd.notna(row['modified_attention_heatmap_path']):
                attention_path = row['modified_attention_heatmap_path']
            else:
                attention_path = row['attention_heatmap_path']

            keyword_mask = generator._load_heatmap(row['keyword_mask_path'])
            keyword_mask = (keyword_mask > 0.5).astype(np.float32)
            attention_heatmap = generator._load_heatmap(attention_path)

            alignment_score = float((keyword_mask * attention_heatmap).mean())
            alignment_scores.append(alignment_score)

        variant_names.append(variant_name)
        mean_alignments.append(np.mean(alignment_scores))
        std_alignments.append(np.std(alignment_scores))

    ax2.bar(variant_names, mean_alignments, yerr=std_alignments, capsize=5, alpha=0.7)
    ax2.set_xlabel('Variant', fontsize=12)
    ax2.set_ylabel('Mean Alignment Score', fontsize=12)
    ax2.set_title('Mean Alignment Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

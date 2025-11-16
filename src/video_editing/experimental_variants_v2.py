"""
Updated Experimental Variant Generator

Creates different video variants with manipulated attention-keyword alignment
using pre-computed alignment scores from alignment_score.csv.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json


class VideoVariantGenerator:
    """
    Generates experimental variants with controlled alignment manipulation.

    Uses pre-computed alignment scores (not raw heatmaps).

    Creates 7 variants:
    1. baseline: Original alignment scores
    2. early_boost: Boost alignment in first 33% of scenes
    3. middle_boost: Boost alignment in middle 33% of scenes
    4. late_boost: Boost alignment in last 33% of scenes
    5. full_boost: Boost alignment in all scenes
    6. reduction: Reduce alignment in middle 33% of scenes
    7. placebo: Modify non-keyword regions only
    """

    def __init__(
        self,
        alignment_score_file: str,
        keywords_file: str,
        boost_alpha: float = 1.5,
        reduction_alpha: float = 0.5,
    ):
        """
        Initialize variant generator.

        Args:
            alignment_score_file: Path to alignment_score.csv
            keywords_file: Path to keywords.csv
            boost_alpha: Multiplication factor for boosting (default: 1.5)
            reduction_alpha: Multiplication factor for reduction (default: 0.5)
        """
        self.boost_alpha = boost_alpha
        self.reduction_alpha = reduction_alpha

        # Load data
        self.alignment_df = pd.read_csv(alignment_score_file)
        self.alignment_df.columns = self.alignment_df.columns.str.strip()

        self.keywords_df = pd.read_csv(keywords_file)
        self.keywords_df.columns = self.keywords_df.columns.str.strip()

        # Create keyword mapping
        if '_id' in self.keywords_df.columns:
            video_id_col = '_id'
        elif 'video_id' in self.keywords_df.columns:
            video_id_col = 'video_id'
        else:
            raise ValueError("Could not find video ID column in keywords.csv")

        if 'keyword_list[0]' in self.keywords_df.columns:
            keyword_col = 'keyword_list[0]'
        elif 'keyword' in self.keywords_df.columns:
            keyword_col = 'keyword'
        else:
            raise ValueError("Could not find keyword column in keywords.csv")

        self.keywords = dict(zip(
            self.keywords_df[video_id_col].astype(str),
            self.keywords_df[keyword_col]
        ))

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
            Dict mapping window name to (start_scene, end_scene) indices
        """
        total_scenes = len(scenes_df)

        if window_type == "thirds":
            third = total_scenes // 3
            return {
                "early": (1, third + 1),
                "middle": (third + 1, 2 * third + 1),
                "late": (2 * third + 1, total_scenes + 1),
            }
        elif window_type == "halves":
            half = total_scenes // 2
            return {
                "first_half": (1, half + 1),
                "second_half": (half + 1, total_scenes + 1),
            }
        else:
            return self.define_temporal_windows(scenes_df, "thirds")

    def create_variant_for_video(
        self,
        video_id: str,
        variant_type: str
    ) -> pd.DataFrame:
        """
        Create a variant for a single video.

        Args:
            video_id: Video identifier
            variant_type: One of: 'baseline', 'early_boost', 'middle_boost',
                          'late_boost', 'full_boost', 'reduction', 'placebo'

        Returns:
            DataFrame with modified alignment_proportion for each scene
        """
        # Get scenes for this video
        video_scenes = self.alignment_df[
            self.alignment_df['video id'].astype(str) == str(video_id)
        ].copy()

        if len(video_scenes) == 0:
            raise ValueError(f"No scenes found for video {video_id}")

        # Sort by scene number
        video_scenes = video_scenes.sort_values('Scene Number').reset_index(drop=True)

        # Define temporal windows
        windows = self.define_temporal_windows(video_scenes)

        # Apply variant-specific modification
        if variant_type == 'baseline':
            # No modification
            pass

        elif variant_type == 'early_boost':
            # Boost early scenes
            start_scene, end_scene = windows['early']
            mask = (video_scenes['Scene Number'] >= start_scene) & \
                   (video_scenes['Scene Number'] < end_scene)
            video_scenes.loc[mask, 'attention_proportion'] *= self.boost_alpha
            video_scenes.loc[mask, 'attention_proportion'] = \
                video_scenes.loc[mask, 'attention_proportion'].clip(0, 1)

        elif variant_type == 'middle_boost':
            # Boost middle scenes
            start_scene, end_scene = windows['middle']
            mask = (video_scenes['Scene Number'] >= start_scene) & \
                   (video_scenes['Scene Number'] < end_scene)
            video_scenes.loc[mask, 'attention_proportion'] *= self.boost_alpha
            video_scenes.loc[mask, 'attention_proportion'] = \
                video_scenes.loc[mask, 'attention_proportion'].clip(0, 1)

        elif variant_type == 'late_boost':
            # Boost late scenes
            start_scene, end_scene = windows['late']
            mask = (video_scenes['Scene Number'] >= start_scene) & \
                   (video_scenes['Scene Number'] < end_scene)
            video_scenes.loc[mask, 'attention_proportion'] *= self.boost_alpha
            video_scenes.loc[mask, 'attention_proportion'] = \
                video_scenes.loc[mask, 'attention_proportion'].clip(0, 1)

        elif variant_type == 'full_boost':
            # Boost all scenes
            video_scenes['attention_proportion'] *= self.boost_alpha
            video_scenes['attention_proportion'] = \
                video_scenes['attention_proportion'].clip(0, 1)

        elif variant_type == 'reduction':
            # Reduce middle scenes
            start_scene, end_scene = windows['middle']
            mask = (video_scenes['Scene Number'] >= start_scene) & \
                   (video_scenes['Scene Number'] < end_scene)
            video_scenes.loc[mask, 'attention_proportion'] *= self.reduction_alpha
            video_scenes.loc[mask, 'attention_proportion'] = \
                video_scenes.loc[mask, 'attention_proportion'].clip(0, 1)

        elif variant_type == 'placebo':
            # Placebo: no actual modification to alignment
            # (In practice, would modify non-keyword regions during inference)
            pass

        else:
            raise ValueError(f"Unknown variant type: {variant_type}")

        # Add variant identifier
        video_scenes['variant'] = variant_type

        return video_scenes

    def create_all_variants_for_video(
        self,
        video_id: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Create all 7 experimental variants for a video.

        Args:
            video_id: Video identifier

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
        for variant_type in variant_types:
            variants[variant_type] = self.create_variant_for_video(
                video_id, variant_type
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
            stats[variant_name] = {
                'mean_alignment': df['attention_proportion'].mean(),
                'std_alignment': df['attention_proportion'].std(),
                'min_alignment': df['attention_proportion'].min(),
                'max_alignment': df['attention_proportion'].max(),
                'num_scenes': len(df),
                'temporal_profile': df['attention_proportion'].tolist(),
            }

        return stats

    def generate_variants_for_all_videos(
        self,
        output_dir: str = "outputs/variants"
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate all variants for all videos.

        Uses alignment_score.csv as source of truth for video IDs.

        Args:
            output_dir: Directory to save variant specifications

        Returns:
            Dict mapping video_id to dict of variants
        """
        os.makedirs(output_dir, exist_ok=True)

        # IMPORTANT: Use alignment_score.csv as source of truth
        # Get unique videos from alignment_score.csv that also have keywords
        alignment_video_ids = set(self.alignment_df['video id'].astype(str).unique())
        keyword_video_ids = set(self.keywords.keys())

        # Intersection: videos with both alignment scores AND keywords
        video_ids = sorted(list(alignment_video_ids & keyword_video_ids))

        print(f"Video ID validation:")
        print(f"  Videos in alignment_score.csv: {len(alignment_video_ids)}")
        print(f"  Videos with keywords: {len(keyword_video_ids)}")
        print(f"  Valid videos (will generate variants): {len(video_ids)}")
        print(f"\nGenerating variants for {len(video_ids)} videos...")

        all_variants = {}
        for video_id in video_ids:
            try:
                variants = self.create_all_variants_for_video(video_id)
                all_variants[video_id] = variants

                # Save variant specifications
                video_output_dir = os.path.join(output_dir, str(video_id))
                os.makedirs(video_output_dir, exist_ok=True)

                for variant_name, variant_df in variants.items():
                    variant_df.to_csv(
                        os.path.join(video_output_dir, f"{variant_name}.csv"),
                        index=False
                    )

                # Save statistics
                stats = self.compute_variant_statistics(variants)
                with open(os.path.join(video_output_dir, "statistics.json"), 'w') as f:
                    json.dump(stats, f, indent=2)

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue

        print(f"✓ Generated variants for {len(all_variants)} videos")
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
            keyword = self.keywords.get(str(video_id), "unknown")
            num_scenes = len(variants['baseline'])

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
    keyword: str
):
    """
    Visualize alignment profiles for all variants.

    Args:
        variants: Dict of variant DataFrames
        video_id: Video identifier
        keyword: Product keyword
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    for variant_name, df in variants.items():
        scenes = df['Scene Number'].values
        alignment = df['attention_proportion'].values
        ax.plot(scenes, alignment, marker='o', label=variant_name, linewidth=2)

    ax.set_xlabel('Scene Number', fontsize=12)
    ax.set_ylabel('Alignment Proportion', fontsize=12)
    ax.set_title(f'Experimental Variants: Video {video_id} (Keyword: {keyword})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

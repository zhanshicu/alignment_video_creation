"""
Experimental Variant Generator

Creates different video variants with manipulated attention-keyword alignment
for experimental studies.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import json

from ..data_preparation.control_tensor import ControlTensorBuilder


class ExperimentalVariantGenerator:
    """
    Generates experimental variants with controlled alignment manipulation.

    Creates variants like:
    - Baseline: no change
    - Early boost: increased alignment in early frames
    - Middle boost: increased alignment in middle frames
    - Late boost: increased alignment in late frames
    - Reduction: decreased alignment
    - Placebo: manipulation outside keyword region
    """

    def __init__(
        self,
        control_builder: Optional[ControlTensorBuilder] = None
    ):
        """
        Initialize variant generator.

        Args:
            control_builder: Control tensor builder instance
        """
        if control_builder is None:
            control_builder = ControlTensorBuilder()
        self.control_builder = control_builder

    def define_temporal_windows(
        self,
        total_frames: int,
        window_type: str = "thirds"
    ) -> Dict[str, Tuple[int, int]]:
        """
        Define temporal windows for experimental manipulation.

        Args:
            total_frames: Total number of frames in video
            window_type: Type of windowing ('thirds', 'halves', 'custom')

        Returns:
            Dict mapping window name to (start, end) frame indices
        """
        if window_type == "thirds":
            third = total_frames // 3
            return {
                "early": (0, third),
                "middle": (third, 2 * third),
                "late": (2 * third, total_frames),
            }
        elif window_type == "halves":
            half = total_frames // 2
            return {
                "first_half": (0, half),
                "second_half": (half, total_frames),
            }
        else:
            # Default to thirds
            return self.define_temporal_windows(total_frames, "thirds")

    def create_baseline_variant(
        self,
        attention_maps: List[np.ndarray],
        keyword_maps: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Create baseline variant (no manipulation).

        Args:
            attention_maps: List of attention heatmaps A_t
            keyword_maps: List of keyword heatmaps K_t

        Returns:
            List of control tensors
        """
        control_tensors = []

        for A_t, K_t in zip(attention_maps, keyword_maps):
            C_t = self.control_builder.build_control_tensor(A_t, K_t)
            control_tensors.append(C_t)

        return control_tensors

    def create_alignment_boost_variant(
        self,
        attention_maps: List[np.ndarray],
        keyword_maps: List[np.ndarray],
        alpha: float = 1.5,
        window: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Create variant with boosted alignment in specified window.

        Args:
            attention_maps: List of attention heatmaps
            keyword_maps: List of keyword heatmaps
            alpha: Boost factor (>1 for increase)
            window: Optional (start, end) frame indices for manipulation

        Returns:
            List of control tensors
        """
        control_tensors = []

        if window is None:
            window = (0, len(attention_maps))

        for i, (A_t, K_t) in enumerate(zip(attention_maps, keyword_maps)):
            # Compute base alignment
            S_t = self.control_builder.compute_alignment_map(A_t, K_t)

            # Apply boost if in window
            if window[0] <= i < window[1]:
                S_t = self.control_builder.modify_alignment(S_t, alpha)

            # Build control tensor
            M_t = self.control_builder.compute_keyword_mask(K_t)
            C_t = self.control_builder.build_control_tensor(
                A_t, K_t,
                keyword_mask=M_t,
                alignment_map=S_t
            )
            control_tensors.append(C_t)

        return control_tensors

    def create_alignment_reduction_variant(
        self,
        attention_maps: List[np.ndarray],
        keyword_maps: List[np.ndarray],
        alpha: float = 0.5,
        window: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Create variant with reduced alignment.

        Args:
            attention_maps: List of attention heatmaps
            keyword_maps: List of keyword heatmaps
            alpha: Reduction factor (<1 for decrease)
            window: Optional window for manipulation

        Returns:
            List of control tensors
        """
        return self.create_alignment_boost_variant(
            attention_maps, keyword_maps, alpha, window
        )

    def create_placebo_variant(
        self,
        attention_maps: List[np.ndarray],
        keyword_maps: List[np.ndarray],
        window: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Create placebo variant with manipulation outside keyword region.

        Args:
            attention_maps: List of attention heatmaps
            keyword_maps: List of keyword heatmaps
            window: Optional window for manipulation

        Returns:
            List of control tensors
        """
        control_tensors = []

        if window is None:
            window = (0, len(attention_maps))

        for i, (A_t, K_t) in enumerate(zip(attention_maps, keyword_maps)):
            # Get keyword mask
            M_t = self.control_builder.compute_keyword_mask(K_t)
            B_t = self.control_builder.compute_background_mask(M_t)

            # Compute alignment
            S_t = self.control_builder.compute_alignment_map(A_t, K_t)

            # In window: boost background attention instead of keyword
            if window[0] <= i < window[1]:
                # Boost background attention
                A_t_modified = A_t * (1 + 0.5 * B_t)
                S_t = self.control_builder.compute_alignment_map(A_t_modified, K_t)

            # Build control tensor
            C_t = self.control_builder.build_control_tensor(
                A_t, K_t,
                keyword_mask=M_t,
                alignment_map=S_t
            )
            control_tensors.append(C_t)

        return control_tensors

    def create_all_variants(
        self,
        attention_maps: List[np.ndarray],
        keyword_maps: List[np.ndarray],
        boost_alpha: float = 1.5,
        reduction_alpha: float = 0.5,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Create all standard experimental variants.

        Args:
            attention_maps: List of attention heatmaps
            keyword_maps: List of keyword heatmaps
            boost_alpha: Alpha for boost conditions
            reduction_alpha: Alpha for reduction conditions

        Returns:
            Dict mapping variant name to list of control tensors
        """
        num_frames = len(attention_maps)
        windows = self.define_temporal_windows(num_frames)

        variants = {}

        # Baseline
        variants['baseline'] = self.create_baseline_variant(
            attention_maps, keyword_maps
        )

        # Temporal boost variants
        variants['early_boost'] = self.create_alignment_boost_variant(
            attention_maps, keyword_maps,
            alpha=boost_alpha,
            window=windows['early']
        )

        variants['middle_boost'] = self.create_alignment_boost_variant(
            attention_maps, keyword_maps,
            alpha=boost_alpha,
            window=windows['middle']
        )

        variants['late_boost'] = self.create_alignment_boost_variant(
            attention_maps, keyword_maps,
            alpha=boost_alpha,
            window=windows['late']
        )

        # Full video boost
        variants['full_boost'] = self.create_alignment_boost_variant(
            attention_maps, keyword_maps,
            alpha=boost_alpha,
            window=(0, num_frames)
        )

        # Reduction
        variants['reduction'] = self.create_alignment_reduction_variant(
            attention_maps, keyword_maps,
            alpha=reduction_alpha,
            window=windows['middle']
        )

        # Placebo
        variants['placebo'] = self.create_placebo_variant(
            attention_maps, keyword_maps,
            window=windows['middle']
        )

        return variants

    def compute_variant_statistics(
        self,
        variants: Dict[str, List[np.ndarray]],
        attention_maps: List[np.ndarray],
        keyword_masks: List[np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Compute statistics for each variant.

        Args:
            variants: Dict of variants
            attention_maps: Original attention maps
            keyword_masks: Keyword masks

        Returns:
            Dict of statistics for each variant
        """
        stats = {}

        for variant_name, control_tensors in variants.items():
            # Extract alignment maps (channel 1)
            alignment_maps = [C[:, :, 1] for C in control_tensors]

            # Compute mean alignment over time
            mean_alignment = np.mean([
                self.control_builder.calculate_alignment_score(A_t, M_t)
                for A_t, M_t in zip(attention_maps, keyword_masks)
            ])

            # Compute alignment from control tensors
            mean_controlled_alignment = np.mean([S.mean() for S in alignment_maps])

            # Temporal profile
            temporal_profile = [S.mean() for S in alignment_maps]

            stats[variant_name] = {
                'mean_alignment_score': mean_alignment,
                'mean_controlled_alignment': mean_controlled_alignment,
                'temporal_profile': temporal_profile,
                'num_frames': len(control_tensors),
            }

        return stats

    def save_variants(
        self,
        variants: Dict[str, List[np.ndarray]],
        output_dir: str,
        video_name: str
    ):
        """
        Save all variants to disk.

        Args:
            variants: Dict of variant control tensors
            output_dir: Output directory
            video_name: Video identifier
        """
        os.makedirs(output_dir, exist_ok=True)

        for variant_name, control_tensors in variants.items():
            variant_dir = os.path.join(output_dir, video_name, variant_name)
            os.makedirs(variant_dir, exist_ok=True)

            # Save each control tensor
            for i, C_t in enumerate(control_tensors):
                np.save(
                    os.path.join(variant_dir, f"control_{i:05d}.npy"),
                    C_t
                )

        print(f"Saved {len(variants)} variants to {output_dir}/{video_name}")

    def save_variant_metadata(
        self,
        variants: Dict[str, List[np.ndarray]],
        statistics: Dict[str, Dict],
        output_path: str,
        video_info: Optional[Dict] = None
    ):
        """
        Save metadata about variants.

        Args:
            variants: Dict of variants
            statistics: Statistics for each variant
            output_path: Output JSON file path
            video_info: Optional additional video information
        """
        metadata = {
            'variants': list(variants.keys()),
            'num_frames': len(list(variants.values())[0]),
            'statistics': statistics,
        }

        if video_info is not None:
            metadata['video_info'] = video_info

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {output_path}")

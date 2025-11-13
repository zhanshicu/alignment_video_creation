"""
Example: Creating Experimental Variants

This script demonstrates how to create different experimental variants
with manipulated attention-keyword alignment.
"""

import os
import sys
import numpy as np
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preparation import ControlTensorBuilder
from src.video_editing import ExperimentalVariantGenerator
from src.utils import visualize_control_tensor, plot_temporal_alignment, compare_variants


def main():
    # Load configuration
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("Creating Experimental Variants")
    print("=" * 50)

    # Load pre-computed heatmaps
    data_dir = "data/processed"
    output_dir = "data/variants"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/4] Loading heatmaps...")

    # For demonstration, create mock data
    num_frames = 90  # 3 seconds at 30 FPS
    H, W = 512, 512

    # Simulate attention and keyword heatmaps
    attention_maps = []
    keyword_maps = []

    for i in range(num_frames):
        # Mock attention that varies over time
        attention = np.random.rand(H, W) * 0.5 + 0.5 * np.sin(i / 10)
        attention = np.clip(attention, 0, 1)
        attention_maps.append(attention)

        # Mock keyword region (static)
        y, x = np.ogrid[:H, :W]
        keyword = np.exp(-((x - W//2)**2 + (y - H//2)**2) / (2 * (W//6)**2))
        keyword_maps.append(keyword)

    print(f"Loaded {num_frames} frames of heatmaps")

    # Step 2: Initialize variant generator
    print("\n[2/4] Initializing variant generator...")

    control_builder = ControlTensorBuilder(include_raw_maps=False)
    variant_gen = ExperimentalVariantGenerator(control_builder)

    # Step 3: Create all variants
    print("\n[3/4] Creating experimental variants...")

    variants = variant_gen.create_all_variants(
        attention_maps=attention_maps,
        keyword_maps=keyword_maps,
        boost_alpha=config['experiments']['boost_alpha'],
        reduction_alpha=config['experiments']['reduction_alpha'],
    )

    print(f"Created {len(variants)} variants:")
    for variant_name in variants.keys():
        print(f"  - {variant_name}")

    # Step 4: Compute statistics
    print("\n[4/4] Computing variant statistics...")

    keyword_masks = [
        control_builder.compute_keyword_mask(k) for k in keyword_maps
    ]

    stats = variant_gen.compute_variant_statistics(
        variants, attention_maps, keyword_masks
    )

    for variant_name, variant_stats in stats.items():
        print(f"\n{variant_name}:")
        print(f"  Mean alignment: {variant_stats['mean_alignment_score']:.4f}")
        print(f"  Mean controlled alignment: {variant_stats['mean_controlled_alignment']:.4f}")

    # Save variants
    print("\nSaving variants...")
    video_name = "example_video"

    variant_gen.save_variants(variants, output_dir, video_name)
    variant_gen.save_variant_metadata(
        variants, stats,
        os.path.join(output_dir, video_name, "metadata.json"),
        video_info={'fps': 30, 'duration': 3.0}
    )

    # Visualize
    print("\nCreating visualizations...")

    # Temporal alignment profiles
    windows = variant_gen.define_temporal_windows(num_frames)
    plot_temporal_alignment(
        stats['baseline']['temporal_profile'],
        window_labels=windows,
        save_path=os.path.join(output_dir, "temporal_profile_baseline.png"),
        title="Baseline Temporal Alignment"
    )

    # Compare variants
    compare_variants(
        stats,
        save_path=os.path.join(output_dir, "variants_comparison.png")
    )

    # Visualize control tensor for middle frame
    middle_frame = num_frames // 2
    for variant_name, control_tensors in list(variants.items())[:3]:
        visualize_control_tensor(
            control_tensors[middle_frame],
            save_path=os.path.join(output_dir, f"control_{variant_name}.png")
        )

    print("\n" + "=" * 50)
    print("Variant creation complete!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 50)

    print("\nNext steps:")
    print("1. Use these control tensors with the video editor")
    print("2. Generate edited videos for each variant")
    print("3. Run lab experiments to assess human responses")


if __name__ == "__main__":
    main()

"""
Example: Data Preparation Pipeline

This script demonstrates how to:
1. Load video frames
2. Generate attention heatmaps
3. Generate keyword heatmaps
4. Build control tensors
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preparation import (
    AttentionHeatmapGenerator,
    KeywordHeatmapGenerator,
    ControlTensorBuilder
)
from src.utils import VideoLoader, visualize_heatmaps


def main():
    # Configuration
    video_path = "data/raw/video.mp4"
    keyword = "jewelry"
    output_dir = "data/processed"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("Data Preparation Pipeline")
    print("=" * 50)

    # Step 1: Load video frames
    print("\n[1/4] Loading video frames...")
    loader = VideoLoader()
    frames, video_info = loader.load_video_frames(video_path)
    print(f"Loaded {len(frames)} frames at {video_info['fps']} FPS")

    # Step 2: Extract patches for attention heatmap
    print("\n[2/4] Extracting circular patches for attention scoring...")
    attention_gen = AttentionHeatmapGenerator(
        patch_degrees=[3, 7],
        overlap=0.1,
        sigma=5.0,
        gamma=3.0
    )

    # Process one frame as example
    frame = frames[0]
    scene_name = "example_frame_000"

    patch_dir = os.path.join(output_dir, "patches")
    metadata_dir = os.path.join(output_dir, "metadata")
    csv_path = os.path.join(output_dir, "patch_info.csv")

    metadata = attention_gen.extract_circular_patches(
        frame, scene_name,
        save_dir_patch=os.path.join(patch_dir, scene_name),
        save_dir_meta=metadata_dir
    )
    attention_gen.save_patch_info_to_csv(metadata, scene_name, csv_path)

    print(f"Extracted {len(metadata)} patches")
    print(f"CSV saved to: {csv_path}")
    print("\nNote: Run your LLaVA model to score these patches.")
    print("The scored CSV should have a 'likert_label_predicted' or 'mean_score' column.")

    # For demonstration, create a mock attention heatmap
    print("\n[3/4] Generating keyword heatmap...")
    attention_map = np.random.rand(*frame.shape[:2])  # Mock attention
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Generate keyword heatmap using CLIPSeg
    keyword_gen = KeywordHeatmapGenerator(threshold=0.5)
    keyword_map = keyword_gen.generate_keyword_heatmap(frame, keyword)

    print(f"Keyword heatmap shape: {keyword_map.shape}")

    # Step 4: Build control tensor
    print("\n[4/4] Building control tensor...")
    control_builder = ControlTensorBuilder(include_raw_maps=False)

    control_tensor = control_builder.build_control_tensor(
        attention_map, keyword_map
    )

    print(f"Control tensor shape: {control_tensor.shape}")
    print(f"  - Channel 0: Keyword Mask (M_t)")
    print(f"  - Channel 1: Alignment Map (S_t)")

    # Visualize
    print("\nVisualizing results...")
    vis_path = os.path.join(output_dir, "visualization.png")
    visualize_heatmaps(
        frame, attention_map, keyword_map,
        alignment_map=control_tensor[:, :, 1],
        save_path=vis_path
    )

    # Save outputs
    np.save(os.path.join(output_dir, "attention_map.npy"), attention_map)
    np.save(os.path.join(output_dir, "keyword_map.npy"), keyword_map)
    np.save(os.path.join(output_dir, "control_tensor.npy"), control_tensor)

    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()

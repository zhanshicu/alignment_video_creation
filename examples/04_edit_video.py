"""
Example: Video Editing with Temporal Consistency

This script demonstrates how to edit a video using the trained model
with temporal consistency.
"""

import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import StableDiffusionControlNetWrapper
from src.video_editing import VideoEditor, TemporalConsistencyWrapper
from src.utils import VideoLoader, VideoSaver


def main():
    # Load configuration
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("Video Editing with Temporal Consistency")
    print("=" * 50)

    # Configuration
    video_path = "data/raw/video.mp4"
    variant_dir = "data/variants/example_video/middle_boost"
    checkpoint_path = "outputs/best_model.pt"
    output_dir = "outputs/edited_videos"
    keyword = "jewelry"

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Load model
    print(f"\n[1/5] Loading trained model from {checkpoint_path}...")

    model = StableDiffusionControlNetWrapper(
        sd_model_name=config['model']['sd_model_name'],
        controlnet_config=config['model']['controlnet'],
        device=device,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.controlnet.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")

    # Step 2: Load video and control tensors
    print("\n[2/5] Loading video and control tensors...")

    loader = VideoLoader()
    frames, video_info = loader.load_video_frames(video_path)

    # Load control tensors for this variant
    control_tensors = []
    for i in range(len(frames)):
        control_path = os.path.join(variant_dir, f"control_{i:05d}.npy")
        if os.path.exists(control_path):
            control = np.load(control_path)
            control_tensors.append(control)
        else:
            print(f"Warning: Control tensor {i} not found, creating default")
            # Create default control tensor (mock)
            control = np.random.rand(512, 512, 2)
            control_tensors.append(control)

    print(f"Loaded {len(frames)} frames and {len(control_tensors)} control tensors")

    # Step 3: Edit keyframes
    print("\n[3/5] Editing keyframes...")

    editor = VideoEditor(model, device=device)

    edited_keyframes = editor.edit_keyframes_only(
        frames=frames,
        control_tensors=control_tensors,
        keyword=keyword,
        keyframe_interval=config['video_editing']['keyframe_interval'],
        num_inference_steps=config['video_editing']['num_inference_steps'],
        guidance_scale=config['video_editing']['guidance_scale'],
        strength=config['video_editing']['strength'],
    )

    print(f"Edited {len(edited_keyframes)} keyframes")

    # Step 4: Propagate edits for temporal consistency
    print("\n[4/5] Propagating edits for temporal consistency...")

    temporal_wrapper = TemporalConsistencyWrapper(
        method=config['video_editing']['temporal_method']
    )

    edited_frames = temporal_wrapper.propagate_edits(
        original_frames=frames,
        edited_keyframes=edited_keyframes,
        blend_window=config['video_editing']['blend_window'],
    )

    # Apply temporal smoothing
    edited_frames = temporal_wrapper.apply_temporal_smoothing(edited_frames)

    print("Temporal consistency ensured")

    # Step 5: Save results
    print("\n[5/5] Saving results...")

    # Save edited video
    output_video = os.path.join(output_dir, "edited_middle_boost.mp4")
    saver = VideoSaver()
    saver.save_video(edited_frames, output_video, fps=video_info['fps'])

    # Save side-by-side comparison
    comparison_video = os.path.join(output_dir, "comparison_middle_boost.mp4")
    saver.create_side_by_side_video(
        frames, edited_frames, comparison_video,
        fps=video_info['fps'],
        labels=("Original", "Middle Boost")
    )

    # Save keyframes as images
    keyframes_dir = os.path.join(output_dir, "keyframes")
    for idx, frame in edited_keyframes.items():
        os.makedirs(keyframes_dir, exist_ok=True)
        frame_path = os.path.join(keyframes_dir, f"keyframe_{idx:05d}.png")
        import cv2
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 50)
    print("Video editing complete!")
    print(f"Edited video: {output_video}")
    print(f"Comparison: {comparison_video}")
    print(f"Keyframes: {keyframes_dir}")
    print("=" * 50)

    # Optional: Integration with Rerender A Video
    print("\nOptional: Preparing files for Rerender A Video...")
    rerender_dir = os.path.join(output_dir, "rerender_input")
    temporal_wrapper.save_for_rerender(
        frames, edited_keyframes, rerender_dir, video_name="example"
    )


if __name__ == "__main__":
    main()

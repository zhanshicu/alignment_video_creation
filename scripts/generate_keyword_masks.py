#!/usr/bin/env python3
"""
Generate keyword masks from screenshots using CLIPSeg.

This script processes screenshots and generates spatial masks indicating
where the product/keyword appears in each scene.

Usage:
    python scripts/generate_keyword_masks.py --screenshots_dir data/screenshots_tiktok \
                                              --keywords_file data/keywords.csv \
                                              --output_dir data/keyword_masks
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_clipseg_model(device='cuda'):
    """Load CLIPSeg model for zero-shot segmentation."""
    print("Loading CLIPSeg model...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)
    model.eval()
    print("✓ CLIPSeg model loaded")
    return processor, model


def generate_keyword_mask(image, keyword, processor, model, device='cuda', threshold=0.5):
    """
    Generate spatial mask for a keyword in an image.

    Args:
        image: PIL Image
        keyword: Text description of product/object to segment
        processor: CLIPSeg processor
        model: CLIPSeg model
        device: Device to run on
        threshold: Threshold for binarizing mask (default: 0.5)

    Returns:
        mask: numpy array [H, W] with values in [0, 255]
    """
    # Prepare inputs
    inputs = processor(
        text=[keyword],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get segmentation mask
    preds = outputs.logits  # [1, H, W]
    mask = torch.sigmoid(preds[0])  # Normalize to [0, 1]

    # Resize to original image size
    mask = mask.cpu().numpy()
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(image.size, Image.BILINEAR)

    # Convert back to numpy
    mask = np.array(mask_pil).astype(np.float32) / 255.0

    # Apply threshold
    mask = (mask > threshold).astype(np.uint8) * 255

    return mask


def process_video_screenshots(video_id, keyword, screenshots_dir, output_dir,
                               processor, model, device='cuda'):
    """
    Process all screenshots for a single video.

    Args:
        video_id: Video identifier
        keyword: Product keyword for this video
        screenshots_dir: Directory containing screenshots
        output_dir: Output directory for masks
        processor: CLIPSeg processor
        model: CLIPSeg model
        device: Device to run on

    Returns:
        num_processed: Number of screenshots processed
    """
    # Find screenshot directory
    video_screenshot_dir = os.path.join(screenshots_dir, str(video_id))

    if not os.path.exists(video_screenshot_dir):
        print(f"  Warning: Screenshot directory not found for video {video_id}")
        return 0

    # Create output directory
    video_output_dir = os.path.join(output_dir, str(video_id))
    os.makedirs(video_output_dir, exist_ok=True)

    # Get all screenshot files
    screenshot_files = sorted([
        f for f in os.listdir(video_screenshot_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(screenshot_files) == 0:
        print(f"  Warning: No screenshots found for video {video_id}")
        return 0

    # Process each screenshot
    for screenshot_file in screenshot_files:
        screenshot_path = os.path.join(video_screenshot_dir, screenshot_file)

        # Load image
        image = Image.open(screenshot_path).convert('RGB')

        # Generate mask
        mask = generate_keyword_mask(
            image, keyword, processor, model, device
        )

        # Save mask
        mask_filename = screenshot_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(video_output_dir, mask_filename)
        Image.fromarray(mask).save(mask_path)

    return len(screenshot_files)


def main():
    parser = argparse.ArgumentParser(description='Generate keyword masks from screenshots')
    parser.add_argument('--screenshots_dir', type=str, default='data/screenshots_tiktok',
                        help='Directory containing screenshots organized by video_id')
    parser.add_argument('--keywords_file', type=str, default='data/keywords.csv',
                        help='CSV file with video_id and keywords')
    parser.add_argument('--output_dir', type=str, default='data/keyword_masks',
                        help='Output directory for keyword masks')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Threshold for binarizing masks (default: 0.4)')

    args = parser.parse_args()

    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, running on CPU (will be slow)")

    # Load keywords
    print(f"\nLoading keywords from: {args.keywords_file}")
    keywords_df = pd.read_csv(args.keywords_file)
    keywords_df.columns = keywords_df.columns.str.strip()  # Remove whitespace

    # Handle different possible column names
    if '_id' in keywords_df.columns:
        video_id_col = '_id'
    elif 'video_id' in keywords_df.columns:
        video_id_col = 'video_id'
    else:
        raise ValueError(f"Could not find video ID column. Available columns: {keywords_df.columns.tolist()}")

    if 'keyword_list[0]' in keywords_df.columns:
        keyword_col = 'keyword_list[0]'
    elif 'keyword' in keywords_df.columns:
        keyword_col = 'keyword'
    else:
        raise ValueError(f"Could not find keyword column. Available columns: {keywords_df.columns.tolist()}")

    # Filter out rows with missing keywords
    keywords_df = keywords_df[keywords_df[keyword_col].notna() & (keywords_df[keyword_col] != '')]

    print(f"Found {len(keywords_df)} videos with keywords")

    # Load model
    processor, model = load_clipseg_model(device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each video
    print("\nGenerating keyword masks...")
    total_processed = 0

    for idx, row in tqdm(keywords_df.iterrows(), total=len(keywords_df), desc="Processing videos"):
        video_id = row[video_id_col]
        keyword = row[keyword_col]

        if pd.isna(keyword) or keyword == '':
            continue

        num_processed = process_video_screenshots(
            video_id=video_id,
            keyword=keyword,
            screenshots_dir=args.screenshots_dir,
            output_dir=args.output_dir,
            processor=processor,
            model=model,
            device=device
        )

        total_processed += num_processed

    print(f"\n✓ Complete!")
    print(f"  Processed {total_processed} screenshots")
    print(f"  Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

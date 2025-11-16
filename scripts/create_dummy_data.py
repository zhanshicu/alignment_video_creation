#!/usr/bin/env python3
"""
Generate dummy test data for pipeline testing.

This creates synthetic screenshots and keyword masks for testing the framework
when real data is not available.

Usage:
    python scripts/create_dummy_data.py --num_videos 5
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_dummy_screenshot(video_id, scene_number, image_size=512):
    """
    Create a dummy screenshot with some visual content.

    Args:
        video_id: Video identifier
        scene_number: Scene number
        image_size: Image size (default: 512x512)

    Returns:
        PIL Image
    """
    # Create image with gradient background
    img = Image.new('RGB', (image_size, image_size))
    draw = ImageDraw.Draw(img)

    # Create gradient background
    for y in range(image_size):
        color_val = int(100 + (y / image_size) * 100)
        draw.line([(0, y), (image_size, y)], fill=(color_val, color_val, color_val + 20))

    # Draw a colored rectangle (simulating product)
    product_color = (200, 100, 50)  # Orange-ish
    product_x = image_size // 4
    product_y = image_size // 4
    product_w = image_size // 2
    product_h = image_size // 2

    draw.rectangle(
        [product_x, product_y, product_x + product_w, product_y + product_h],
        fill=product_color,
        outline=(255, 255, 255),
        width=3
    )

    # Add some circles (simulating details)
    for i in range(3):
        cx = product_x + product_w // 2
        cy = product_y + product_h // 4 * (i + 1)
        r = 20
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))

    # Add text
    try:
        # Try to add text (font may not be available)
        draw.text(
            (10, 10),
            f"Video: {video_id}\nScene: {scene_number}",
            fill=(255, 255, 255)
        )
    except:
        pass

    return img


def create_dummy_keyword_mask(video_id, scene_number, image_size=512):
    """
    Create a dummy keyword mask corresponding to the screenshot.

    Args:
        video_id: Video identifier
        scene_number: Scene number
        image_size: Image size (default: 512x512)

    Returns:
        PIL Image (grayscale)
    """
    # Create mask
    mask = Image.new('L', (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)

    # Draw white rectangle where the "product" is
    product_x = image_size // 4
    product_y = image_size // 4
    product_w = image_size // 2
    product_h = image_size // 2

    draw.rectangle(
        [product_x, product_y, product_x + product_w, product_y + product_h],
        fill=255
    )

    return mask


def create_dummy_data_for_videos(
    alignment_score_file: str,
    num_videos: int = None,
    screenshots_dir: str = 'data/screenshots_tiktok',
    keyword_masks_dir: str = 'data/keyword_masks',
    image_size: int = 512
):
    """
    Create dummy data for videos in alignment_score.csv.

    Args:
        alignment_score_file: Path to alignment_score.csv
        num_videos: Number of videos to create data for (None = all)
        screenshots_dir: Output directory for screenshots
        keyword_masks_dir: Output directory for keyword masks
        image_size: Image size (default: 512)
    """
    # Load alignment scores
    alignment_df = pd.read_csv(alignment_score_file)
    alignment_df.columns = alignment_df.columns.str.strip()

    # Get unique video IDs
    video_ids = alignment_df['video id'].unique()

    if num_videos is not None:
        video_ids = video_ids[:num_videos]

    print(f"Creating dummy data for {len(video_ids)} videos...")
    print(f"  Screenshots: {screenshots_dir}")
    print(f"  Keyword masks: {keyword_masks_dir}\n")

    total_scenes = 0

    for video_id in video_ids:
        # Get scenes for this video
        video_scenes = alignment_df[alignment_df['video id'] == video_id]
        scene_numbers = video_scenes['Scene Number'].values

        # Create directories
        screenshot_video_dir = os.path.join(screenshots_dir, str(video_id))
        mask_video_dir = os.path.join(keyword_masks_dir, str(video_id))
        os.makedirs(screenshot_video_dir, exist_ok=True)
        os.makedirs(mask_video_dir, exist_ok=True)

        # Create data for each scene
        for scene_num in scene_numbers:
            # Create screenshot
            screenshot = create_dummy_screenshot(video_id, scene_num, image_size)
            screenshot_path = os.path.join(screenshot_video_dir, f"scene_{scene_num}.png")
            screenshot.save(screenshot_path)

            # Create keyword mask
            mask = create_dummy_keyword_mask(video_id, scene_num, image_size)
            mask_path = os.path.join(mask_video_dir, f"scene_{scene_num}.png")
            mask.save(mask_path)

            total_scenes += 1

        print(f"  ✓ Video {video_id}: {len(scene_numbers)} scenes")

    print(f"\n✓ Created dummy data:")
    print(f"  Videos: {len(video_ids)}")
    print(f"  Total scenes: {total_scenes}")
    print(f"  Screenshots: {screenshots_dir}/")
    print(f"  Keyword masks: {keyword_masks_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Create dummy test data for pipeline testing'
    )
    parser.add_argument(
        '--alignment_score_file',
        type=str,
        default='data/alignment_score.csv',
        help='Path to alignment_score.csv'
    )
    parser.add_argument(
        '--num_videos',
        type=int,
        default=5,
        help='Number of videos to create data for (default: 5)'
    )
    parser.add_argument(
        '--screenshots_dir',
        type=str,
        default='data/screenshots_tiktok',
        help='Output directory for screenshots'
    )
    parser.add_argument(
        '--keyword_masks_dir',
        type=str,
        default='data/keyword_masks',
        help='Output directory for keyword masks'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='Image size (default: 512)'
    )

    args = parser.parse_args()

    create_dummy_data_for_videos(
        alignment_score_file=args.alignment_score_file,
        num_videos=args.num_videos,
        screenshots_dir=args.screenshots_dir,
        keyword_masks_dir=args.keyword_masks_dir,
        image_size=args.image_size,
    )


if __name__ == '__main__':
    main()

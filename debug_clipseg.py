#!/usr/bin/env python3
"""
Diagnostic script to debug CLIPSeg mask generation.
Run this to see why masks aren't being generated.
"""

import os
import pandas as pd
from pathlib import Path

# Configuration
ALIGNMENT_SCORE_FILE = 'data/alignment_score.csv'
KEYWORDS_FILE = 'data/keywords.csv'
SCREENSHOTS_DIR = 'data/screenshots_tiktok'
KEYWORD_MASKS_DIR = 'data/keyword_masks'

print("=" * 70)
print("CLIPSeg Mask Generation Diagnostic")
print("=" * 70)

# 1. Check if files exist
print("\n1. Checking data files...")
print(f"   alignment_score.csv exists: {os.path.exists(ALIGNMENT_SCORE_FILE)}")
print(f"   keywords.csv exists: {os.path.exists(KEYWORDS_FILE)}")
print(f"   screenshots_tiktok/ exists: {os.path.exists(SCREENSHOTS_DIR)}")
print(f"   keyword_masks/ exists: {os.path.exists(KEYWORD_MASKS_DIR)}")

# 2. Load alignment scores
print("\n2. Loading alignment_score.csv...")
alignment_df = pd.read_csv(ALIGNMENT_SCORE_FILE)
alignment_df.columns = alignment_df.columns.str.strip()
print(f"   Columns: {list(alignment_df.columns)}")
print(f"   Total rows: {len(alignment_df)}")
print(f"   Unique videos: {alignment_df['video id'].nunique()}")

# 3. Load keywords
print("\n3. Loading keywords.csv...")
keywords_df = pd.read_csv(KEYWORDS_FILE)
keywords_df.columns = keywords_df.columns.str.strip()
print(f"   Columns: {list(keywords_df.columns)}")
print(f"   Total rows: {len(keywords_df)}")

# Determine column names
if '_id' in keywords_df.columns:
    video_id_col = '_id'
else:
    video_id_col = 'video_id'

if 'keyword_list[0]' in keywords_df.columns:
    keyword_col = 'keyword_list[0]'
else:
    keyword_col = 'keyword'

print(f"   Video ID column: '{video_id_col}'")
print(f"   Keyword column: '{keyword_col}'")
print(f"   Unique videos: {keywords_df[video_id_col].nunique()}")

# Check for missing keywords
missing_keywords = keywords_df[keyword_col].isna().sum()
print(f"   Missing keywords: {missing_keywords}/{len(keywords_df)}")

# 4. Find intersection
print("\n4. Finding valid video IDs (intersection)...")
alignment_video_ids = set(alignment_df['video id'].astype(str).unique())
keyword_video_ids = set(keywords_df[video_id_col].astype(str).unique())
valid_video_ids = sorted(list(alignment_video_ids & keyword_video_ids))

print(f"   Videos in alignment_score.csv: {len(alignment_video_ids)}")
print(f"   Videos in keywords.csv: {len(keyword_video_ids)}")
print(f"   Videos in BOTH (valid): {len(valid_video_ids)}")
print(f"   First 5 valid videos: {valid_video_ids[:5]}")

# 5. Check screenshots for first few videos
print("\n5. Checking screenshots for first 5 valid videos...")

if not os.path.exists(SCREENSHOTS_DIR):
    print(f"   ❌ ERROR: Directory '{SCREENSHOTS_DIR}' does not exist!")
    print(f"   This is why 0 masks were generated.")
    print(f"\n   Solution:")
    print(f"   - Ensure screenshots are in '{SCREENSHOTS_DIR}/' directory")
    print(f"   - Structure should be: {SCREENSHOTS_DIR}/{{video_id}}/scene_{{N}}.png")
else:
    for video_id in valid_video_ids[:5]:
        video_dir = os.path.join(SCREENSHOTS_DIR, str(video_id))

        if not os.path.exists(video_dir):
            print(f"   ❌ Video {video_id}: directory doesn't exist")
            continue

        # Check what files are in this directory
        files = os.listdir(video_dir)
        scene_files = [f for f in files if f.startswith('scene_')]

        print(f"   ✓ Video {video_id}: {len(scene_files)} scene files")
        if len(scene_files) > 0:
            print(f"      Examples: {scene_files[:3]}")

        # Check if scene numbers match alignment data
        video_scenes = alignment_df[alignment_df['video id'] == int(video_id)]
        expected_scenes = video_scenes['Scene Number'].values
        print(f"      Expected {len(expected_scenes)} scenes: {expected_scenes[:5]}...")

        # Check naming conventions
        for scene_num in expected_scenes[:3]:
            path1 = os.path.join(video_dir, f"scene_{scene_num}.png")
            path2 = os.path.join(video_dir, f"scene_{scene_num:02d}.png")

            exists1 = os.path.exists(path1)
            exists2 = os.path.exists(path2)

            if exists1:
                print(f"      ✓ Scene {scene_num}: found at scene_{scene_num}.png")
            elif exists2:
                print(f"      ✓ Scene {scene_num}: found at scene_{scene_num:02d}.png")
            else:
                print(f"      ❌ Scene {scene_num}: NOT FOUND (tried scene_{scene_num}.png and scene_{scene_num:02d}.png)")

# 6. Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if not os.path.exists(SCREENSHOTS_DIR):
    print("❌ PROBLEM: Screenshots directory does not exist!")
    print(f"   Expected: {SCREENSHOTS_DIR}/")
    print(f"\n   Next steps:")
    print(f"   1. Make sure your screenshots are uploaded to '{SCREENSHOTS_DIR}/'")
    print(f"   2. Directory structure should be:")
    print(f"      {SCREENSHOTS_DIR}/")
    print(f"      ├── 7163329870906884097/")
    print(f"      │   ├── scene_1.png")
    print(f"      │   ├── scene_2.png")
    print(f"      │   └── ...")
    print(f"      ├── 7195702874672234498/")
    print(f"      │   └── ...")
    print(f"   3. Re-run the mask generation cell in the notebook")
else:
    print("✓ Screenshots directory exists")
    print(f"\n   If masks still aren't generated, check:")
    print(f"   1. Scene file naming (scene_N.png vs scene_0N.png)")
    print(f"   2. Video IDs match between alignment_score.csv and folder names")
    print(f"   3. Scene numbers in CSV match the screenshot filenames")

print("=" * 70)

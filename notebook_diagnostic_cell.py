"""
Add this cell to your notebook BEFORE the mask generation cell to diagnose the issue
"""

# ========================================
# DIAGNOSTIC CELL - Add this to notebook
# ========================================

import os
import pandas as pd

# Check 1: Directory exists?
print("=" * 70)
print("DIAGNOSTIC: Checking data structure")
print("=" * 70)

print(f"\n1. Screenshots directory exists: {os.path.exists(SCREENSHOTS_DIR)}")
if os.path.exists(SCREENSHOTS_DIR):
    video_folders = os.listdir(SCREENSHOTS_DIR)
    print(f"   Number of video folders: {len(video_folders)}")
    print(f"   First 5 folders: {video_folders[:5]}")
else:
    print("   ❌ STOP: Directory does not exist!")
    print(f"   Expected: {SCREENSHOTS_DIR}")

# Check 2: How many videos are we trying to process?
print(f"\n2. Videos selected for training: {len(video_ids_for_training)}")
print(f"   Video IDs: {video_ids_for_training[:5]}...")

# Check 3: Load alignment data and check scenes
alignment_df = pd.read_csv(ALIGNMENT_SCORE_FILE)
alignment_df.columns = alignment_df.columns.str.strip()

print(f"\n3. Alignment data:")
print(f"   Total rows: {len(alignment_df)}")
print(f"   Unique videos: {alignment_df['video id'].nunique()}")
print(f"   Video ID type: {type(alignment_df['video id'].iloc[0])}")

# Check 4: For first video, check scenes
if len(video_ids_for_training) > 0:
    test_video_id = video_ids_for_training[0]
    print(f"\n4. Testing first video: {test_video_id}")
    print(f"   Type: {type(test_video_id)}")

    # Try to filter alignment_df
    video_scenes = alignment_df[alignment_df['video id'] == test_video_id]
    print(f"   Scenes found (direct match): {len(video_scenes)}")

    # Try with type conversion
    video_scenes_str = alignment_df[alignment_df['video id'].astype(str) == str(test_video_id)]
    print(f"   Scenes found (string match): {len(video_scenes_str)}")

    video_scenes_int = alignment_df[alignment_df['video id'].astype(int) == int(test_video_id)]
    print(f"   Scenes found (int match): {len(video_scenes_int)}")

    if len(video_scenes_int) > 0:
        scene_numbers = video_scenes_int['Scene Number'].values
        print(f"   Scene numbers: {scene_numbers[:10]}")

        # Check if screenshot exists for first scene
        first_scene = scene_numbers[0]
        screenshot_path = os.path.join(
            SCREENSHOTS_DIR,
            str(test_video_id),
            f"{test_video_id}-Scene-{first_scene:03d}-01.jpg"
        )
        print(f"\n5. Checking first scene file:")
        print(f"   Path: {screenshot_path}")
        print(f"   Exists: {os.path.exists(screenshot_path)}")

        if not os.path.exists(screenshot_path):
            # Check what files are actually in that directory
            video_dir = os.path.join(SCREENSHOTS_DIR, str(test_video_id))
            if os.path.exists(video_dir):
                files = os.listdir(video_dir)
                print(f"   Files in directory: {files[:10]}")
            else:
                print(f"   Video directory doesn't exist: {video_dir}")

# Check 5: Keywords
keywords_df = pd.read_csv(KEYWORDS_FILE)
keywords_df.columns = keywords_df.columns.str.strip()

if '_id' in keywords_df.columns:
    video_id_col = '_id'
else:
    video_id_col = 'video_id'

if 'keyword_list[0]' in keywords_df.columns:
    keyword_col = 'keyword_list[0]'
else:
    keyword_col = 'keyword'

print(f"\n6. Keywords:")
print(f"   Video ID column: '{video_id_col}'")
print(f"   Keyword column: '{keyword_col}'")
print(f"   Video ID type in keywords: {type(keywords_df[video_id_col].iloc[0])}")

if len(video_ids_for_training) > 0:
    test_video_id = video_ids_for_training[0]
    keyword_row = keywords_df[keywords_df[video_id_col].astype(str) == str(test_video_id)]
    if len(keyword_row) > 0:
        keyword = keyword_row.iloc[0][keyword_col]
        print(f"   Keyword for video {test_video_id}: '{keyword}'")
    else:
        print(f"   ❌ No keyword found for video {test_video_id}")

print("\n" + "=" * 70)

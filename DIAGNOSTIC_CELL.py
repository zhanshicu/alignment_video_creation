"""
PASTE THIS CELL INTO YOUR NOTEBOOK RIGHT BEFORE THE MASK GENERATION CELL
This will diagnose the type mismatch issue
"""

# ========================================
# DIAGNOSTIC CELL - Paste into notebook
# ========================================

import pandas as pd

print("=" * 70)
print("DIAGNOSTIC: Checking type mismatch issue")
print("=" * 70)

# Load data
alignment_df = pd.read_csv(ALIGNMENT_SCORE_FILE)
alignment_df.columns = alignment_df.columns.str.strip()

# Check 1: What type are video_ids_for_training?
print(f"\n1. video_ids_for_training:")
print(f"   Length: {len(video_ids_for_training)}")
print(f"   First video ID: {video_ids_for_training[0]}")
print(f"   Type: {type(video_ids_for_training[0])}")

# Check 2: What type are video IDs in alignment_df?
print(f"\n2. alignment_df['video id']:")
print(f"   First value: {alignment_df['video id'].iloc[0]}")
print(f"   Type: {type(alignment_df['video id'].iloc[0])}")

# Check 3: Try filtering
test_video_id = video_ids_for_training[0]
print(f"\n3. Testing filter for video: {test_video_id}")

# Try direct comparison (will likely fail)
result1 = alignment_df[alignment_df['video id'] == test_video_id]
print(f"   Direct comparison: {len(result1)} rows")

# Try with int conversion
result2 = alignment_df[alignment_df['video id'] == int(test_video_id)]
print(f"   With int(video_id): {len(result2)} rows")

# Try with str conversion
result3 = alignment_df[alignment_df['video id'].astype(str) == str(test_video_id)]
print(f"   With str comparison: {len(result3)} rows")

print(f"\n4. Conclusion:")
if len(result1) > 0:
    print("   ✓ Direct comparison works - no type issue")
elif len(result2) > 0:
    print("   ❌ TYPE MISMATCH: need to convert video_id to int")
    print("   FIX: Change line to: video_scenes = alignment_df[alignment_df['video id'] == int(video_id)]")
elif len(result3) > 0:
    print("   ❌ TYPE MISMATCH: need to convert alignment column to str")
    print("   FIX: Change line to: video_scenes = alignment_df[alignment_df['video id'].astype(str) == str(video_id)]")

print("=" * 70)

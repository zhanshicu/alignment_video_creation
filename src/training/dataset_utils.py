"""
Helper functions for dataset splitting and validation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def get_valid_video_ids(
    alignment_score_file: str,
    keywords_file: str
) -> List[str]:
    """
    Get list of valid video IDs that have both alignment scores and keywords.

    Uses alignment_score.csv as the source of truth.

    Args:
        alignment_score_file: Path to alignment_score.csv
        keywords_file: Path to keywords.csv

    Returns:
        List of valid video IDs (as strings)
    """
    # Load data
    alignment_df = pd.read_csv(alignment_score_file)
    alignment_df.columns = alignment_df.columns.str.strip()

    keywords_df = pd.read_csv(keywords_file)
    keywords_df.columns = keywords_df.columns.str.strip()

    # Get video IDs from alignment_score.csv (source of truth)
    alignment_video_ids = set(alignment_df['video id'].astype(str).unique())

    # Get video IDs that have valid keywords
    if '_id' in keywords_df.columns:
        video_id_col = '_id'
    elif 'video_id' in keywords_df.columns:
        video_id_col = 'video_id'
    else:
        raise ValueError("Could not find video ID column in keywords.csv")

    if 'keyword_list[0]' in keywords_df.columns:
        keyword_col = 'keyword_list[0]'
    elif 'keyword' in keywords_df.columns:
        keyword_col = 'keyword'
    else:
        raise ValueError("Could not find keyword column in keywords.csv")

    # Filter out videos with missing keywords
    keywords_df_clean = keywords_df[
        keywords_df[keyword_col].notna() & (keywords_df[keyword_col] != '')
    ]
    keyword_video_ids = set(keywords_df_clean[video_id_col].astype(str).unique())

    # Get intersection: videos that have both alignment scores AND keywords
    valid_video_ids = list(alignment_video_ids & keyword_video_ids)

    print(f"Video ID validation:")
    print(f"  Videos in alignment_score.csv: {len(alignment_video_ids)}")
    print(f"  Videos with valid keywords: {len(keyword_video_ids)}")
    print(f"  Valid videos (intersection): {len(valid_video_ids)}")

    return sorted(valid_video_ids)


def split_train_val_videos(
    video_ids: List[str],
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split video IDs into train and validation sets.

    Args:
        video_ids: List of video IDs
        val_ratio: Ratio of videos to use for validation (default: 0.2)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_video_ids, val_video_ids)
    """
    np.random.seed(random_seed)
    video_ids = np.array(video_ids)

    # Shuffle videos
    shuffled_indices = np.random.permutation(len(video_ids))
    shuffled_videos = video_ids[shuffled_indices]

    # Split
    split_idx = int(len(shuffled_videos) * (1 - val_ratio))
    train_videos = shuffled_videos[:split_idx].tolist()
    val_videos = shuffled_videos[split_idx:].tolist()

    print(f"\nDataset split (random seed={random_seed}):")
    print(f"  Training videos: {len(train_videos)} ({len(train_videos)/len(video_ids)*100:.1f}%)")
    print(f"  Validation videos: {len(val_videos)} ({len(val_videos)/len(video_ids)*100:.1f}%)")

    return train_videos, val_videos


def get_dataset_statistics(
    alignment_score_file: str,
    video_ids: List[str]
) -> Dict:
    """
    Get statistics about the dataset.

    Args:
        alignment_score_file: Path to alignment_score.csv
        video_ids: List of video IDs to analyze

    Returns:
        Dictionary with statistics
    """
    # Load data
    alignment_df = pd.read_csv(alignment_score_file)
    alignment_df.columns = alignment_df.columns.str.strip()

    # Filter to specified videos
    video_ids_str = [str(vid) for vid in video_ids]
    df_filtered = alignment_df[
        alignment_df['video id'].astype(str).isin(video_ids_str)
    ]

    stats = {
        'num_videos': len(video_ids),
        'num_scenes': len(df_filtered),
        'scenes_per_video': {
            'mean': df_filtered.groupby('video id').size().mean(),
            'std': df_filtered.groupby('video id').size().std(),
            'min': df_filtered.groupby('video id').size().min(),
            'max': df_filtered.groupby('video id').size().max(),
        },
        'alignment_score': {
            'mean': df_filtered['attention_proportion'].mean(),
            'std': df_filtered['attention_proportion'].std(),
            'min': df_filtered['attention_proportion'].min(),
            'max': df_filtered['attention_proportion'].max(),
        },
        'industries': df_filtered['industry'].value_counts().to_dict() if 'industry' in df_filtered.columns else {},
    }

    return stats


def print_dataset_statistics(
    alignment_score_file: str,
    train_videos: List[str],
    val_videos: List[str]
):
    """
    Print dataset statistics for train and validation sets.

    Args:
        alignment_score_file: Path to alignment_score.csv
        train_videos: List of training video IDs
        val_videos: List of validation video IDs
    """
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    train_stats = get_dataset_statistics(alignment_score_file, train_videos)
    val_stats = get_dataset_statistics(alignment_score_file, val_videos)

    print("\nTRAINING SET:")
    print(f"  Videos: {train_stats['num_videos']}")
    print(f"  Scenes: {train_stats['num_scenes']}")
    print(f"  Scenes per video: {train_stats['scenes_per_video']['mean']:.1f} ± {train_stats['scenes_per_video']['std']:.1f}")
    print(f"  Alignment score: {train_stats['alignment_score']['mean']:.4f} ± {train_stats['alignment_score']['std']:.4f}")

    print("\nVALIDATION SET:")
    print(f"  Videos: {val_stats['num_videos']}")
    print(f"  Scenes: {val_stats['num_scenes']}")
    print(f"  Scenes per video: {val_stats['scenes_per_video']['mean']:.1f} ± {val_stats['scenes_per_video']['std']:.1f}")
    print(f"  Alignment score: {val_stats['alignment_score']['mean']:.4f} ± {val_stats['alignment_score']['std']:.4f}")

    if train_stats['industries']:
        print("\nINDUSTRY DISTRIBUTION (Training):")
        for industry, count in sorted(train_stats['industries'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {industry}: {count} scenes")

    print("="*60 + "\n")

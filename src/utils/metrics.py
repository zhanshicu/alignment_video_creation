"""Metrics for evaluating attention-keyword alignment."""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats


def calculate_alignment_score(
    attention_map: np.ndarray,
    keyword_mask: np.ndarray
) -> float:
    """
    Calculate proportion of attention on keyword region.

    Args:
        attention_map: Attention heatmap (H, W)
        keyword_mask: Binary keyword mask (H, W)

    Returns:
        Alignment score in [0, 1]
    """
    total_attention = np.sum(attention_map)
    if total_attention < 1e-8:
        return 0.0

    keyword_attention = np.sum(attention_map * keyword_mask)
    return keyword_attention / total_attention


def calculate_temporal_consistency(
    heatmaps: List[np.ndarray],
    method: str = "correlation"
) -> float:
    """
    Calculate temporal consistency of heatmaps.

    Args:
        heatmaps: List of heatmaps
        method: Method for computing consistency ('correlation', 'mse')

    Returns:
        Consistency score
    """
    if len(heatmaps) < 2:
        return 1.0

    consistencies = []

    for i in range(len(heatmaps) - 1):
        map1 = heatmaps[i].flatten()
        map2 = heatmaps[i + 1].flatten()

        if method == "correlation":
            corr, _ = stats.pearsonr(map1, map2)
            consistencies.append(corr)
        elif method == "mse":
            mse = np.mean((map1 - map2) ** 2)
            consistencies.append(1.0 / (1.0 + mse))  # Convert to similarity

    return np.mean(consistencies)


def calculate_alignment_metrics(
    attention_maps: List[np.ndarray],
    keyword_maps: List[np.ndarray],
    keyword_masks: List[np.ndarray]
) -> Dict[str, float]:
    """
    Calculate comprehensive alignment metrics.

    Args:
        attention_maps: List of attention heatmaps
        keyword_maps: List of keyword heatmaps
        keyword_masks: List of keyword masks

    Returns:
        Dict of metrics
    """
    # Alignment scores per frame
    alignment_scores = [
        calculate_alignment_score(A, M)
        for A, M in zip(attention_maps, keyword_masks)
    ]

    # Temporal consistency
    attention_consistency = calculate_temporal_consistency(attention_maps)
    keyword_consistency = calculate_temporal_consistency(keyword_maps)

    # Statistics
    metrics = {
        'mean_alignment': np.mean(alignment_scores),
        'std_alignment': np.std(alignment_scores),
        'min_alignment': np.min(alignment_scores),
        'max_alignment': np.max(alignment_scores),
        'median_alignment': np.median(alignment_scores),
        'attention_temporal_consistency': attention_consistency,
        'keyword_temporal_consistency': keyword_consistency,
        'alignment_scores': alignment_scores,
    }

    return metrics


def calculate_spatial_overlap(
    map1: np.ndarray,
    map2: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate spatial overlap between two heatmaps.

    Args:
        map1: First heatmap
        map2: Second heatmap
        threshold: Threshold for binarization

    Returns:
        Dict with IoU, Dice, and other metrics
    """
    # Binarize
    mask1 = (map1 > threshold).astype(float)
    mask2 = (map2 > threshold).astype(float)

    # Intersection and union
    intersection = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2) > 0)

    # Metrics
    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-8)

    # Correlation
    corr, _ = stats.pearsonr(map1.flatten(), map2.flatten())

    return {
        'iou': iou,
        'dice': dice,
        'correlation': corr,
        'intersection': intersection,
        'union': union,
    }


def evaluate_variant_effects(
    baseline_scores: List[float],
    variant_scores: List[float],
    window: Tuple[int, int]
) -> Dict[str, float]:
    """
    Evaluate the effect of a variant compared to baseline.

    Args:
        baseline_scores: Alignment scores for baseline
        variant_scores: Alignment scores for variant
        window: (start, end) frame indices of manipulation window

    Returns:
        Dict of evaluation metrics
    """
    # Overall differences
    mean_diff = np.mean(variant_scores) - np.mean(baseline_scores)
    mean_ratio = np.mean(variant_scores) / (np.mean(baseline_scores) + 1e-8)

    # Window-specific differences
    baseline_window = baseline_scores[window[0]:window[1]]
    variant_window = variant_scores[window[0]:window[1]]

    window_diff = np.mean(variant_window) - np.mean(baseline_window)
    window_ratio = np.mean(variant_window) / (np.mean(baseline_window) + 1e-8)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(baseline_window, variant_window)

    return {
        'mean_difference': mean_diff,
        'mean_ratio': mean_ratio,
        'window_difference': window_diff,
        'window_ratio': window_ratio,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


def compute_attention_distribution(
    attention_map: np.ndarray,
    num_bins: int = 10
) -> np.ndarray:
    """
    Compute histogram of attention distribution.

    Args:
        attention_map: Attention heatmap
        num_bins: Number of histogram bins

    Returns:
        Histogram values
    """
    hist, _ = np.histogram(attention_map.flatten(), bins=num_bins, range=(0, 1))
    return hist / hist.sum()


def calculate_entropy(heatmap: np.ndarray) -> float:
    """
    Calculate entropy of attention distribution.

    Args:
        heatmap: Attention heatmap

    Returns:
        Entropy value
    """
    # Normalize to probability distribution
    heatmap_norm = heatmap / (heatmap.sum() + 1e-8)

    # Compute entropy
    entropy = -np.sum(heatmap_norm * np.log(heatmap_norm + 1e-8))

    return entropy

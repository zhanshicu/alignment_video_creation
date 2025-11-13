"""Visualization utilities for heatmaps and control tensors."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import cv2


def visualize_heatmaps(
    image: np.ndarray,
    attention_map: np.ndarray,
    keyword_map: np.ndarray,
    alignment_map: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    titles: Optional[List[str]] = None
):
    """
    Visualize attention and keyword heatmaps overlaid on image.

    Args:
        image: Original image (H, W, 3) in [0, 255] or [0, 1]
        attention_map: Attention heatmap (H, W)
        keyword_map: Keyword heatmap (H, W)
        alignment_map: Optional alignment map (H, W)
        save_path: Optional path to save figure
        titles: Optional list of subplot titles
    """
    # Normalize image
    if image.max() > 1:
        image = image / 255.0

    # Determine layout
    n_plots = 4 if alignment_map is not None else 3
    fig, axes = plt.subplots(1, n_plots + 1, figsize=(4 * (n_plots + 1), 4))

    if titles is None:
        titles = ["Original", "Attention", "Keyword", "Alignment"]

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    # Attention heatmap overlay
    axes[1].imshow(image)
    axes[1].imshow(attention_map, alpha=0.5, cmap='hot')
    axes[1].set_title(titles[1])
    axes[1].axis('off')

    # Keyword heatmap overlay
    axes[2].imshow(image)
    axes[2].imshow(keyword_map, alpha=0.5, cmap='viridis')
    axes[2].set_title(titles[2])
    axes[2].axis('off')

    # Alignment map overlay
    if alignment_map is not None:
        axes[3].imshow(image)
        axes[3].imshow(alignment_map, alpha=0.5, cmap='plasma')
        axes[3].set_title(titles[3])
        axes[3].axis('off')

        # Standalone heatmaps
        axes[4].imshow(attention_map, cmap='hot')
        axes[4].set_title("Attention (raw)")
        axes[4].axis('off')
    else:
        # Standalone heatmaps
        axes[3].imshow(attention_map, cmap='hot')
        axes[3].set_title("Attention (raw)")
        axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_control_tensor(
    control_tensor: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize all channels of control tensor.

    Args:
        control_tensor: Control tensor (H, W, C) where C=2 or 4
        save_path: Optional path to save figure
    """
    n_channels = control_tensor.shape[-1]

    fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))

    if n_channels == 1:
        axes = [axes]

    channel_names = {
        0: "Keyword Mask (M_t)",
        1: "Alignment Map (S_t)",
        2: "Attention Map (A_t)",
        3: "Keyword Map (K_t)"
    }

    for i in range(n_channels):
        axes[i].imshow(control_tensor[:, :, i], cmap='viridis')
        axes[i].set_title(channel_names.get(i, f"Channel {i}"))
        axes[i].axis('off')
        plt.colorbar(axes[i].images[0], ax=axes[i], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved control tensor visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def create_heatmap_video_overlay(
    frames: List[np.ndarray],
    heatmaps: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    colormap: str = 'hot',
    alpha: float = 0.5
):
    """
    Create video with heatmap overlay.

    Args:
        frames: List of video frames (H, W, 3)
        heatmaps: List of heatmaps (H, W)
        output_path: Output video path
        fps: Frames per second
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
    """
    import os
    from matplotlib import cm

    # Get colormap
    cmap = cm.get_cmap(colormap)

    overlaid_frames = []

    for frame, heatmap in zip(frames, heatmaps):
        # Normalize frame
        if frame.max() > 1:
            frame = frame.astype(np.float32) / 255.0

        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Apply colormap
        heatmap_colored = cmap(heatmap_norm)[:, :, :3]  # RGB only

        # Overlay
        overlaid = (1 - alpha) * frame + alpha * heatmap_colored
        overlaid = np.clip(overlaid * 255, 0, 255).astype(np.uint8)

        overlaid_frames.append(overlaid)

    # Save video
    from .video_utils import VideoSaver
    VideoSaver.save_video(overlaid_frames, output_path, fps)


def plot_temporal_alignment(
    alignment_scores: List[float],
    window_labels: Optional[Dict[str, Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    title: str = "Temporal Alignment Profile"
):
    """
    Plot alignment scores over time.

    Args:
        alignment_scores: List of alignment scores per frame
        window_labels: Optional dict of window names to (start, end) indices
        save_path: Optional path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    # Plot alignment
    plt.plot(alignment_scores, linewidth=2)
    plt.xlabel("Frame Index")
    plt.ylabel("Alignment Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add window labels if provided
    if window_labels:
        colors = ['red', 'green', 'blue', 'orange']
        for i, (name, (start, end)) in enumerate(window_labels.items()):
            color = colors[i % len(colors)]
            plt.axvspan(start, end, alpha=0.2, color=color, label=name)
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved temporal alignment plot to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_variants(
    variants_stats: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Compare alignment profiles across variants.

    Args:
        variants_stats: Dict of variant statistics
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot temporal profiles
    for variant_name, stats in variants_stats.items():
        temporal_profile = stats['temporal_profile']
        ax1.plot(temporal_profile, label=variant_name, alpha=0.7)

    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Alignment Value")
    ax1.set_title("Temporal Alignment Profiles")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot mean alignments
    variant_names = list(variants_stats.keys())
    mean_alignments = [
        stats['mean_controlled_alignment']
        for stats in variants_stats.values()
    ]

    ax2.bar(variant_names, mean_alignments)
    ax2.set_xlabel("Variant")
    ax2.set_ylabel("Mean Alignment")
    ax2.set_title("Mean Alignment by Variant")
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved variant comparison to {save_path}")
    else:
        plt.show()

    plt.close()

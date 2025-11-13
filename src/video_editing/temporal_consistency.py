"""
Temporal Consistency Wrapper

Wrapper for integrating with video editing frameworks like Rerender A Video
for temporal consistency in edited videos.
"""

import os
import numpy as np
import torch
from typing import List, Dict, Optional
import cv2
from tqdm import tqdm


class TemporalConsistencyWrapper:
    """
    Wrapper for ensuring temporal consistency in video editing.

    This is a placeholder that provides the interface for integration
    with SOTA video editing frameworks like Rerender A Video.
    """

    def __init__(
        self,
        method: str = "optical_flow",
        flow_model: Optional[str] = None,
    ):
        """
        Initialize temporal consistency wrapper.

        Args:
            method: Consistency method ('optical_flow', 'rerender', 'ebsynth')
            flow_model: Optional optical flow model name
        """
        self.method = method
        self.flow_model = flow_model

        if method == "optical_flow":
            # Use OpenCV optical flow
            self.flow_calculator = cv2.FarnebackOpticalFlow_create()
        elif method == "rerender":
            # Placeholder for Rerender A Video integration
            print("Note: Rerender A Video integration requires external setup.")
            print("See: https://github.com/williamyang1991/Rerender_A_Video")
        elif method == "ebsynth":
            # Placeholder for EbSynth integration
            print("Note: EbSynth integration requires external setup.")
            print("See: https://ebsynth.com/")

    def compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        Compute optical flow between two frames.

        Args:
            frame1: First frame (H, W, 3)
            frame2: Second frame (H, W, 3)

        Returns:
            Flow field (H, W, 2)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )

        return flow

    def warp_frame(
        self,
        frame: np.ndarray,
        flow: np.ndarray
    ) -> np.ndarray:
        """
        Warp frame according to optical flow.

        Args:
            frame: Frame to warp (H, W, 3)
            flow: Optical flow (H, W, 2)

        Returns:
            Warped frame (H, W, 3)
        """
        h, w = frame.shape[:2]
        flow_map = np.zeros((h, w, 2), dtype=np.float32)

        # Create pixel coordinate grid
        for y in range(h):
            for x in range(w):
                flow_map[y, x, 0] = x + flow[y, x, 0]
                flow_map[y, x, 1] = y + flow[y, x, 1]

        # Warp
        warped = cv2.remap(
            frame, flow_map, None,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return warped

    def propagate_edits(
        self,
        original_frames: List[np.ndarray],
        edited_keyframes: Dict[int, np.ndarray],
        blend_window: int = 5,
    ) -> List[np.ndarray]:
        """
        Propagate edits from keyframes to all frames using optical flow.

        Args:
            original_frames: List of original frames
            edited_keyframes: Dict mapping keyframe index to edited frame
            blend_window: Window size for blending

        Returns:
            List of edited frames with temporal consistency
        """
        num_frames = len(original_frames)
        result_frames = [None] * num_frames

        # Sort keyframe indices
        keyframe_indices = sorted(edited_keyframes.keys())

        print("Propagating edits with optical flow...")

        for i in tqdm(range(num_frames), desc="Temporal propagation"):
            if i in edited_keyframes:
                # This is a keyframe, use directly
                result_frames[i] = edited_keyframes[i]
            else:
                # Find nearest keyframes
                prev_key = None
                next_key = None

                for key in keyframe_indices:
                    if key < i:
                        prev_key = key
                    elif key > i and next_key is None:
                        next_key = key
                        break

                # Interpolate between keyframes
                if prev_key is not None and next_key is not None:
                    # Bilateral interpolation
                    alpha = (i - prev_key) / (next_key - prev_key)

                    # Warp from prev keyframe
                    flow_prev = self.compute_optical_flow(
                        original_frames[prev_key],
                        original_frames[i]
                    )
                    warped_prev = self.warp_frame(edited_keyframes[prev_key], flow_prev)

                    # Warp from next keyframe
                    flow_next = self.compute_optical_flow(
                        original_frames[next_key],
                        original_frames[i]
                    )
                    warped_next = self.warp_frame(edited_keyframes[next_key], flow_next)

                    # Blend
                    result_frames[i] = (
                        (1 - alpha) * warped_prev + alpha * warped_next
                    ).astype(np.uint8)

                elif prev_key is not None:
                    # Only prev keyframe available
                    flow = self.compute_optical_flow(
                        original_frames[prev_key],
                        original_frames[i]
                    )
                    result_frames[i] = self.warp_frame(edited_keyframes[prev_key], flow)

                elif next_key is not None:
                    # Only next keyframe available
                    flow = self.compute_optical_flow(
                        original_frames[next_key],
                        original_frames[i]
                    )
                    result_frames[i] = self.warp_frame(edited_keyframes[next_key], flow)

                else:
                    # No keyframes available, use original
                    result_frames[i] = original_frames[i]

        return result_frames

    def apply_temporal_smoothing(
        self,
        frames: List[np.ndarray],
        window_size: int = 3
    ) -> List[np.ndarray]:
        """
        Apply temporal smoothing to reduce flickering.

        Args:
            frames: List of frames
            window_size: Size of temporal window

        Returns:
            Smoothed frames
        """
        smoothed = []
        half_window = window_size // 2

        for i in range(len(frames)):
            # Get window
            start = max(0, i - half_window)
            end = min(len(frames), i + half_window + 1)

            window = frames[start:end]

            # Average
            smoothed_frame = np.mean(window, axis=0).astype(np.uint8)
            smoothed.append(smoothed_frame)

        return smoothed

    def ensure_consistency(
        self,
        original_frames: List[np.ndarray],
        edited_frames: List[np.ndarray],
        edited_keyframes: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        Ensure temporal consistency in edited video.

        Args:
            original_frames: Original frames
            edited_frames: Edited frames (may have inconsistencies)
            edited_keyframes: Optional keyframes for propagation-based methods

        Returns:
            Temporally consistent edited frames
        """
        if self.method == "optical_flow" and edited_keyframes is not None:
            # Use keyframe propagation
            return self.propagate_edits(original_frames, edited_keyframes)
        elif self.method == "optical_flow":
            # Just apply smoothing
            return self.apply_temporal_smoothing(edited_frames)
        elif self.method == "rerender":
            # Placeholder: would integrate with Rerender A Video here
            print("Warning: Rerender A Video integration not implemented.")
            print("Falling back to temporal smoothing.")
            return self.apply_temporal_smoothing(edited_frames)
        elif self.method == "ebsynth":
            # Placeholder: would integrate with EbSynth here
            print("Warning: EbSynth integration not implemented.")
            print("Falling back to temporal smoothing.")
            return self.apply_temporal_smoothing(edited_frames)
        else:
            return edited_frames

    def save_for_rerender(
        self,
        original_frames: List[np.ndarray],
        edited_keyframes: Dict[int, np.ndarray],
        output_dir: str,
        video_name: str = "video"
    ):
        """
        Save frames in format expected by Rerender A Video.

        Args:
            original_frames: Original frames
            edited_keyframes: Edited keyframes
            output_dir: Output directory
            video_name: Video name prefix
        """
        # Create directories
        original_dir = os.path.join(output_dir, "original_frames")
        edited_dir = os.path.join(output_dir, "edited_keyframes")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(edited_dir, exist_ok=True)

        # Save original frames
        for i, frame in enumerate(original_frames):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(original_dir, f"{video_name}_{i:05d}.png"),
                frame_bgr
            )

        # Save edited keyframes
        for i, frame in edited_keyframes.items():
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(edited_dir, f"{video_name}_{i:05d}.png"),
                frame_bgr
            )

        print(f"Saved frames to {output_dir}")
        print("You can now use Rerender A Video to propagate these edits.")

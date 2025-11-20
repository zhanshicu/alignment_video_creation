"""
Temporal Smoothing Utilities

Ensures edited frames blend smoothly into videos without flickering or artifacts.

Methods:
1. Simple blending: Fast, basic temporal consistency
2. Optical flow warping: Better quality, uses motion information
3. Frame interpolation: Highest quality, smooth transitions
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Set


class TemporalSmoother:
    """
    Apply temporal smoothing to edited video frames.
    """

    def __init__(self, method: str = "simple_blend"):
        """
        Initialize smoother.

        Args:
            method: "simple_blend", "optical_flow", or "interpolation"
        """
        self.method = method

    def smooth(
        self,
        original_frames: List[Image.Image],
        edited_frames: List[Image.Image],
        edited_indices: Set[int],
        blend_strength: float = 0.7,
    ) -> List[Image.Image]:
        """
        Apply temporal smoothing.

        Args:
            original_frames: List of original frames
            edited_frames: List of frames (original + edited mixed)
            edited_indices: Set of indices that were edited
            blend_strength: 0.0 = all original, 1.0 = all edited

        Returns:
            List of smoothed frames
        """
        if self.method == "simple_blend":
            return self._simple_blend(
                original_frames, edited_frames, edited_indices, blend_strength
            )

        elif self.method == "optical_flow":
            return self._optical_flow_smooth(
                original_frames, edited_frames, edited_indices, blend_strength
            )

        elif self.method == "interpolation":
            return self._interpolation_smooth(
                original_frames, edited_frames, edited_indices
            )

        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")

    def _simple_blend(
        self,
        original_frames: List[Image.Image],
        edited_frames: List[Image.Image],
        edited_indices: Set[int],
        blend_strength: float,
    ) -> List[Image.Image]:
        """
        Simple weighted blending between original and edited frames.

        Applies gradual transitions at boundaries.
        """
        smoothed = []
        num_frames = len(original_frames)

        # Convert edited_indices to sorted list for easier processing
        edited_list = sorted(list(edited_indices))

        for i in range(num_frames):
            original = np.array(original_frames[i]).astype(np.float32)
            edited = np.array(edited_frames[i]).astype(np.float32)

            if i in edited_indices:
                # This frame was edited - apply blend
                # Reduce blend near boundaries for smooth transitions
                alpha = self._get_boundary_alpha(i, edited_list, num_frames, blend_strength)

                blended = alpha * edited + (1 - alpha) * original
                smoothed_frame = Image.fromarray(blended.astype(np.uint8))

            else:
                # Frame wasn't edited - use original
                smoothed_frame = original_frames[i]

            smoothed.append(smoothed_frame)

        return smoothed

    def _get_boundary_alpha(
        self,
        idx: int,
        edited_list: List[int],
        num_frames: int,
        base_strength: float,
    ) -> float:
        """
        Compute blend alpha with reduced strength near boundaries.

        Creates smooth transitions at the start/end of edited segments.
        """
        # Find position in edited segment
        position = edited_list.index(idx)

        # Check if near start of segment (first few frames)
        if position < len(edited_list) and position > 0:
            if edited_list[position] - edited_list[position - 1] > 1:
                # Start of new segment
                return base_strength * 0.5  # Reduce strength

        # Check if near end of segment
        if position < len(edited_list) - 1:
            if edited_list[position + 1] - edited_list[position] > 1:
                # End of segment
                return base_strength * 0.5

        return base_strength

    def _optical_flow_smooth(
        self,
        original_frames: List[Image.Image],
        edited_frames: List[Image.Image],
        edited_indices: Set[int],
        blend_strength: float,
    ) -> List[Image.Image]:
        """
        Use optical flow to warp edited frames for temporal consistency.

        Better quality than simple blending - uses motion information.
        """
        smoothed = list(original_frames)  # Start with originals
        edited_list = sorted(list(edited_indices))

        for i in edited_list:
            edited = np.array(edited_frames[i])
            original = np.array(original_frames[i])

            # Warp based on motion from previous frame
            if i > 0:
                prev_frame = np.array(smoothed[i - 1])

                # Compute optical flow
                flow = self._compute_optical_flow(prev_frame, original)

                # Warp edited frame according to flow
                warped = self._warp_frame(edited, flow)

                # Blend warped with original
                blended = blend_strength * warped + (1 - blend_strength) * original
                smoothed[i] = Image.fromarray(blended.astype(np.uint8))

            else:
                # First frame - just blend
                blended = blend_strength * edited + (1 - blend_strength) * original
                smoothed[i] = Image.fromarray(blended.astype(np.uint8))

        return smoothed

    def _compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """Compute dense optical flow between two frames."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        return flow

    def _warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp frame according to optical flow."""
        h, w = flow.shape[:2]

        # Create mesh grid
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[:, :, 0] = np.arange(w)
        flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]

        # Add flow
        flow_map += flow

        # Remap
        warped = cv2.remap(
            frame,
            flow_map,
            None,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return warped

    def _interpolation_smooth(
        self,
        original_frames: List[Image.Image],
        edited_frames: List[Image.Image],
        edited_indices: Set[int],
    ) -> List[Image.Image]:
        """
        Create smooth transitions using frame interpolation.

        Generates intermediate frames between original and edited.
        """
        smoothed = list(original_frames)
        edited_list = sorted(list(edited_indices))

        transition_length = 3  # Number of frames for transition

        for i in edited_list:
            edited = np.array(edited_frames[i])

            # Create transition from previous frame
            if i >= transition_length:
                prev_frame = np.array(smoothed[i - transition_length])

                # Linearly interpolate
                for j in range(transition_length):
                    alpha = (j + 1) / (transition_length + 1)
                    interpolated = (1 - alpha) * prev_frame + alpha * edited
                    smoothed[i - transition_length + j] = Image.fromarray(
                        interpolated.astype(np.uint8)
                    )

            # Set final edited frame
            smoothed[i] = Image.fromarray(edited)

        return smoothed


def apply_temporal_consistency_filter(
    frames: List[Image.Image],
    window_size: int = 3,
) -> List[Image.Image]:
    """
    Apply temporal median filter to reduce flickering.

    Args:
        frames: List of frames
        window_size: Size of temporal window (odd number)

    Returns:
        Filtered frames
    """
    num_frames = len(frames)
    filtered = []

    # Convert to numpy
    frames_np = np.stack([np.array(f) for f in frames], axis=0)

    for i in range(num_frames):
        # Define window
        start = max(0, i - window_size // 2)
        end = min(num_frames, i + window_size // 2 + 1)

        # Compute temporal median
        window = frames_np[start:end]
        median = np.median(window, axis=0).astype(np.uint8)

        filtered.append(Image.fromarray(median))

    return filtered


def create_crossfade(
    frames: List[Image.Image],
    crossfade_duration: int = 5,
) -> List[Image.Image]:
    """
    Add crossfade transitions between scene cuts.

    Args:
        frames: List of frames
        crossfade_duration: Number of frames for crossfade

    Returns:
        Frames with crossfade transitions
    """
    # This is a simplified version - in practice, you'd detect scene cuts first
    result = list(frames)

    # Detect scene cuts (simplified: assume every N frames)
    # In practice, use proper scene detection
    scene_length = 30  # Assume scenes are ~30 frames
    num_frames = len(frames)

    for scene_start in range(scene_length, num_frames, scene_length):
        # Apply crossfade at scene boundaries
        for j in range(crossfade_duration):
            if scene_start - crossfade_duration + j < 0:
                continue
            if scene_start + j >= num_frames:
                break

            prev_frame = np.array(frames[scene_start - 1])
            next_frame = np.array(frames[scene_start])

            alpha = j / crossfade_duration
            blended = (1 - alpha) * prev_frame + alpha * next_frame

            result[scene_start - crossfade_duration + j] = Image.fromarray(
                blended.astype(np.uint8)
            )

    return result

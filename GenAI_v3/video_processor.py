"""
Video Processing Utilities

Handles video loading, scene detection, frame replacement, and video export.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd


class VideoProcessor:
    """
    Process videos: load, identify scenes, replace frames, export.
    """

    def __init__(self, video_dir: str = "../data/data_tiktok"):
        """
        Initialize video processor.

        Args:
            video_dir: Directory containing original videos ({video_id}.mp4)
        """
        self.video_dir = Path(video_dir)

    def load_video(self, video_id: str) -> Tuple[List[np.ndarray], float]:
        """
        Load video as list of frames.

        Args:
            video_id: Video identifier

        Returns:
            (frames, fps) - List of frames as numpy arrays, frames per second
        """
        video_path = self.video_dir / f"{video_id}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        print(f"✓ Loaded video {video_id}: {len(frames)} frames @ {fps:.2f} fps")
        return frames, fps

    def get_scene_frame_range(
        self,
        video_id: str,
        scene_number: int,
        valid_scenes_df: pd.DataFrame,
        fps: float,
    ) -> Tuple[int, int]:
        """
        Get frame range for a specific scene in the video.

        Args:
            video_id: Video identifier
            scene_number: Scene number (1-indexed)
            valid_scenes_df: DataFrame with scene information
            fps: Video frames per second

        Returns:
            (start_frame, end_frame) - Frame indices for this scene
        """
        # Get all scenes for this video, sorted by scene number
        video_scenes = valid_scenes_df[
            valid_scenes_df['video_id'] == video_id
        ].sort_values('scene_number')

        if len(video_scenes) == 0:
            raise ValueError(f"No scenes found for video {video_id}")

        # Find the target scene
        scene_row = video_scenes[video_scenes['scene_number'] == scene_number]

        if len(scene_row) == 0:
            raise ValueError(
                f"Scene {scene_number} not found for video {video_id}. "
                f"Available scenes: {video_scenes['scene_number'].tolist()}"
            )

        # Estimate frame range based on scene position
        # Assumption: scenes are evenly distributed in the video
        # For better accuracy, you'd use actual scene cut detection

        total_scenes = len(video_scenes)
        scene_idx = scene_number - 1  # 0-indexed

        # Get total frames from video
        video_path = self.video_dir / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Estimate frames per scene
        frames_per_scene = total_frames // total_scenes

        # Calculate range
        start_frame = scene_idx * frames_per_scene
        end_frame = start_frame + frames_per_scene

        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        print(f"Scene {scene_number} estimated frame range: {start_frame}-{end_frame}")
        print(f"  (Total video frames: {total_frames}, scenes: {total_scenes})")

        return start_frame, end_frame

    def detect_scene_cuts(
        self,
        frames: List[np.ndarray],
        threshold: float = 30.0,
    ) -> List[int]:
        """
        Detect scene cuts in video using frame difference.

        Args:
            frames: List of video frames
            threshold: Scene cut threshold (higher = fewer cuts)

        Returns:
            List of frame indices where scene cuts occur
        """
        scene_cuts = [0]  # First frame is always a scene cut

        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        for i in range(1, len(frames)):
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

            # Compute frame difference
            diff = cv2.absdiff(curr_frame, prev_frame)
            mean_diff = np.mean(diff)

            if mean_diff > threshold:
                scene_cuts.append(i)

            prev_frame = curr_frame

        print(f"✓ Detected {len(scene_cuts)} scene cuts")
        return scene_cuts

    def get_precise_scene_range(
        self,
        frames: List[np.ndarray],
        scene_number: int,
    ) -> Tuple[int, int]:
        """
        Get precise frame range using scene cut detection.

        Args:
            frames: List of video frames
            scene_number: Scene number (1-indexed)

        Returns:
            (start_frame, end_frame) - Exact frame indices for this scene
        """
        # Detect scene cuts
        scene_cuts = self.detect_scene_cuts(frames)

        if scene_number > len(scene_cuts):
            raise ValueError(
                f"Scene {scene_number} exceeds detected scenes ({len(scene_cuts)})"
            )

        # Get start frame
        start_frame = scene_cuts[scene_number - 1]

        # Get end frame
        if scene_number < len(scene_cuts):
            end_frame = scene_cuts[scene_number]
        else:
            end_frame = len(frames)

        print(f"Scene {scene_number} precise range: {start_frame}-{end_frame}")
        return start_frame, end_frame

    def replace_scene_frames(
        self,
        video_frames: List[np.ndarray],
        start_frame: int,
        end_frame: int,
        replacement_frame: Image.Image,
        blend_frames: int = 5,
    ) -> List[np.ndarray]:
        """
        Replace frames in a video with a manipulated scene.

        Args:
            video_frames: Original video frames
            start_frame: Start frame index
            end_frame: End frame index
            replacement_frame: Manipulated scene (PIL Image)
            blend_frames: Number of frames for smooth blending at boundaries

        Returns:
            Video frames with replaced scene
        """
        edited_frames = video_frames.copy()

        # Convert replacement to numpy
        replacement_np = np.array(replacement_frame)

        # Resize replacement to match video frame size
        h, w = video_frames[0].shape[:2]
        replacement_resized = cv2.resize(
            replacement_np,
            (w, h),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Replace frames with blending
        for i in range(start_frame, end_frame):
            # Compute blend alpha
            if i < start_frame + blend_frames:
                # Blend in
                alpha = (i - start_frame) / blend_frames
            elif i >= end_frame - blend_frames:
                # Blend out
                alpha = (end_frame - i) / blend_frames
            else:
                # Full replacement
                alpha = 1.0

            # Blend
            original = video_frames[i].astype(np.float32)
            replacement = replacement_resized.astype(np.float32)

            blended = alpha * replacement + (1 - alpha) * original
            edited_frames[i] = blended.astype(np.uint8)

        print(f"✓ Replaced frames {start_frame}-{end_frame} with {blend_frames}-frame blending")

        return edited_frames

    def export_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v',
    ):
        """
        Export frames as video file.

        Args:
            frames: List of video frames (RGB numpy arrays)
            output_path: Output video path
            fps: Frames per second
            codec: Video codec ('mp4v', 'avc1', etc.)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get frame dimensions
        h, w = frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {output_path}")

        # Write frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

        print(f"✓ Video exported: {output_path}")
        print(f"  Frames: {len(frames)}, FPS: {fps}, Resolution: {w}x{h}")

    def extract_scene_frame(
        self,
        frames: List[np.ndarray],
        start_frame: int,
        end_frame: int,
        method: str = "middle",
    ) -> np.ndarray:
        """
        Extract a representative frame from a scene.

        Args:
            frames: Video frames
            start_frame: Scene start frame
            end_frame: Scene end frame
            method: "middle", "first", "last", or "median"

        Returns:
            Representative frame as numpy array
        """
        scene_frames = frames[start_frame:end_frame]

        if method == "middle":
            idx = len(scene_frames) // 2
            return scene_frames[idx]

        elif method == "first":
            return scene_frames[0]

        elif method == "last":
            return scene_frames[-1]

        elif method == "median":
            # Compute temporal median to reduce noise
            scene_stack = np.stack(scene_frames, axis=0)
            median_frame = np.median(scene_stack, axis=0).astype(np.uint8)
            return median_frame

        else:
            raise ValueError(f"Unknown method: {method}")


def verify_scene_alignment(
    scene_image_path: str,
    video_frames: List[np.ndarray],
    start_frame: int,
    end_frame: int,
) -> bool:
    """
    Verify that a scene image matches the video frames.

    Args:
        scene_image_path: Path to scene image file
        video_frames: Video frames
        start_frame: Expected start frame
        end_frame: Expected end frame

    Returns:
        True if scene matches, False otherwise
    """
    # Load scene image
    scene_img = Image.open(scene_image_path).convert('RGB')
    scene_np = np.array(scene_img)

    # Get middle frame from video
    mid_frame = (start_frame + end_frame) // 2
    video_frame = video_frames[mid_frame]

    # Resize to same size
    h, w = video_frame.shape[:2]
    scene_resized = cv2.resize(scene_np, (w, h))

    # Compute similarity (MSE)
    mse = np.mean((scene_resized.astype(float) - video_frame.astype(float)) ** 2)

    # Threshold (you may need to tune this)
    threshold = 1000.0

    if mse < threshold:
        print(f"✓ Scene verified (MSE: {mse:.2f})")
        return True
    else:
        print(f"⚠ Scene mismatch (MSE: {mse:.2f}, threshold: {threshold})")
        return False

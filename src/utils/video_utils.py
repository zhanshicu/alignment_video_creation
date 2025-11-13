"""Video loading and saving utilities."""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm


class VideoLoader:
    """Utility for loading videos and extracting frames."""

    @staticmethod
    def load_video_frames(
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
        grayscale: bool = False
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Load frames from a video file.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index (None = until end)
            resize: Optional (width, height) for resizing
            grayscale: Whether to convert to grayscale

        Returns:
            Tuple of (frames list, video_info dict)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if end_frame is None:
            end_frame = total_frames

        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'start_frame': start_frame,
            'end_frame': end_frame,
        }

        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for i in tqdm(range(start_frame, end_frame), desc="Loading frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            if not grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize if needed
            if resize is not None:
                frame = cv2.resize(frame, resize)

            frames.append(frame)

        cap.release()

        return frames, video_info

    @staticmethod
    def extract_scenes(
        video_path: str,
        output_dir: str,
        scene_detector: str = "content",
        threshold: float = 30.0
    ) -> List[Tuple[int, int]]:
        """
        Extract scene boundaries from video.

        Args:
            video_path: Path to video
            output_dir: Output directory for scene frames
            scene_detector: Type of scene detector
            threshold: Detection threshold

        Returns:
            List of (start_frame, end_frame) tuples for each scene
        """
        try:
            from scenedetect import detect, ContentDetector, AdaptiveDetector
        except ImportError:
            print("Warning: scenedetect not installed. Treating as single scene.")
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return [(0, total_frames)]

        # Detect scenes
        if scene_detector == "content":
            detector = ContentDetector(threshold=threshold)
        else:
            detector = AdaptiveDetector()

        scene_list = detect(video_path, detector)

        # Convert to frame indices
        scenes = []
        for scene in scene_list:
            start = scene[0].get_frames()
            end = scene[1].get_frames()
            scenes.append((start, end))

        return scenes


class VideoSaver:
    """Utility for saving videos."""

    @staticmethod
    def save_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v',
        quality: Optional[int] = None
    ):
        """
        Save frames as video file.

        Args:
            frames: List of frames (H, W, 3) in RGB
            output_path: Output video path
            fps: Frames per second
            codec: Video codec
            quality: Optional quality setting
        """
        if len(frames) == 0:
            raise ValueError("No frames to save")

        # Get dimensions from first frame
        h, w = frames[0].shape[:2]

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Write frames
        for frame in tqdm(frames, desc="Saving video"):
            # Convert RGB to BGR
            if frame.ndim == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            out.write(frame_bgr)

        out.release()
        print(f"Saved video to {output_path}")

    @staticmethod
    def save_frames_as_images(
        frames: List[np.ndarray],
        output_dir: str,
        prefix: str = "frame",
        extension: str = "png"
    ):
        """
        Save frames as individual images.

        Args:
            frames: List of frames
            output_dir: Output directory
            prefix: Filename prefix
            extension: Image format
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, frame in enumerate(tqdm(frames, desc="Saving frames")):
            filename = f"{prefix}_{i:05d}.{extension}"
            filepath = os.path.join(output_dir, filename)

            if frame.ndim == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            cv2.imwrite(filepath, frame_bgr)

        print(f"Saved {len(frames)} frames to {output_dir}")

    @staticmethod
    def create_side_by_side_video(
        original_frames: List[np.ndarray],
        edited_frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        labels: Optional[Tuple[str, str]] = None
    ):
        """
        Create side-by-side comparison video.

        Args:
            original_frames: Original frames
            edited_frames: Edited frames
            output_path: Output path
            fps: FPS
            labels: Optional (left_label, right_label)
        """
        assert len(original_frames) == len(edited_frames)

        combined_frames = []

        for orig, edit in zip(original_frames, edited_frames):
            # Ensure same size
            if orig.shape != edit.shape:
                edit = cv2.resize(edit, (orig.shape[1], orig.shape[0]))

            # Add labels if provided
            if labels is not None:
                orig = orig.copy()
                edit = edit.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(orig, labels[0], (10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(edit, labels[1], (10, 30), font, 1, (255, 255, 255), 2)

            # Concatenate horizontally
            combined = np.hstack([orig, edit])
            combined_frames.append(combined)

        # Save
        VideoSaver.save_video(combined_frames, output_path, fps)

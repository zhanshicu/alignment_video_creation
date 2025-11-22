"""
Scene Replacer - Replace scenes with Veo-generated video

After you download the Veo output, this module handles:
1. Loading the Veo-generated video
2. Detecting the exact scene boundaries in original video
3. Seamlessly replacing the scene with temporal blending
4. Exporting the final video

Uses PySceneDetect for accurate scene detection.
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ReplacementResult:
    """Result of scene replacement."""
    output_video_path: str
    output_frame_path: str
    scene_start: int
    scene_end: int
    frames_replaced: int
    blend_frames: int


class SceneReplacer:
    """
    Replace a specific scene in a video with Veo-generated content.

    Handles:
    - Scene boundary detection (PySceneDetect)
    - Frame rate matching
    - Temporal blending at boundaries
    - Seamless video export
    """

    def __init__(
        self,
        video_dir: str = "data/data_tiktok",
        veo_inputs_dir: str = "outputs/genai_v4/veo_inputs",
        output_dir: str = "outputs/genai_v4/final_videos",
    ):
        """
        Initialize scene replacer.

        Args:
            video_dir: Directory containing original videos
            veo_inputs_dir: Directory containing Veo inputs and outputs
            output_dir: Directory for final output videos
        """
        self.video_dir = Path(video_dir)
        self.veo_inputs_dir = Path(veo_inputs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def replace_scene(
        self,
        video_id: str,
        scene_index: int,
        action: str,
        veo_video_path: Optional[str] = None,
        blend_frames: int = 5,
        match_colors: bool = True,
    ) -> ReplacementResult:
        """
        Replace a scene with Veo-generated video.

        Args:
            video_id: Video identifier
            scene_index: Scene number (1-indexed)
            action: "increase" or "decrease" (for finding the right Veo output)
            veo_video_path: Path to Veo output (auto-detected if None)
            blend_frames: Number of frames for blending at boundaries
            match_colors: Apply color matching for seamless integration

        Returns:
            ReplacementResult with output paths and stats
        """
        print(f"\n{'='*70}")
        print(f"  SCENE REPLACEMENT")
        print(f"{'='*70}")
        print(f"Video: {video_id}")
        print(f"Scene: {scene_index}")
        print(f"Action: {action}")
        print(f"{'='*70}\n")

        # Step 1: Load original video
        print("[1/6] Loading original video...")
        original_frames, fps = self._load_video(video_id)
        print(f"  ✓ Loaded {len(original_frames)} frames @ {fps:.1f} fps")

        # Step 2: Detect scenes
        print("\n[2/6] Detecting scene boundaries...")
        scene_list = self._detect_scenes(self.video_dir / f"{video_id}.mp4")
        print(f"  ✓ Found {len(scene_list)} scenes")

        if scene_index < 1 or scene_index > len(scene_list):
            raise ValueError(f"Scene {scene_index} not found. Video has {len(scene_list)} scenes.")

        start_frame, end_frame = scene_list[scene_index - 1]
        num_scene_frames = end_frame - start_frame
        print(f"  Scene {scene_index}: frames {start_frame}-{end_frame} ({num_scene_frames} frames)")

        # Step 3: Load Veo output
        print("\n[3/6] Loading Veo-generated video...")
        if veo_video_path is None:
            veo_video_path = self._find_veo_output(video_id, scene_index, action)

        veo_frames = self._load_veo_video(veo_video_path, num_scene_frames)
        print(f"  ✓ Loaded {len(veo_frames)} Veo frames")

        # Step 4: Match frame sizes
        print("\n[4/6] Matching frame dimensions...")
        h, w = original_frames[0].shape[:2]
        veo_frames = self._resize_frames(veo_frames, w, h)
        print(f"  ✓ Resized to {w}x{h}")

        # Step 5: Color matching (optional)
        if match_colors:
            print("\n[5/6] Applying color matching...")
            veo_frames = self._match_colors(
                veo_frames,
                original_frames[start_frame:end_frame],
            )
            print("  ✓ Colors matched")
        else:
            print("\n[5/6] Skipping color matching...")

        # Step 6: Replace scene with blending
        print("\n[6/6] Replacing scene with temporal blending...")
        final_frames = self._replace_with_blending(
            original_frames=original_frames,
            replacement_frames=veo_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            blend_frames=blend_frames,
        )
        print(f"  ✓ Replaced frames {start_frame}-{end_frame}")

        # Export
        output_video_path = self.output_dir / f"{video_id}_scene{scene_index}_{action}_final.mp4"
        self._export_video(final_frames, output_video_path, fps)

        # Save sample frame
        output_frame_path = self.output_dir / f"{video_id}_scene{scene_index}_{action}_sample.png"
        mid_idx = start_frame + num_scene_frames // 2
        Image.fromarray(final_frames[mid_idx]).save(output_frame_path)
        print(f"  ✓ Saved sample: {output_frame_path}")

        result = ReplacementResult(
            output_video_path=str(output_video_path),
            output_frame_path=str(output_frame_path),
            scene_start=start_frame,
            scene_end=end_frame,
            frames_replaced=num_scene_frames,
            blend_frames=blend_frames,
        )

        print(f"\n{'='*70}")
        print(f"  ✓ SUCCESS!")
        print(f"{'='*70}")
        print(f"Output video: {result.output_video_path}")
        print(f"Sample frame: {result.output_frame_path}")
        print(f"Frames replaced: {result.frames_replaced}")
        print(f"{'='*70}\n")

        return result

    def _find_veo_output(self, video_id: str, scene_index: int, action: str) -> str:
        """Find Veo output video in expected location."""
        expected_dir = self.veo_inputs_dir / f"{video_id}_scene{scene_index}_{action}"
        expected_path = expected_dir / "veo_output.mp4"

        if expected_path.exists():
            return str(expected_path)

        # Try other common names
        for name in ["veo_output.mp4", "output.mp4", "generated.mp4", "video.mp4"]:
            path = expected_dir / name
            if path.exists():
                return str(path)

        # List available files
        if expected_dir.exists():
            files = list(expected_dir.glob("*.mp4"))
            if files:
                print(f"  Found video files: {[f.name for f in files]}")
                return str(files[0])

        raise FileNotFoundError(
            f"Veo output not found. Expected at: {expected_path}\n"
            f"Please download from Veo and save as: {expected_path}"
        )

    def _load_video(self, video_id: str) -> Tuple[List[np.ndarray], float]:
        """Load video frames."""
        video_path = self.video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames, fps

    def _load_veo_video(self, path: str, target_frames: int) -> List[np.ndarray]:
        """Load Veo-generated video and resample to target frame count."""
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from: {path}")

        # Resample to match target frame count
        if len(frames) != target_frames:
            print(f"    Resampling {len(frames)} → {target_frames} frames")
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]

        return frames

    def _detect_scenes(self, video_path: Path) -> List[Tuple[int, int]]:
        """Detect scene boundaries using PySceneDetect."""
        try:
            from scenedetect import detect, AdaptiveDetector
        except ImportError:
            raise ImportError("PySceneDetect required: pip install scenedetect[opencv]")

        scene_list = detect(
            str(video_path),
            AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=15),
        )

        frame_ranges = []
        for scene in scene_list:
            start = scene[0].get_frames()
            end = scene[1].get_frames()
            frame_ranges.append((start, end))

        if not frame_ranges:
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_ranges = [(0, total)]

        return frame_ranges

    def _resize_frames(
        self,
        frames: List[np.ndarray],
        target_w: int,
        target_h: int,
    ) -> List[np.ndarray]:
        """Resize frames to target dimensions."""
        resized = []
        for frame in frames:
            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            resized.append(frame)
        return resized

    def _match_colors(
        self,
        source_frames: List[np.ndarray],
        target_frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Apply color matching to make source match target's color distribution.

        Uses histogram matching on a per-channel basis.
        """
        # Get reference from middle of target
        ref_frame = target_frames[len(target_frames) // 2]

        matched = []
        for frame in source_frames:
            matched_frame = self._histogram_match(frame, ref_frame)
            matched.append(matched_frame)

        return matched

    def _histogram_match(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Match histogram of source to reference."""
        result = np.zeros_like(source)

        for channel in range(3):
            src_hist, _ = np.histogram(source[:, :, channel].flatten(), 256, [0, 256])
            ref_hist, _ = np.histogram(reference[:, :, channel].flatten(), 256, [0, 256])

            src_cdf = src_hist.cumsum()
            src_cdf = src_cdf / src_cdf[-1]

            ref_cdf = ref_hist.cumsum()
            ref_cdf = ref_cdf / ref_cdf[-1]

            # Create lookup table
            lookup = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                j = np.searchsorted(ref_cdf, src_cdf[i])
                lookup[i] = min(j, 255)

            result[:, :, channel] = lookup[source[:, :, channel]]

        return result

    def _replace_with_blending(
        self,
        original_frames: List[np.ndarray],
        replacement_frames: List[np.ndarray],
        start_frame: int,
        end_frame: int,
        blend_frames: int,
    ) -> List[np.ndarray]:
        """Replace scene frames with temporal blending at boundaries."""
        result = original_frames.copy()
        num_replacement = len(replacement_frames)

        for i, replace_idx in enumerate(range(start_frame, end_frame)):
            if i >= num_replacement:
                break

            # Calculate blend factor at boundaries
            if i < blend_frames:
                # Blend in
                alpha = i / blend_frames
            elif i >= num_replacement - blend_frames:
                # Blend out
                alpha = (num_replacement - i - 1) / blend_frames
            else:
                # Full replacement
                alpha = 1.0

            alpha = max(0.0, min(1.0, alpha))

            # Blend
            blended = (
                alpha * replacement_frames[i].astype(float) +
                (1 - alpha) * original_frames[replace_idx].astype(float)
            ).astype(np.uint8)

            result[replace_idx] = blended

        return result

    def _export_video(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: float,
    ):
        """Export frames as video."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]

        # Try codecs
        codecs = [('avc1', '.mp4'), ('mp4v', '.mp4'), ('XVID', '.avi')]

        for codec, ext in codecs:
            try:
                out_path = output_path.with_suffix(ext)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

                if writer.isOpened():
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                    print(f"  ✓ Exported: {out_path}")
                    return
            except:
                continue

        raise RuntimeError("Failed to export video")

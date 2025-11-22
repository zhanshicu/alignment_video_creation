"""
Scene Manipulator - Consistent Video Background Manipulation

Manipulate ENTIRE SCENES with CONSISTENT backgrounds.
No more frame-by-frame inconsistency!

Key Approach:
- Generate ONE background, apply to ALL frames
- Uses video-native APIs when available (Google Veo, Runway)
- Falls back to consistent local SDXL (single generation)
- PySceneDetect for real scene boundaries

Supported backends:
- Google Veo (via Vertex AI) - Best for Colab
- Runway Gen-3 (via API)
- Consistent SDXL (local fallback)
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Literal, Optional, List, Tuple
import cv2
from dataclasses import dataclass


@dataclass
class ManipulationResult:
    """Result of scene manipulation."""
    video_path: str
    frame_path: str
    scene_start: int
    scene_end: int
    frames_manipulated: int
    action: str


class SceneManipulator:
    """
    Manipulate ENTIRE SCENES with CONSISTENT backgrounds.

    Uses video-native APIs (Google Veo, Runway) or generates ONE
    background and applies it to all frames - no inconsistency!

    Usage:
        manipulator = SceneManipulator()

        result = manipulator.manipulate(
            video_id="123456",
            scene_index=6,
            action="increase",
        )

        print(result.video_path)  # Manipulated video
        print(result.frame_path)  # Sample frame
    """

    def __init__(
        self,
        valid_scenes_file: str = "data/valid_scenes.csv",
        video_dir: str = "data/data_tiktok",
        output_dir: str = "outputs/genai_v3",
        device: str = "cuda",
        auto_detect_product: bool = True,
        backend: Literal["auto", "google", "runway", "consistent"] = "auto",
        google_project_id: Optional[str] = None,
        runway_api_key: Optional[str] = None,
    ):
        """
        Initialize scene manipulator.

        Args:
            valid_scenes_file: Path to valid_scenes.csv
            video_dir: Directory containing videos ({video_id}.mp4)
            output_dir: Directory for output videos and frames
            device: "cuda" or "cpu"
            auto_detect_product: Auto-detect product using SAM + DINO
            backend:
                - "auto": Try Google first, fall back to consistent
                - "google": Google Veo/Imagen (best for Colab)
                - "runway": Runway Gen-3 API
                - "consistent": Local SDXL (single bg generation)
            google_project_id: Google Cloud project ID
            runway_api_key: Runway API key
        """
        self.valid_scenes_file = Path(valid_scenes_file)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.frames_output_dir = self.output_dir / "frames"
        self.videos_output_dir = self.output_dir / "videos"
        self.frames_output_dir.mkdir(parents=True, exist_ok=True)
        self.videos_output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.auto_detect_product = auto_detect_product

        # Load valid scenes (optional)
        if valid_scenes_file and Path(valid_scenes_file).exists():
            self.scenes_df = pd.read_csv(valid_scenes_file)
            print(f"✓ Loaded {len(self.scenes_df)} valid scenes")
        else:
            self.scenes_df = None
            print("✓ Running without valid_scenes.csv")

        # Initialize video backend
        print(f"\nInitializing video backend ({backend})...")
        self._init_backend(backend, device, google_project_id, runway_api_key)

        # Initialize product detector
        self.product_detector = None
        if auto_detect_product:
            print("Loading product detector (SAM + DINO)...")
            try:
                from .product_detector import MainProductDetector
                self.product_detector = MainProductDetector(device=device)
                print("✓ Product detector loaded")
            except Exception as e:
                print(f"⚠ Product detector unavailable: {e}")
                print("  Will use saliency detection fallback")

    def _init_backend(
        self,
        backend: str,
        device: str,
        google_project_id: Optional[str],
        runway_api_key: Optional[str],
    ):
        """Initialize the video manipulation backend."""
        from .video_backends import get_backend

        kwargs = {"device": device}
        if google_project_id:
            kwargs["project_id"] = google_project_id
        if runway_api_key:
            kwargs["api_key"] = runway_api_key

        self.backend = get_backend(backend, **kwargs)
        print(f"✓ Backend initialized: {type(self.backend).__name__}")

    def manipulate(
        self,
        video_id: str,
        scene_index: int,
        action: Literal["increase", "decrease"],
        output_path: Optional[str] = None,
        use_keyword_mask: bool = False,
    ) -> ManipulationResult:
        """
        Manipulate an ENTIRE SCENE with CONSISTENT background.

        Args:
            video_id: Video identifier
            scene_index: Scene number (1-indexed)
            action:
                - "increase": Plain gray background → attention on product
                - "decrease": Vibrant background → attention diverts
            output_path: Custom output path (auto-generated if None)
            use_keyword_mask: Use keyword mask instead of auto-detection

        Returns:
            ManipulationResult with paths to video and frame outputs
        """
        print(f"\n{'='*70}")
        print(f"  CONSISTENT SCENE MANIPULATION")
        print(f"{'='*70}")
        print(f"Video: {video_id}")
        print(f"Scene: {scene_index}")
        print(f"Action: {action.upper()}")
        print(f"Backend: {type(self.backend).__name__}")
        print(f"{'='*70}\n")

        # Step 1: Load video
        print(f"[1/5] Loading video...")
        video_path = self.video_dir / f"{video_id}.mp4"
        video_frames, fps = self._load_video(video_id)
        print(f"  ✓ Loaded {len(video_frames)} frames @ {fps:.1f} fps")

        # Step 2: Detect scenes with PySceneDetect
        print(f"\n[2/5] Detecting scenes...")
        scene_list = self._detect_scenes(video_path)
        print(f"  ✓ Found {len(scene_list)} scenes")

        if scene_index < 1 or scene_index > len(scene_list):
            raise ValueError(
                f"Scene {scene_index} not found. Video has {len(scene_list)} scenes."
            )

        start_frame, end_frame = scene_list[scene_index - 1]
        scene_frames = video_frames[start_frame:end_frame]
        num_scene_frames = end_frame - start_frame
        print(f"  Scene {scene_index}: frames {start_frame}-{end_frame} ({num_scene_frames} frames)")

        # Step 3: Get product mask
        print(f"\n[3/5] Detecting product...")
        if use_keyword_mask and self.scenes_df is not None:
            scene_info = self._get_scene_info(video_id, scene_index)
            product_mask = self._load_mask(scene_info['keyword_mask_path'])
            print(f"  ✓ Using keyword mask")
        elif self.product_detector is not None:
            product_mask = self.product_detector.detect_main_product(
                video_frames,
                num_sample_frames=5,
            )
            print(f"  ✓ Auto-detected product (SAM + DINO)")
        else:
            product_mask = self._saliency_detect(scene_frames[len(scene_frames)//2])
            print(f"  ✓ Using saliency detection")

        product_pct = product_mask.sum() / product_mask.size * 100
        print(f"  Product region: {product_pct:.1f}%")

        # Step 4: Manipulate scene with backend (CONSISTENT!)
        print(f"\n[4/5] Manipulating scene (consistent background)...")
        manipulated_frames = self.backend.manipulate_video_segment(
            video_segment=scene_frames,
            mask=product_mask,
            action=action,
            fps=fps,
        )

        # Step 5: Replace scene and export
        print(f"\n[5/5] Exporting...")

        # Replace frames in original video
        edited_frames = video_frames.copy()
        for i, frame in enumerate(manipulated_frames):
            edited_frames[start_frame + i] = frame

        # Export video
        if output_path is None:
            video_output = self.videos_output_dir / f"{video_id}_scene{scene_index}_{action}.mp4"
        else:
            video_output = Path(output_path)

        self._export_video(edited_frames, video_output, fps)

        # Export sample frame
        mid_idx = len(manipulated_frames) // 2
        frame_output = self.frames_output_dir / f"{video_id}_scene{scene_index}_{action}.png"
        Image.fromarray(manipulated_frames[mid_idx]).save(frame_output)
        print(f"  ✓ Saved frame: {frame_output}")

        result = ManipulationResult(
            video_path=str(video_output),
            frame_path=str(frame_output),
            scene_start=start_frame,
            scene_end=end_frame,
            frames_manipulated=num_scene_frames,
            action=action,
        )

        print(f"\n{'='*70}")
        print(f"  ✓ SUCCESS!")
        print(f"{'='*70}")
        print(f"Video:  {result.video_path}")
        print(f"Frame:  {result.frame_path}")
        print(f"Frames: {num_scene_frames} (all consistent)")
        print(f"{'='*70}\n")

        return result

    def _detect_scenes(self, video_path: Path) -> List[Tuple[int, int]]:
        """Detect scene boundaries using PySceneDetect."""
        try:
            from scenedetect import detect, AdaptiveDetector
        except ImportError:
            raise ImportError(
                "PySceneDetect required. Install: pip install scenedetect[opencv]"
            )

        scene_list = detect(
            str(video_path),
            AdaptiveDetector(
                adaptive_threshold=3.0,
                min_scene_len=15,
            ),
        )

        frame_ranges = []
        for scene in scene_list:
            start = scene[0].get_frames()
            end = scene[1].get_frames()
            frame_ranges.append((start, end))

        if not frame_ranges:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_ranges = [(0, total_frames)]

        return frame_ranges

    def _saliency_detect(self, frame: np.ndarray) -> np.ndarray:
        """Fallback: Detect main content using saliency."""
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(frame)

        if not success:
            h, w = frame.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_prior = np.exp(-((x - w/2)**2 / (0.3*w)**2 + (y - h/2)**2 / (0.3*h)**2))
            return (center_prior > 0.5).astype(np.float32)

        threshold = np.percentile(saliency_map, 70)
        mask = (saliency_map > threshold).astype(np.float32)

        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask.astype(np.float32)

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load and binarize keyword mask."""
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask).astype(np.float32) / 255.0
        return (mask_np > 0.5).astype(np.float32)

    def _get_scene_info(self, video_id: str, scene_index: int) -> dict:
        """Get scene information from valid_scenes.csv."""
        scene_row = self.scenes_df[
            (self.scenes_df['video_id'].astype(str) == str(video_id)) &
            (self.scenes_df['scene_number'] == scene_index)
        ]

        if len(scene_row) == 0:
            available = self.scenes_df[
                self.scenes_df['video_id'].astype(str) == str(video_id)
            ]['scene_number'].tolist()
            raise ValueError(
                f"Scene {scene_index} not found for video {video_id}. "
                f"Available: {available}"
            )

        return scene_row.iloc[0].to_dict()

    def _load_video(self, video_id: str) -> tuple:
        """Load video as frames."""
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

    def _export_video(self, frames: list, output_path: Path, fps: float):
        """Export frames as video."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]

        codecs = [
            ('avc1', '.mp4'),
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
        ]

        for codec, ext in codecs:
            try:
                out_path = output_path.with_suffix(ext)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

                if writer.isOpened():
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                    print(f"  ✓ Exported video: {out_path}")
                    return
            except:
                continue

        raise RuntimeError("Failed to export video")

"""
Scene Manipulator - Full Scene Background Manipulation

Manipulate ENTIRE SCENES (not just single frames) to control attention.
Uses PySceneDetect to find real scene boundaries, then manipulates
ALL frames in the scene for smooth, realistic video output.

Key Concept:
- Product is AUTO-DETECTED (no keyword mask needed!)
- Uses SAM + DINO for episodic memory across video frames
- Background is modified to draw/deflect attention
- "increase" → Make background DRAMATICALLY less distracting
- "decrease" → Make background DRAMATICALLY more attention-grabbing
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Literal, Optional, Dict, List, Tuple
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
    Manipulate ENTIRE SCENES in videos by editing the BACKGROUND.

    Uses PySceneDetect to find real scene boundaries, then manipulates
    all frames in the scene - not just a single static image.

    The product/keyword region stays unchanged.
    Background is modified to increase/decrease attention on product.

    Usage:
        manipulator = SceneManipulator()

        result = manipulator.manipulate(
            video_id="123456",
            scene_index=6,
            action="increase",  # Make background DRAMATICALLY less distracting
        )

        print(result.video_path)  # Manipulated video
        print(result.frame_path)  # Sample manipulated frame
    """

    def __init__(
        self,
        valid_scenes_file: str = "data/valid_scenes.csv",
        video_dir: str = "data/data_tiktok",
        output_dir: str = "outputs/genai_v3",
        device: str = "cuda",
        auto_detect_product: bool = True,
    ):
        """
        Initialize scene manipulator.

        Args:
            valid_scenes_file: Path to valid_scenes.csv
            video_dir: Directory containing videos ({video_id}.mp4)
            output_dir: Directory for output videos and frames
            device: "cuda" or "cpu"
            auto_detect_product: If True, automatically detect main product
                                 using SAM + DINO (no keyword mask needed)
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

        # Load valid scenes (optional - only needed if using keyword masks)
        if valid_scenes_file and Path(valid_scenes_file).exists():
            self.scenes_df = pd.read_csv(valid_scenes_file)
            print(f"✓ Loaded {len(self.scenes_df)} valid scenes")
        else:
            self.scenes_df = None
            print("✓ Running without valid_scenes.csv (will auto-detect products)")

        # Initialize SOTA inpainting model
        print("Loading SDXL Inpainting model (SOTA)...")
        self._load_model()
        print("✓ SDXL loaded")

        # Initialize product detector (if auto-detect enabled)
        self.product_detector = None
        if auto_detect_product:
            print("Loading product detector (SAM + DINO)...")
            try:
                from .product_detector import MainProductDetector
                self.product_detector = MainProductDetector(device=device)
                print("✓ Product detector loaded")
            except Exception as e:
                print(f"⚠ Product detector not available: {e}")
                print("  Will fall back to keyword masks or saliency detection")

    def _load_model(self):
        """Load SOTA inpainting model (SDXL)."""
        from diffusers import AutoPipelineForInpainting

        # Use SDXL Inpainting - state of the art
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)

        # Enable optimizations
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except:
                pass

        # Enable attention slicing for lower memory
        self.pipe.enable_attention_slicing()

    def manipulate(
        self,
        video_id: str,
        scene_index: int,
        action: Literal["increase", "decrease"],
        strength: float = 0.95,
        num_inference_steps: int = 40,
        output_path: Optional[str] = None,
        use_keyword_mask: bool = False,
        keyframe_interval: int = 10,
    ) -> ManipulationResult:
        """
        Manipulate an ENTIRE SCENE and replace it in the video.

        Uses PySceneDetect to find actual scene boundaries, then manipulates
        ALL frames in the scene for smooth, realistic output.

        Modifies BACKGROUND only - product stays unchanged.
        Product is AUTO-DETECTED using episodic memory (SAM + DINO).

        Args:
            video_id: Video identifier
            scene_index: Scene number (1-indexed)
            action:
                - "increase": Make background DRAMATICALLY less distracting
                - "decrease": Make background DRAMATICALLY more attention-grabbing
            strength: Manipulation strength (0.0-1.0, higher = stronger effect)
            num_inference_steps: Quality setting (30-50 recommended)
            output_path: Custom output path (auto-generated if None)
            use_keyword_mask: If True, use keyword mask instead of auto-detection
            keyframe_interval: Manipulate every N frames as keyframe (interpolate between)

        Returns:
            ManipulationResult with paths to video and frame outputs
        """
        print(f"\n{'='*70}")
        print(f"  FULL SCENE MANIPULATION")
        print(f"{'='*70}")
        print(f"Video: {video_id}")
        print(f"Scene: {scene_index}")
        print(f"Action: {action.upper()} attention on product")
        print(f"Method: Edit BACKGROUND across ENTIRE scene")
        print(f"{'='*70}\n")

        # Step 1: Load original video
        print(f"[1/7] Loading original video...")
        video_path = self.video_dir / f"{video_id}.mp4"
        video_frames, fps = self._load_video(video_id)
        print(f"  ✓ Loaded {len(video_frames)} frames @ {fps:.1f} fps")

        # Step 2: Detect scene boundaries using PySceneDetect
        print(f"\n[2/7] Detecting scene boundaries with PySceneDetect...")
        scene_list = self._detect_scenes(video_path)
        print(f"  ✓ Detected {len(scene_list)} scenes")

        # Get the target scene's frame range
        if scene_index < 1 or scene_index > len(scene_list):
            raise ValueError(
                f"Scene {scene_index} not found. Video has {len(scene_list)} scenes."
            )

        start_frame, end_frame = scene_list[scene_index - 1]
        scene_frames = video_frames[start_frame:end_frame]
        num_scene_frames = end_frame - start_frame
        print(f"  Scene {scene_index}: frames {start_frame}-{end_frame} ({num_scene_frames} frames)")

        # Step 3: Get product mask (auto-detect or keyword mask)
        print(f"\n[3/7] Detecting main product...")
        if use_keyword_mask and self.scenes_df is not None:
            scene_info = self._get_scene_info(video_id, scene_index)
            product_mask = self._load_mask(scene_info['keyword_mask_path'])
            print(f"  ✓ Using keyword mask: '{scene_info.get('keyword', 'unknown')}'")
        elif self.product_detector is not None:
            product_mask = self.product_detector.detect_main_product(
                video_frames,
                num_sample_frames=5,
            )
            print(f"  ✓ Auto-detected main product (SAM + DINO)")
        else:
            product_mask = self._saliency_detect(scene_frames[len(scene_frames)//2])
            print(f"  ✓ Using saliency fallback")

        # IMPORTANT: Invert mask - we want to edit BACKGROUND, not product
        background_mask = 1.0 - product_mask

        product_pct = product_mask.sum() / product_mask.size * 100
        bg_pct = background_mask.sum() / background_mask.size * 100
        print(f"  Product: {product_pct:.1f}% | Background: {bg_pct:.1f}%")

        # Step 4: Manipulate keyframes across the scene
        print(f"\n[4/7] Manipulating scene keyframes...")
        keyframe_indices = list(range(0, num_scene_frames, keyframe_interval))
        if num_scene_frames - 1 not in keyframe_indices:
            keyframe_indices.append(num_scene_frames - 1)

        print(f"  Keyframes to process: {len(keyframe_indices)}")

        manipulated_keyframes = {}
        for i, kf_idx in enumerate(keyframe_indices):
            frame = scene_frames[kf_idx]
            frame_image = Image.fromarray(frame)

            # Resize mask to match frame
            h, w = frame.shape[:2]
            bg_mask_resized = cv2.resize(
                background_mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            manipulated = self._manipulate_background(
                image=frame_image,
                background_mask=bg_mask_resized,
                action=action,
                strength=strength,
                num_inference_steps=num_inference_steps,
            )
            manipulated_keyframes[kf_idx] = np.array(manipulated)
            print(f"    ✓ Keyframe {i+1}/{len(keyframe_indices)} (frame {kf_idx})")

        # Step 5: Interpolate between keyframes to create smooth video
        print(f"\n[5/7] Interpolating between keyframes...")
        manipulated_scene = self._interpolate_frames(
            original_frames=scene_frames,
            keyframes=manipulated_keyframes,
            product_mask=product_mask,
        )
        print(f"  ✓ Created {len(manipulated_scene)} smooth manipulated frames")

        # Step 6: Replace scene in original video
        print(f"\n[6/7] Replacing scene in video...")
        edited_frames = video_frames.copy()
        for i, frame in enumerate(manipulated_scene):
            edited_frames[start_frame + i] = frame
        print(f"  ✓ Replaced frames {start_frame}-{end_frame}")

        # Step 7: Export outputs
        print(f"\n[7/7] Exporting outputs...")

        # Export video
        if output_path is None:
            video_output = self.videos_output_dir / f"{video_id}_scene{scene_index}_{action}.mp4"
        else:
            video_output = Path(output_path)

        self._export_video(edited_frames, video_output, fps)

        # Export sample frame (middle of scene)
        mid_idx = len(manipulated_scene) // 2
        frame_output = self.frames_output_dir / f"{video_id}_scene{scene_index}_{action}.png"
        Image.fromarray(manipulated_scene[mid_idx]).save(frame_output)
        print(f"  ✓ Saved frame: {frame_output}")

        # Create result
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
        print(f"Scene:  {scene_index} (frames {start_frame}-{end_frame})")
        print(f"Frames: {num_scene_frames} manipulated")
        print(f"Action: Background {action}d to {'reduce' if action == 'increase' else 'add'} distraction")
        print(f"{'='*70}\n")

        return result

    def _detect_scenes(self, video_path: Path) -> List[Tuple[int, int]]:
        """
        Detect scene boundaries using PySceneDetect.

        Returns list of (start_frame, end_frame) tuples.
        """
        try:
            from scenedetect import detect, ContentDetector, AdaptiveDetector
        except ImportError:
            raise ImportError(
                "PySceneDetect required. Install: pip install scenedetect[opencv]"
            )

        # Use AdaptiveDetector for better results on diverse content
        scene_list = detect(
            str(video_path),
            AdaptiveDetector(
                adaptive_threshold=3.0,
                min_scene_len=15,  # Minimum 15 frames per scene
            ),
        )

        # Convert to frame ranges
        frame_ranges = []
        for scene in scene_list:
            start = scene[0].get_frames()
            end = scene[1].get_frames()
            frame_ranges.append((start, end))

        # Handle edge case: no scenes detected
        if not frame_ranges:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_ranges = [(0, total_frames)]

        return frame_ranges

    def _manipulate_background(
        self,
        image: Image.Image,
        background_mask: np.ndarray,
        action: str,
        strength: float,
        num_inference_steps: int,
    ) -> Image.Image:
        """
        Manipulate background with DRAMATIC effect.

        - "increase": Make background EXTREMELY boring/muted
        - "decrease": Make background EXTREMELY vibrant/attention-grabbing
        """
        # Resize for SDXL (1024x1024 optimal)
        original_size = image.size
        image_resized = image.resize((1024, 1024), Image.LANCZOS)

        # Resize mask
        mask_resized = cv2.resize(
            background_mask,
            (1024, 1024),
            interpolation=cv2.INTER_NEAREST
        )

        # Expand mask for better blending at edges
        kernel = np.ones((21, 21), np.uint8)
        mask_expanded = cv2.dilate(mask_resized.astype(np.uint8), kernel, iterations=2)
        mask_pil = Image.fromarray((mask_expanded * 255).astype(np.uint8))

        # DRAMATIC prompts based on action
        if action == "increase":
            # Make background EXTREMELY boring → all attention on product
            prompt = (
                "completely plain solid gray background, "
                "extremely out of focus, heavily blurred, "
                "no details whatsoever, pure muted tones, "
                "flat featureless backdrop, studio gray seamless, "
                "professional product photography on plain background, "
                "absolutely nothing interesting, boring empty space"
            )
            negative_prompt = (
                "detailed, colorful, vibrant, interesting, patterns, "
                "textures, shapes, objects, busy, cluttered, sharp focus, "
                "any detail whatsoever, anything eye-catching"
            )
            guidance = 12.0  # High guidance for strong effect
        else:  # decrease
            # Make background EXTREMELY attention-grabbing
            prompt = (
                "extremely vibrant colorful psychedelic background, "
                "wild crazy patterns, neon colors, "
                "rainbow gradients, explosive visual chaos, "
                "eye-catching detailed textures everywhere, "
                "maximum visual interest, disco lights, "
                "kaleidoscope patterns, graffiti art style, "
                "attention-demanding visual explosion"
            )
            negative_prompt = (
                "plain, simple, muted, boring, gray, beige, "
                "minimal, subtle, professional, clean, empty, "
                "out of focus, blurry, solid colors"
            )
            guidance = 15.0  # Very high guidance for extreme effect

        # Run inpainting on BACKGROUND only
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_resized,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance,
            ).images[0]

        # Resize back to original size
        result_final = result.resize(original_size, Image.LANCZOS)

        return result_final

    def _interpolate_frames(
        self,
        original_frames: List[np.ndarray],
        keyframes: Dict[int, np.ndarray],
        product_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Interpolate between manipulated keyframes for smooth video.

        Uses optical flow to warp background changes between keyframes,
        while keeping product region from original frames.
        """
        num_frames = len(original_frames)
        result = []

        # Sort keyframe indices
        kf_indices = sorted(keyframes.keys())

        for i in range(num_frames):
            h, w = original_frames[i].shape[:2]

            # Resize product mask
            mask = cv2.resize(product_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_3ch = np.stack([mask] * 3, axis=-1)

            if i in keyframes:
                # This is a keyframe - use directly
                manipulated = keyframes[i]
            else:
                # Interpolate between nearest keyframes
                prev_kf = max([k for k in kf_indices if k <= i], default=kf_indices[0])
                next_kf = min([k for k in kf_indices if k >= i], default=kf_indices[-1])

                if prev_kf == next_kf:
                    manipulated = keyframes[prev_kf]
                else:
                    # Linear interpolation weight
                    alpha = (i - prev_kf) / (next_kf - prev_kf)

                    prev_frame = keyframes[prev_kf]
                    next_frame = keyframes[next_kf]

                    # Resize if needed
                    if prev_frame.shape[:2] != (h, w):
                        prev_frame = cv2.resize(prev_frame, (w, h))
                    if next_frame.shape[:2] != (h, w):
                        next_frame = cv2.resize(next_frame, (w, h))

                    # Blend between keyframes
                    manipulated = (
                        (1 - alpha) * prev_frame.astype(float) +
                        alpha * next_frame.astype(float)
                    ).astype(np.uint8)

            # Ensure manipulated frame matches size
            if manipulated.shape[:2] != (h, w):
                manipulated = cv2.resize(manipulated, (w, h))

            # Composite: keep product from original, use manipulated background
            final = (
                mask_3ch * original_frames[i].astype(float) +
                (1 - mask_3ch) * manipulated.astype(float)
            ).astype(np.uint8)

            result.append(final)

        return result

    def _saliency_detect(self, frame: np.ndarray) -> np.ndarray:
        """Fallback: Detect main content using saliency."""
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(frame)

        if not success:
            # Last resort: center prior
            h, w = frame.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_prior = np.exp(-((x - w/2)**2 / (0.3*w)**2 + (y - h/2)**2 / (0.3*h)**2))
            return (center_prior > 0.5).astype(np.float32)

        # Threshold
        threshold = np.percentile(saliency_map, 70)
        mask = (saliency_map > threshold).astype(np.float32)

        # Clean up
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask.astype(np.float32)

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load and binarize keyword mask."""
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        return mask_np

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
        """Export frames as video with proper codec."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]

        # Try different codecs in order of preference
        codecs = [
            ('avc1', '.mp4'),  # H.264
            ('mp4v', '.mp4'),  # MPEG-4
            ('XVID', '.avi'),  # Xvid
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

        raise RuntimeError("Failed to export video with any codec")

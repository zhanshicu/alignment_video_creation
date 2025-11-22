"""
Frame Generator for Veo Input

Generates START and END frames for Google Veo's "Frame to Video" mode.
These frames define the background manipulation while keeping product consistent.

Workflow:
1. Extract reference frame from target scene
2. Auto-detect product using SAM + DINO
3. Generate START frame (original background slightly modified)
4. Generate END frame (target background - plain/vibrant)
5. Save frames + provide Veo prompt instructions
"""

import numpy as np
import cv2
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict
from dataclasses import dataclass
import torch


@dataclass
class VeoInputs:
    """Generated inputs for Veo website."""
    start_frame_path: str
    end_frame_path: str
    prompt: str
    negative_prompt: str
    recommended_settings: Dict[str, str]
    scene_info: Dict[str, int]


class FrameGenerator:
    """
    Generate start and end frames for Veo's Frame-to-Video mode.

    The key insight:
    - START frame: Original scene (or slightly modified)
    - END frame: Target background state (plain gray or vibrant)
    - Product stays IDENTICAL in both frames
    - Veo interpolates the background transition smoothly
    """

    def __init__(
        self,
        video_dir: str = "data/data_tiktok",
        output_dir: str = "outputs/genai_v4/veo_inputs",
        device: str = "cuda",
    ):
        """
        Initialize frame generator.

        Args:
            video_dir: Directory containing source videos
            output_dir: Directory to save generated frames
            device: "cuda" or "cpu"
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize image generation and product detection models."""
        # Product detector
        print("Loading product detector...")
        self.product_detector = None
        try:
            from GenAI_v3.product_detector import MainProductDetector
            self.product_detector = MainProductDetector(device=self.device)
            print("✓ Product detector loaded (SAM + DINO)")
        except Exception as e:
            print(f"⚠ Product detector unavailable: {e}")
            print("  Will use saliency detection fallback")

        # Image generator (SDXL for background generation)
        print("Loading SDXL Inpainting...")
        from diffusers import AutoPipelineForInpainting

        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        self.pipe.enable_attention_slicing()
        print("✓ SDXL loaded")

    def generate_veo_inputs(
        self,
        video_id: str,
        scene_index: int,
        action: Literal["increase", "decrease"],
        transition_style: Literal["smooth", "dramatic"] = "smooth",
    ) -> VeoInputs:
        """
        Generate start and end frames for Veo.

        Args:
            video_id: Video identifier
            scene_index: Scene number (1-indexed)
            action: "increase" (plain bg) or "decrease" (vibrant bg)
            transition_style: "smooth" (subtle) or "dramatic" (strong)

        Returns:
            VeoInputs with paths to frames and Veo instructions
        """
        print(f"\n{'='*70}")
        print(f"  GENERATING VEO INPUTS")
        print(f"{'='*70}")
        print(f"Video: {video_id}")
        print(f"Scene: {scene_index}")
        print(f"Action: {action}")
        print(f"{'='*70}\n")

        # Step 1: Load video and detect scenes
        print("[1/5] Loading video and detecting scenes...")
        video_frames, fps = self._load_video(video_id)
        scene_list = self._detect_scenes(self.video_dir / f"{video_id}.mp4")

        if scene_index < 1 or scene_index > len(scene_list):
            raise ValueError(f"Scene {scene_index} not found. Video has {len(scene_list)} scenes.")

        start_frame_idx, end_frame_idx = scene_list[scene_index - 1]
        scene_frames = video_frames[start_frame_idx:end_frame_idx]
        print(f"  ✓ Scene {scene_index}: frames {start_frame_idx}-{end_frame_idx}")

        # Step 2: Get reference frames (first and last of scene)
        print("\n[2/5] Extracting reference frames...")
        first_frame = scene_frames[0]
        last_frame = scene_frames[-1]
        mid_frame = scene_frames[len(scene_frames) // 2]
        h, w = first_frame.shape[:2]
        print(f"  ✓ Frame size: {w}x{h}")

        # Step 3: Detect product
        print("\n[3/5] Detecting product...")
        if self.product_detector:
            product_mask = self.product_detector.detect_main_product(
                video_frames,
                num_sample_frames=5,
            )
        else:
            product_mask = self._saliency_detect(mid_frame)

        product_mask = cv2.resize(product_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        product_pct = product_mask.sum() / product_mask.size * 100
        print(f"  ✓ Product region: {product_pct:.1f}%")

        # Step 4: Generate START and END frames
        print("\n[4/5] Generating frames for Veo...")

        # START frame: Original or slightly modified
        start_frame = self._generate_start_frame(
            first_frame, product_mask, action, transition_style
        )

        # END frame: Target background state
        end_frame = self._generate_end_frame(
            last_frame, product_mask, action, transition_style
        )

        # Step 5: Save frames and generate instructions
        print("\n[5/5] Saving frames and generating Veo instructions...")

        # Create output directory for this video
        video_output_dir = self.output_dir / f"{video_id}_scene{scene_index}_{action}"
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Save frames
        start_path = video_output_dir / "start_frame.png"
        end_path = video_output_dir / "end_frame.png"

        Image.fromarray(start_frame).save(start_path)
        Image.fromarray(end_frame).save(end_path)
        print(f"  ✓ Saved: {start_path}")
        print(f"  ✓ Saved: {end_path}")

        # Also save product mask for reference
        mask_path = video_output_dir / "product_mask.png"
        Image.fromarray((product_mask * 255).astype(np.uint8)).save(mask_path)

        # Generate Veo prompt and settings
        prompt, negative_prompt, settings = self._generate_veo_instructions(
            action, transition_style, len(scene_frames), fps
        )

        # Save instructions to text file
        instructions_path = video_output_dir / "veo_instructions.txt"
        self._save_instructions(
            instructions_path, prompt, negative_prompt, settings,
            start_path, end_path, len(scene_frames), fps
        )
        print(f"  ✓ Saved: {instructions_path}")

        result = VeoInputs(
            start_frame_path=str(start_path),
            end_frame_path=str(end_path),
            prompt=prompt,
            negative_prompt=negative_prompt,
            recommended_settings=settings,
            scene_info={
                "scene_index": scene_index,
                "start_frame": start_frame_idx,
                "end_frame": end_frame_idx,
                "num_frames": len(scene_frames),
                "fps": fps,
            }
        )

        print(f"\n{'='*70}")
        print(f"  ✓ VEO INPUTS GENERATED")
        print(f"{'='*70}")
        print(f"\nFiles saved to: {video_output_dir}/")
        print(f"  - start_frame.png")
        print(f"  - end_frame.png")
        print(f"  - product_mask.png")
        print(f"  - veo_instructions.txt")
        print(f"\n{'='*70}")

        return result

    def _generate_start_frame(
        self,
        frame: np.ndarray,
        product_mask: np.ndarray,
        action: str,
        style: str,
    ) -> np.ndarray:
        """Generate START frame for Veo (original or slight modification)."""
        # For smooth transition, start frame is mostly original
        # For dramatic, we slightly modify toward neutral

        if style == "smooth":
            # Keep original
            return frame.copy()
        else:
            # Slight modification toward target direction
            h, w = frame.shape[:2]
            bg_mask = 1.0 - product_mask

            # Apply subtle blur/desaturation to background
            frame_pil = Image.fromarray(frame)

            if action == "increase":
                # Slightly desaturate background
                blurred = frame_pil.filter(ImageFilter.GaussianBlur(radius=2))
            else:
                # Slightly enhance background
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Color(frame_pil)
                blurred = enhancer.enhance(1.2)

            blurred_np = np.array(blurred)

            # Composite
            mask_3ch = np.stack([bg_mask] * 3, axis=-1)
            result = (
                (1 - mask_3ch) * frame.astype(float) +
                mask_3ch * blurred_np.astype(float) * 0.3 +
                mask_3ch * frame.astype(float) * 0.7
            ).astype(np.uint8)

            return result

    def _generate_end_frame(
        self,
        frame: np.ndarray,
        product_mask: np.ndarray,
        action: str,
        style: str,
    ) -> np.ndarray:
        """Generate END frame for Veo (target background state)."""
        h, w = frame.shape[:2]
        bg_mask = 1.0 - product_mask

        # Prepare for SDXL
        frame_pil = Image.fromarray(frame)
        frame_resized = frame_pil.resize((1024, 1024), Image.LANCZOS)

        bg_mask_1024 = cv2.resize(bg_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((15, 15), np.uint8)
        bg_mask_expanded = cv2.dilate(bg_mask_1024.astype(np.uint8), kernel, iterations=2)
        mask_pil = Image.fromarray((bg_mask_expanded * 255).astype(np.uint8))

        # Get prompt based on action and style
        if action == "increase":
            if style == "dramatic":
                prompt = (
                    "completely plain solid neutral gray background, "
                    "professional studio backdrop, extremely simple, "
                    "no details, no patterns, uniform matte gray, "
                    "soft diffused lighting, product photography setup"
                )
            else:
                prompt = (
                    "soft muted background, gentle out of focus, "
                    "subtle neutral tones, calm unobtrusive backdrop, "
                    "professional product photography environment"
                )
            negative = "colorful, vibrant, detailed, patterns, busy, distracting"
            guidance = 12.0 if style == "dramatic" else 9.0
        else:  # decrease
            if style == "dramatic":
                prompt = (
                    "extremely vibrant colorful background, "
                    "bold bright colors, eye-catching patterns, "
                    "dynamic visual elements, neon accents, "
                    "exciting energetic atmosphere, rich textures"
                )
            else:
                prompt = (
                    "colorful interesting background, "
                    "warm inviting tones, subtle patterns, "
                    "engaging visual environment, pleasant colors"
                )
            negative = "plain, gray, muted, boring, simple, dull"
            guidance = 15.0 if style == "dramatic" else 10.0

        # Generate with SDXL
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=frame_resized,
                mask_image=mask_pil,
                num_inference_steps=40,
                strength=0.9 if style == "dramatic" else 0.7,
                guidance_scale=guidance,
            ).images[0]

        # Resize back and composite
        result_resized = np.array(result.resize((w, h), Image.LANCZOS))

        mask_3ch = np.stack([product_mask] * 3, axis=-1)
        final = (
            mask_3ch * frame.astype(float) +
            (1 - mask_3ch) * result_resized.astype(float)
        ).astype(np.uint8)

        return final

    def _generate_veo_instructions(
        self,
        action: str,
        style: str,
        num_frames: int,
        fps: float,
    ) -> Tuple[str, str, Dict[str, str]]:
        """Generate prompt and settings for Veo website."""
        duration = num_frames / fps

        if action == "increase":
            prompt = (
                "Smooth transition of background becoming simpler and more muted. "
                "The central subject remains perfectly still and unchanged. "
                "Background gradually transforms to plain neutral gray. "
                "Soft, professional product photography lighting. "
                "Camera is completely static. No movement of the main subject."
            )
            negative = (
                "subject movement, camera shake, colorful background, "
                "busy patterns, distracting elements, zooming"
            )
        else:  # decrease
            prompt = (
                "Smooth transition of background becoming more vibrant and colorful. "
                "The central subject remains perfectly still and unchanged. "
                "Background gradually transforms with rich colors and subtle patterns. "
                "Dynamic lighting effects in background only. "
                "Camera is completely static. No movement of the main subject."
            )
            negative = (
                "subject movement, camera shake, plain background, "
                "gray tones, muted colors, zooming"
            )

        settings = {
            "Duration": f"{duration:.1f} seconds (to match scene)",
            "Aspect Ratio": "Match your video (e.g., 16:9 or 9:16)",
            "Motion": "Minimal / Static camera",
            "Style": "Photorealistic",
            "Seed": "Use same seed for consistency if regenerating",
        }

        return prompt, negative, settings

    def _save_instructions(
        self,
        path: Path,
        prompt: str,
        negative: str,
        settings: Dict[str, str],
        start_path: Path,
        end_path: Path,
        num_frames: int,
        fps: float,
    ):
        """Save detailed instructions for Veo website."""
        content = f"""
================================================================================
                    VEO INPUT INSTRUCTIONS
================================================================================

STEP 1: Open Google Veo Website
-------------------------------
Go to: https://labs.google/fx/tools/video-fx
Select: "Frame to Video" mode


STEP 2: Upload Frames
---------------------
START FRAME: {start_path}
END FRAME:   {end_path}

Upload these two images as the start and end frames.


STEP 3: Enter Prompt
--------------------
PROMPT:
{prompt}

NEGATIVE PROMPT (if supported):
{negative}


STEP 4: Configure Settings
--------------------------
"""
        for key, value in settings.items():
            content += f"- {key}: {value}\n"

        content += f"""

STEP 5: Generate Video
----------------------
Click generate and wait for the video to be created.


STEP 6: Download and Return
---------------------------
1. Download the generated video
2. Rename it to: veo_output.mp4
3. Place it in the same folder as these instructions:
   {path.parent}/veo_output.mp4


SCENE INFO
----------
- Number of frames to replace: {num_frames}
- Original FPS: {fps:.2f}
- Target duration: {num_frames / fps:.2f} seconds


================================================================================
"""
        with open(path, 'w') as f:
            f.write(content)

    def _load_video(self, video_id: str) -> Tuple[list, float]:
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

    def _detect_scenes(self, video_path: Path) -> list:
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

    def _saliency_detect(self, frame: np.ndarray) -> np.ndarray:
        """Fallback saliency detection."""
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(frame)

        if not success:
            h, w = frame.shape[:2]
            y, x = np.ogrid[:h, :w]
            center = np.exp(-((x - w/2)**2 / (0.3*w)**2 + (y - h/2)**2 / (0.3*h)**2))
            return (center > 0.5).astype(np.float32)

        threshold = np.percentile(saliency_map, 70)
        mask = (saliency_map > threshold).astype(np.float32)

        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask.astype(np.float32)

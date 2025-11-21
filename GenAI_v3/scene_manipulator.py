"""
Scene Manipulator - Simple Background Manipulation

Manipulate BACKGROUND (not product) to control attention-keyword alignment.
Uses SOTA pre-trained models (SDXL Inpainting).

Key Concept:
- Product stays UNCHANGED (authentic)
- Background is modified to draw/deflect attention
- "increase" → Make background less distracting (muted, blurred)
- "decrease" → Make background more attention-grabbing (vibrant, detailed)
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Literal, Optional
import cv2


class SceneManipulator:
    """
    Manipulate specific scenes in videos by editing the BACKGROUND.

    The product/keyword region stays unchanged.
    Background is modified to increase/decrease attention on product.

    Usage:
        manipulator = SceneManipulator()

        output = manipulator.manipulate(
            video_id="123456",
            scene_index=6,
            action="increase",  # Make background less distracting
        )
    """

    def __init__(
        self,
        valid_scenes_file: str = "data/valid_scenes.csv",
        video_dir: str = "data/data_tiktok",
        output_dir: str = "outputs/genai_v3/manipulated_videos",
        device: str = "cuda",
    ):
        """
        Initialize scene manipulator.

        Args:
            valid_scenes_file: Path to valid_scenes.csv
            video_dir: Directory containing videos ({video_id}.mp4)
            output_dir: Directory for output videos
            device: "cuda" or "cpu"
        """
        self.valid_scenes_file = Path(valid_scenes_file)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Load valid scenes
        self.scenes_df = pd.read_csv(valid_scenes_file)
        print(f"✓ Loaded {len(self.scenes_df)} valid scenes")

        # Initialize SOTA model
        print("Loading SDXL Inpainting model (SOTA)...")
        self._load_model()
        print("✓ Model loaded")

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
        strength: float = 0.8,
        num_inference_steps: int = 30,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Manipulate a specific scene and replace it in the video.

        Modifies BACKGROUND only - product stays unchanged.

        Args:
            video_id: Video identifier
            scene_index: Scene number (1-indexed)
            action:
                - "increase": Make background less distracting → more attention on product
                - "decrease": Make background more interesting → less attention on product
            strength: Manipulation strength (0.0-1.0, higher = stronger effect)
            num_inference_steps: Quality setting (20-50)
            output_path: Custom output path (auto-generated if None)

        Returns:
            Path to the output video
        """
        print(f"\n{'='*60}")
        print(f"Manipulating Scene {scene_index} in Video {video_id}")
        print(f"Action: {action.upper()} attention on product")
        print(f"Method: Edit BACKGROUND only (product unchanged)")
        print(f"{'='*60}\n")

        # Step 1: Load scene info
        scene_info = self._get_scene_info(video_id, scene_index)
        keyword = scene_info['keyword']
        print(f"Keyword/Product: '{keyword}'")

        # Step 2: Load scene image and keyword mask
        scene_image = Image.open(scene_info['scene_image_path']).convert('RGB')
        keyword_mask = self._load_mask(scene_info['keyword_mask_path'])

        # IMPORTANT: Invert mask - we want to edit BACKGROUND, not product
        background_mask = 1.0 - keyword_mask

        print(f"✓ Loaded scene image and masks")
        print(f"  Product region: {keyword_mask.sum() / keyword_mask.size * 100:.1f}% of image")
        print(f"  Background region: {background_mask.sum() / background_mask.size * 100:.1f}% of image")

        # Step 3: Manipulate background
        print(f"\nManipulating background ({action})...")
        manipulated_scene = self._manipulate_background(
            image=scene_image,
            background_mask=background_mask,
            keyword=keyword,
            action=action,
            strength=strength,
            num_inference_steps=num_inference_steps,
        )
        print(f"✓ Background manipulation complete")

        # Step 4: Load original video
        print(f"\nLoading original video...")
        video_frames, fps = self._load_video(video_id)
        print(f"✓ Loaded {len(video_frames)} frames @ {fps:.1f} fps")

        # Step 5: Find scene location and replace
        print(f"\nReplacing scene in video...")
        start_frame, end_frame = self._get_scene_frame_range(
            video_id, scene_index, len(video_frames)
        )

        edited_frames = self._replace_scene(
            video_frames=video_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            replacement=manipulated_scene,
        )

        # Step 6: Export video
        if output_path is None:
            output_path = self.output_dir / f"{video_id}_scene{scene_index}_{action}.mp4"
        else:
            output_path = Path(output_path)

        self._export_video(edited_frames, output_path, fps)

        print(f"\n{'='*60}")
        print(f"✓ SUCCESS!")
        print(f"{'='*60}")
        print(f"Original: {self.video_dir / f'{video_id}.mp4'}")
        print(f"Edited:   {output_path}")
        print(f"Scene {scene_index}: Background {action}d to {'reduce' if action == 'increase' else 'add'} distraction")
        print(f"{'='*60}\n")

        return str(output_path)

    def _manipulate_background(
        self,
        image: Image.Image,
        background_mask: np.ndarray,
        keyword: str,
        action: str,
        strength: float,
        num_inference_steps: int,
    ) -> Image.Image:
        """
        Manipulate background to control attention.

        - "increase": Make background boring/muted → attention goes to product
        - "decrease": Make background interesting → attention diverts from product
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

        # Expand mask slightly for better blending
        kernel = np.ones((15, 15), np.uint8)
        mask_expanded = cv2.dilate(mask_resized.astype(np.uint8), kernel, iterations=1)
        mask_pil = Image.fromarray((mask_expanded * 255).astype(np.uint8))

        # Generate prompt based on action
        if action == "increase":
            # Make background LESS interesting → product stands out
            prompt = (
                f"simple plain muted background, soft out of focus, "
                f"minimal details, neutral colors, subtle, unobtrusive, "
                f"professional product photography background"
            )
            negative_prompt = (
                "busy, cluttered, colorful, vibrant, detailed, "
                "interesting, eye-catching, distracting, sharp"
            )
        else:  # decrease
            # Make background MORE interesting → diverts attention
            prompt = (
                f"vibrant colorful detailed background, interesting patterns, "
                f"eye-catching elements, dynamic lighting, rich textures, "
                f"visually engaging environment"
            )
            negative_prompt = (
                "plain, boring, simple, muted, dull, uniform, "
                "out of focus, blurry, minimal"
            )

        # Run inpainting on BACKGROUND only
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_resized,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=7.5,
            ).images[0]

        # Resize back to original size
        result_final = result.resize(original_size, Image.LANCZOS)

        return result_final

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load and binarize keyword mask."""
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask).astype(np.float32) / 255.0
        # Binarize
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

    def _get_scene_frame_range(
        self,
        video_id: str,
        scene_index: int,
        total_frames: int
    ) -> tuple:
        """Estimate frame range for a scene."""
        video_scenes = self.scenes_df[
            self.scenes_df['video_id'].astype(str) == str(video_id)
        ].sort_values('scene_number')

        num_scenes = len(video_scenes)
        frames_per_scene = total_frames // num_scenes

        start = (scene_index - 1) * frames_per_scene
        end = start + frames_per_scene

        return max(0, start), min(total_frames, end)

    def _replace_scene(
        self,
        video_frames: list,
        start_frame: int,
        end_frame: int,
        replacement: Image.Image,
        blend_frames: int = 5,
    ) -> list:
        """Replace frames with smooth blending."""
        edited = video_frames.copy()

        # Resize replacement to match video
        h, w = video_frames[0].shape[:2]
        replacement_np = np.array(replacement.resize((w, h), Image.LANCZOS))

        for i in range(start_frame, end_frame):
            # Smooth blending at boundaries
            if i < start_frame + blend_frames:
                alpha = (i - start_frame) / blend_frames
            elif i >= end_frame - blend_frames:
                alpha = (end_frame - i) / blend_frames
            else:
                alpha = 1.0

            blended = (alpha * replacement_np + (1 - alpha) * video_frames[i]).astype(np.uint8)
            edited[i] = blended

        print(f"✓ Replaced frames {start_frame}-{end_frame}")
        return edited

    def _export_video(self, frames: list, output_path: Path, fps: float):
        """Export frames as video."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )

        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"✓ Exported: {output_path}")

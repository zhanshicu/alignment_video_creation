"""
Zero-Shot Attention-Keyword Alignment Manipulator

Uses pre-trained generative models (InstructPix2Pix, Inpainting) to manipulate
attention-keyword alignment WITHOUT any training.

Key Features:
- No training required - uses pre-trained models as-is
- Fast inference (~2-3s per frame)
- Flexible control via text instructions
- Multiple manipulation strategies
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Literal
import cv2
from pathlib import Path


class ZeroShotAlignmentManipulator:
    """
    Zero-shot manipulation of attention-keyword alignment using pre-trained models.

    Supports two modes:
    1. InstructPix2Pix: Text-guided image editing
    2. Inpainting: Region-based editing with masks
    """

    def __init__(
        self,
        method: Literal["instruct_pix2pix", "inpainting"] = "instruct_pix2pix",
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        """
        Initialize the manipulator.

        Args:
            method: "instruct_pix2pix" (simpler) or "inpainting" (more precise)
            device: "cuda" or "cpu"
            torch_dtype: torch.float16 for speed, torch.float32 for quality
        """
        self.method = method
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"Initializing {method} pipeline...")
        self._load_model()
        print("✓ Model loaded successfully")

    def _load_model(self):
        """Load the appropriate pre-trained model."""
        if self.method == "instruct_pix2pix":
            from diffusers import StableDiffusionInstructPix2PixPipeline

            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            ).to(self.device)

            # Enable optimizations
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass

        elif self.method == "inpainting":
            from diffusers import StableDiffusionInpaintPipeline

            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            ).to(self.device)

            # Enable optimizations
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()

    def manipulate_frame(
        self,
        frame: Image.Image,
        keyword: str,
        boost_level: float = 1.5,
        keyword_mask: Optional[np.ndarray] = None,
        num_inference_steps: int = 20,
    ) -> Image.Image:
        """
        Manipulate a single frame to change keyword prominence.

        Args:
            frame: Input frame (PIL Image)
            keyword: Product keyword (e.g., "sneakers", "smartphone")
            boost_level: Manipulation strength
                - > 1.0: Make keyword MORE prominent (boost)
                - < 1.0: Make keyword LESS prominent (reduce)
                - = 1.0: No change
            keyword_mask: Binary mask for keyword region (required for inpainting)
            num_inference_steps: Diffusion steps (20-30 for speed, 50 for quality)

        Returns:
            Edited frame (PIL Image)
        """
        if self.method == "instruct_pix2pix":
            return self._edit_with_instruction(
                frame, keyword, boost_level, num_inference_steps
            )

        elif self.method == "inpainting":
            if keyword_mask is None:
                raise ValueError("keyword_mask is required for inpainting method")
            return self._edit_with_inpainting(
                frame, keyword_mask, keyword, boost_level, num_inference_steps
            )

    def _edit_with_instruction(
        self,
        frame: Image.Image,
        keyword: str,
        boost_level: float,
        num_inference_steps: int,
    ) -> Image.Image:
        """Edit using InstructPix2Pix (text instructions)."""

        # Generate instruction based on boost level
        if boost_level > 1.0:
            # Boost: make more prominent
            if boost_level >= 2.0:
                instruction = f"Make the {keyword} very prominent and eye-catching, enhance lighting and contrast"
            elif boost_level >= 1.5:
                instruction = f"Make the {keyword} more visually prominent and noticeable"
            else:
                instruction = f"Make the {keyword} slightly more noticeable"

        elif boost_level < 1.0:
            # Reduce: make less prominent
            instruction = f"Make the {keyword} blend more into the background, reduce emphasis"

        else:
            # No change
            return frame

        # Run inference
        with torch.no_grad():
            edited = self.pipe(
                prompt=instruction,
                image=frame,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=1.5,
                guidance_scale=7.5,
            ).images[0]

        return edited

    def _edit_with_inpainting(
        self,
        frame: Image.Image,
        keyword_mask: np.ndarray,
        keyword: str,
        boost_level: float,
        num_inference_steps: int,
    ) -> Image.Image:
        """Edit using inpainting (region-based with mask)."""

        # Expand mask slightly for smooth blending
        mask_expanded = cv2.dilate(
            keyword_mask.astype(np.uint8),
            kernel=np.ones((15, 15), np.uint8),
            iterations=1
        )

        # Convert to PIL
        mask_pil = Image.fromarray((mask_expanded * 255).astype(np.uint8))

        # Generate prompt based on boost level
        if boost_level > 1.0:
            prompt = f"high quality {keyword}, vibrant colors, good lighting, prominent, sharp focus"
            negative_prompt = "blurry, dark, hidden, obscured, low quality"
            strength = min(0.8, 0.5 + (boost_level - 1.0) * 0.3)

        elif boost_level < 1.0:
            prompt = f"subtle {keyword}, muted colors, blends with surroundings"
            negative_prompt = "prominent, bright, eye-catching, vibrant"
            strength = max(0.4, 0.7 - (1.0 - boost_level) * 0.3)

        else:
            return frame

        # Run inpainting
        with torch.no_grad():
            edited = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=frame,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                strength=strength,
            ).images[0]

        return edited

    def create_variant(
        self,
        scenes: List[Image.Image],
        keyword: str,
        variant_type: str,
        keyword_masks: Optional[List[np.ndarray]] = None,
        num_inference_steps: int = 20,
    ) -> List[Image.Image]:
        """
        Create a variant by manipulating multiple scenes.

        Args:
            scenes: List of scene frames (PIL Images)
            keyword: Product keyword
            variant_type: One of:
                - "baseline": No editing
                - "early_boost": Boost first 33%
                - "middle_boost": Boost middle 33%
                - "late_boost": Boost last 33%
                - "full_boost": Boost all scenes
                - "reduction": Reduce middle 33%
                - "placebo": No editing (control)
            keyword_masks: List of keyword masks (required for inpainting)
            num_inference_steps: Diffusion steps

        Returns:
            List of edited scenes
        """
        num_scenes = len(scenes)
        edited_scenes = []

        print(f"Creating {variant_type} variant ({num_scenes} scenes)...")

        for i, scene in enumerate(scenes):
            # Determine if this scene should be edited
            should_edit, boost_level = self._get_edit_params(
                i, num_scenes, variant_type
            )

            if should_edit:
                mask = keyword_masks[i] if keyword_masks is not None else None
                edited = self.manipulate_frame(
                    frame=scene,
                    keyword=keyword,
                    boost_level=boost_level,
                    keyword_mask=mask,
                    num_inference_steps=num_inference_steps,
                )
                print(f"  Scene {i+1}/{num_scenes}: Edited (boost={boost_level:.2f})")
            else:
                edited = scene
                print(f"  Scene {i+1}/{num_scenes}: Unchanged")

            edited_scenes.append(edited)

        print(f"✓ Variant '{variant_type}' created")
        return edited_scenes

    def _get_edit_params(
        self,
        scene_idx: int,
        total_scenes: int,
        variant_type: str,
    ) -> Tuple[bool, float]:
        """
        Determine which scenes to edit and by how much.

        Returns:
            (should_edit, boost_level)
        """
        third = total_scenes // 3

        if variant_type == "baseline" or variant_type == "placebo":
            return False, 1.0

        elif variant_type == "early_boost":
            should_edit = scene_idx < third
            return should_edit, 1.5 if should_edit else 1.0

        elif variant_type == "middle_boost":
            should_edit = third <= scene_idx < 2 * third
            return should_edit, 1.5 if should_edit else 1.0

        elif variant_type == "late_boost":
            should_edit = scene_idx >= 2 * third
            return should_edit, 1.5 if should_edit else 1.0

        elif variant_type == "full_boost":
            return True, 1.5

        elif variant_type == "reduction":
            should_edit = third <= scene_idx < 2 * third
            return should_edit, 0.5 if should_edit else 1.0

        else:
            raise ValueError(f"Unknown variant_type: {variant_type}")


def load_scenes_from_paths(scene_paths: List[str]) -> List[Image.Image]:
    """Load scene images from file paths."""
    scenes = []
    for path in scene_paths:
        img = Image.open(path).convert('RGB')
        scenes.append(img)
    return scenes


def load_masks_from_paths(mask_paths: List[str]) -> List[np.ndarray]:
    """Load keyword masks from file paths."""
    masks = []
    for path in mask_paths:
        mask = Image.open(path).convert('L')
        mask = np.array(mask).astype(np.float32) / 255.0
        # Binarize
        mask = (mask > 0.5).astype(np.float32)
        masks.append(mask)
    return masks


def save_scenes(scenes: List[Image.Image], output_dir: Path, prefix: str = "scene"):
    """Save edited scenes to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, scene in enumerate(scenes):
        output_path = output_dir / f"{prefix}_{i+1:03d}.jpg"
        scene.save(output_path, quality=95)

    print(f"✓ Saved {len(scenes)} scenes to {output_dir}")

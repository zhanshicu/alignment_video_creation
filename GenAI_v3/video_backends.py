"""
Video Manipulation Backends

Provides consistent video-level manipulation using cloud APIs.
Much better quality and consistency than frame-by-frame SDXL.

Supported backends:
- Google Veo 3.1 (via google-genai) - Recommended for Colab
- Runway Gen-3 (via API)
- Fallback: Consistent SDXL (local)
"""

import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, List
import numpy as np
import cv2
from PIL import Image


class VideoBackend(ABC):
    """Abstract base class for video manipulation backends."""

    @abstractmethod
    def manipulate_video_segment(
        self,
        video_segment: List[np.ndarray],
        mask: np.ndarray,
        action: Literal["increase", "decrease"],
        fps: float,
    ) -> List[np.ndarray]:
        """
        Manipulate an entire video segment consistently.

        Args:
            video_segment: List of RGB frames (np.ndarray)
            mask: Binary mask where 1=product (keep), 0=background (edit)
            action: "increase" (plain bg) or "decrease" (vibrant bg)
            fps: Frames per second

        Returns:
            List of manipulated frames
        """
        pass


class GoogleVeoBackend(VideoBackend):
    """
    Google Veo 3.1 backend via google-genai.

    Uses Google's state-of-the-art video generation model for
    consistent video manipulation.

    Requires:
        pip install google-genai

    Setup on Colab:
        from google.colab import auth
        auth.authenticate_user()
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google Veo backend.

        Args:
            api_key: Google API key (optional, uses default auth on Colab)
        """
        self.api_key = api_key
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client."""
        try:
            from google import genai

            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
            else:
                # Uses default credentials (works on Colab after auth)
                self.client = genai.Client()

            print("✓ Initialized Google GenAI client (Veo 3.1)")

        except ImportError:
            raise ImportError(
                "google-genai required. Install: pip install google-genai"
            )

    def manipulate_video_segment(
        self,
        video_segment: List[np.ndarray],
        mask: np.ndarray,
        action: Literal["increase", "decrease"],
        fps: float,
    ) -> List[np.ndarray]:
        """
        Manipulate video using Google Veo 3.1.

        Strategy:
        1. Take reference frame from video
        2. Generate new background using Gemini image generation
        3. Generate video from that image using Veo 3.1
        4. Extract frames and composite with original product
        """
        print(f"  Using Google Veo 3.1 for {len(video_segment)} frames...")

        # Get reference frame (middle of segment)
        ref_idx = len(video_segment) // 2
        ref_frame = video_segment[ref_idx]
        h, w = ref_frame.shape[:2]

        # Resize mask
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Get prompt for background generation
        prompt = self._get_prompt(action)

        try:
            # Step 1: Generate background image with Gemini
            print("  Generating reference background with Gemini...")
            bg_image = self._generate_background_image(ref_frame, mask_resized, prompt)

            # Step 2: Generate video with Veo 3.1 from the background image
            print("  Generating video with Veo 3.1...")
            generated_frames = self._generate_video_from_image(
                bg_image,
                prompt,
                target_frames=len(video_segment),
                fps=fps,
            )

            # Step 3: Composite - keep product from original, use generated background
            print("  Compositing with original product...")
            result_frames = self._composite_frames(
                original_frames=video_segment,
                generated_frames=generated_frames,
                product_mask=mask_resized,
            )

            print(f"  ✓ Generated {len(result_frames)} consistent frames")
            return result_frames

        except Exception as e:
            print(f"  ⚠ Veo generation failed: {e}")
            print("  Falling back to consistent background mode...")
            return self._fallback_consistent(video_segment, mask_resized, action)

    def _get_prompt(self, action: str) -> str:
        """Get prompt based on action."""
        if action == "increase":
            return (
                "A completely plain, solid neutral gray studio backdrop. "
                "Extremely simple, flat, featureless background. "
                "Professional product photography background with no patterns, "
                "no textures, no gradients - just uniform gray. "
                "Static camera, no movement."
            )
        else:  # decrease
            return (
                "An extremely vibrant, colorful, eye-catching background. "
                "Bold colors, interesting patterns, dynamic lighting effects. "
                "Visually engaging environment with rich textures. "
                "Rainbow gradients, neon colors, kaleidoscope patterns. "
                "Subtle camera movement for dynamic feel."
            )

    def _generate_background_image(
        self,
        ref_frame: np.ndarray,
        mask: np.ndarray,
        prompt: str,
    ) -> Image.Image:
        """Generate a background image using Gemini."""
        # Create a version of the frame with masked product
        # This gives context about the scene

        # Generate image with Gemini
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=f"Generate an image: {prompt}",
            config={"response_modalities": ["IMAGE"]}
        )

        # Extract the generated image
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    import io
                    image_data = part.inline_data.data
                    return Image.open(io.BytesIO(image_data))

        raise RuntimeError("Failed to generate background image with Gemini")

    def _generate_video_from_image(
        self,
        image: Image.Image,
        prompt: str,
        target_frames: int,
        fps: float,
    ) -> List[np.ndarray]:
        """Generate video from image using Veo 3.1."""
        # Save image to temp file for upload
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            image.save(temp_path)

        try:
            # Upload the image
            from google.genai import types

            # Read image as bytes
            with open(temp_path, "rb") as f:
                image_bytes = f.read()

            # Create image part
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            )

            # Generate video with Veo 3.1
            operation = self.client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=image_part,
            )

            # Poll until complete
            while not operation.done:
                print("    Waiting for Veo generation...")
                time.sleep(10)
                operation = self.client.operations.get(operation)

            # Download the generated video
            video = operation.response.generated_videos[0]

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name
                self.client.files.download(file=video.video)
                video.video.save(video_path)

            # Extract frames from generated video
            frames = self._extract_frames(video_path, target_frames)

            # Clean up
            os.unlink(video_path)

            return frames

        finally:
            os.unlink(temp_path)

    def _extract_frames(self, video_path: str, target_frames: int) -> List[np.ndarray]:
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        # Resample to match target frame count
        if len(frames) != target_frames:
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]

        return frames

    def _composite_frames(
        self,
        original_frames: List[np.ndarray],
        generated_frames: List[np.ndarray],
        product_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """Composite original product with generated background."""
        result = []
        h, w = original_frames[0].shape[:2]
        mask_3ch = np.stack([product_mask] * 3, axis=-1)

        for orig, gen in zip(original_frames, generated_frames):
            # Resize generated frame if needed
            if gen.shape[:2] != (h, w):
                gen = cv2.resize(gen, (w, h))

            # Composite: product from original, background from generated
            composite = (
                mask_3ch * orig.astype(float) +
                (1 - mask_3ch) * gen.astype(float)
            ).astype(np.uint8)

            result.append(composite)

        return result

    def _fallback_consistent(
        self,
        video_segment: List[np.ndarray],
        mask: np.ndarray,
        action: str,
    ) -> List[np.ndarray]:
        """Fallback: Generate one background, apply to all frames."""
        prompt = self._get_prompt(action)

        # Generate single background image
        ref_frame = video_segment[len(video_segment) // 2]
        bg_image = self._generate_background_image(ref_frame, mask, prompt)
        bg_array = np.array(bg_image)

        h, w = ref_frame.shape[:2]
        if bg_array.shape[:2] != (h, w):
            bg_array = cv2.resize(bg_array, (w, h))

        # Apply to all frames
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = []

        for frame in video_segment:
            composite = (
                mask_3ch * frame.astype(float) +
                (1 - mask_3ch) * bg_array.astype(float)
            ).astype(np.uint8)
            result.append(composite)

        return result


class RunwayBackend(VideoBackend):
    """
    Runway Gen-3 backend via API.

    Provides high-quality video manipulation using Runway's
    video generation and editing capabilities.

    Requires:
        pip install runwayml
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Runway backend.

        Args:
            api_key: Runway API key (or set RUNWAY_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("RUNWAY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Runway API key required. Set RUNWAY_API_KEY env var "
                "or pass api_key explicitly."
            )
        self._init_client()

    def _init_client(self):
        """Initialize Runway client."""
        try:
            from runwayml import RunwayML
            self.client = RunwayML(api_key=self.api_key)
            print("✓ Initialized Runway API")
        except ImportError:
            raise ImportError(
                "runwayml required. Install: pip install runwayml"
            )

    def manipulate_video_segment(
        self,
        video_segment: List[np.ndarray],
        mask: np.ndarray,
        action: Literal["increase", "decrease"],
        fps: float,
    ) -> List[np.ndarray]:
        """Manipulate video using Runway Gen-3."""
        print(f"  Using Runway Gen-3 for {len(video_segment)} frames...")

        # Similar approach: generate background, composite
        # Implementation depends on Runway's specific API

        raise NotImplementedError(
            "Runway backend not fully implemented. "
            "Use 'google' or 'consistent' backend instead."
        )


class ConsistentBackgroundBackend(VideoBackend):
    """
    Simple consistent background replacement using local SDXL.

    Key insight:
    - Generate ONE background image
    - Apply to ALL frames
    - No temporal inconsistency!

    Uses SDXL only once per scene, not per frame.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize with SDXL for single background generation."""
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load SDXL for background generation."""
        import torch
        from diffusers import AutoPipelineForInpainting

        print("Loading SDXL Inpainting (single generation mode)...")
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)

        self.pipe.enable_attention_slicing()
        print("✓ SDXL loaded")

    def manipulate_video_segment(
        self,
        video_segment: List[np.ndarray],
        mask: np.ndarray,
        action: Literal["increase", "decrease"],
        fps: float,
    ) -> List[np.ndarray]:
        """
        Apply consistent background to entire video segment.

        Key: Generate background ONCE, apply to ALL frames.
        """
        import torch

        print(f"  Generating single consistent background for {len(video_segment)} frames...")

        # Use middle frame as reference
        ref_idx = len(video_segment) // 2
        ref_frame = video_segment[ref_idx]
        h, w = ref_frame.shape[:2]

        # Resize mask
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        bg_mask = 1.0 - mask_resized  # Background = 1, product = 0

        # Prepare for SDXL
        ref_image = Image.fromarray(ref_frame).resize((1024, 1024), Image.LANCZOS)
        bg_mask_1024 = cv2.resize(bg_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Expand mask slightly for better blending
        kernel = np.ones((21, 21), np.uint8)
        bg_mask_expanded = cv2.dilate(bg_mask_1024.astype(np.uint8), kernel, iterations=2)
        mask_pil = Image.fromarray((bg_mask_expanded * 255).astype(np.uint8))

        # Get prompt
        if action == "increase":
            prompt = (
                "completely plain solid gray background, "
                "extremely out of focus, heavily blurred, "
                "no details whatsoever, pure muted tones, "
                "flat featureless backdrop, studio gray seamless"
            )
            negative = "detailed, colorful, vibrant, patterns, textures"
            guidance = 12.0
        else:
            prompt = (
                "extremely vibrant colorful background, "
                "bold patterns, neon colors, rainbow gradients, "
                "eye-catching detailed textures, visually exciting"
            )
            negative = "plain, simple, muted, boring, gray"
            guidance = 15.0

        # Generate ONE background
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=ref_image,
                mask_image=mask_pil,
                num_inference_steps=40,
                strength=0.95,
                guidance_scale=guidance,
            ).images[0]

        # Resize back
        new_bg = np.array(result.resize((w, h), Image.LANCZOS))
        print("  ✓ Generated reference background")

        # Apply to ALL frames consistently
        result_frames = []
        mask_3ch = np.stack([mask_resized] * 3, axis=-1)

        for frame in video_segment:
            composite = (
                mask_3ch * frame.astype(float) +
                (1 - mask_3ch) * new_bg.astype(float)
            ).astype(np.uint8)
            result_frames.append(composite)

        print(f"  ✓ Applied consistent background to {len(result_frames)} frames")
        return result_frames


def get_backend(
    backend_type: Literal["google", "runway", "consistent", "auto"] = "auto",
    **kwargs,
) -> VideoBackend:
    """
    Get the appropriate video backend.

    Args:
        backend_type:
            - "google": Google Veo 3.1 via google-genai
            - "runway": Runway Gen-3 API
            - "consistent": Local SDXL with consistent background
            - "auto": Try Google first, fall back to consistent

    Returns:
        VideoBackend instance
    """
    if backend_type == "auto":
        # Try Google first (best for Colab)
        try:
            return GoogleVeoBackend(**kwargs)
        except Exception as e:
            print(f"Google backend unavailable: {e}")
            print("Falling back to consistent background mode...")
            return ConsistentBackgroundBackend(**kwargs)

    elif backend_type == "google":
        return GoogleVeoBackend(**kwargs)

    elif backend_type == "runway":
        return RunwayBackend(**kwargs)

    elif backend_type == "consistent":
        return ConsistentBackgroundBackend(**kwargs)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

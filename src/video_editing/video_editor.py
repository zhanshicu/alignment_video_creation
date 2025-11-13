"""
Video Editor

Frame-level editing with ControlNet conditioning.
"""

import os
import numpy as np
import torch
from typing import List, Optional, Dict, Union
from PIL import Image
import cv2
from tqdm import tqdm

from ..models.stable_diffusion_wrapper import StableDiffusionControlNetWrapper


class VideoEditor:
    """Frame-level video editor using SD+ControlNet."""

    def __init__(
        self,
        model: StableDiffusionControlNetWrapper,
        device: str = "cuda",
    ):
        """
        Initialize video editor.

        Args:
            model: Trained SD+ControlNet model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()

    def preprocess_frame(
        self,
        frame: np.ndarray,
        target_size: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Preprocess frame for model input.

        Args:
            frame: Frame as numpy array (H, W, 3) in [0, 255]
            target_size: Optional target (H, W) for resizing

        Returns:
            Tensor (1, 3, H, W) in [-1, 1]
        """
        # Convert to PIL
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        img = Image.fromarray(frame)

        # Resize if needed
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

        # To numpy and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array * 2.0) - 1.0  # To [-1, 1]

        # To tensor
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def postprocess_frame(
        self,
        tensor: torch.Tensor,
        original_size: Optional[tuple] = None
    ) -> np.ndarray:
        """
        Postprocess model output to frame.

        Args:
            tensor: Output tensor (1, 3, H, W) in [-1, 1]
            original_size: Optional original (H, W) for resizing back

        Returns:
            Frame as numpy array (H, W, 3) in [0, 255]
        """
        # To numpy
        img = tensor[0].cpu().permute(1, 2, 0).numpy()

        # Denormalize
        img = (img + 1.0) / 2.0  # To [0, 1]
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        # Resize if needed
        if original_size is not None:
            img = cv2.resize(img, (original_size[1], original_size[0]))

        return img

    @torch.no_grad()
    def edit_frame(
        self,
        frame: np.ndarray,
        control_tensor: Union[np.ndarray, torch.Tensor],
        keyword: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
    ) -> np.ndarray:
        """
        Edit a single frame with ControlNet conditioning.

        Args:
            frame: Input frame (H, W, 3) in [0, 255]
            control_tensor: Control tensor (C, H, W) or (1, C, H, W)
            keyword: Text prompt
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            strength: Edit strength (0=no change, 1=full edit)

        Returns:
            Edited frame (H, W, 3) in [0, 255]
        """
        original_size = frame.shape[:2]

        # Preprocess
        frame_tensor = self.preprocess_frame(frame)

        # Prepare control tensor
        if isinstance(control_tensor, np.ndarray):
            control_tensor = torch.from_numpy(control_tensor).to(self.device)
        if control_tensor.ndim == 3:
            control_tensor = control_tensor.unsqueeze(0)

        # Generate
        edited_tensor = self.model.generate(
            control_tensor=control_tensor,
            text_prompt=keyword,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            reference_image=frame_tensor,
            strength=strength,
        )

        # Postprocess
        edited_frame = self.postprocess_frame(edited_tensor, original_size)

        return edited_frame

    def edit_video_frames(
        self,
        frames: List[np.ndarray],
        control_tensors: List[Union[np.ndarray, torch.Tensor]],
        keyword: str,
        frame_indices: Optional[List[int]] = None,
        **edit_kwargs
    ) -> List[np.ndarray]:
        """
        Edit multiple video frames.

        Args:
            frames: List of frames
            control_tensors: List of control tensors (one per frame)
            keyword: Text prompt
            frame_indices: Optional list of frame indices to edit (None = edit all)
            **edit_kwargs: Additional arguments for edit_frame

        Returns:
            List of edited frames
        """
        if frame_indices is None:
            frame_indices = list(range(len(frames)))

        edited_frames = []

        for i, (frame, control) in enumerate(tqdm(
            zip(frames, control_tensors),
            total=len(frames),
            desc="Editing frames"
        )):
            if i in frame_indices:
                # Edit this frame
                edited = self.edit_frame(frame, control, keyword, **edit_kwargs)
            else:
                # Keep original
                edited = frame.copy()

            edited_frames.append(edited)

        return edited_frames

    def edit_video_file(
        self,
        input_video_path: str,
        output_video_path: str,
        control_tensors: List[Union[np.ndarray, torch.Tensor]],
        keyword: str,
        fps: Optional[float] = None,
        **edit_kwargs
    ):
        """
        Edit a video file.

        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video
            control_tensors: List of control tensors (one per frame)
            keyword: Text prompt
            fps: Optional FPS (auto-detected if None)
            **edit_kwargs: Additional arguments for edit_frame
        """
        # Load video
        cap = cv2.VideoCapture(input_video_path)
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        print(f"Loaded {len(frames)} frames from {input_video_path}")

        # Ensure we have enough control tensors
        assert len(control_tensors) == len(frames), \
            f"Need {len(frames)} control tensors, got {len(control_tensors)}"

        # Edit frames
        edited_frames = self.edit_video_frames(
            frames, control_tensors, keyword, **edit_kwargs
        )

        # Save video
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame in edited_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"Saved edited video to {output_video_path}")

    def edit_keyframes_only(
        self,
        frames: List[np.ndarray],
        control_tensors: List[Union[np.ndarray, torch.Tensor]],
        keyword: str,
        keyframe_interval: int = 4,
        **edit_kwargs
    ) -> Dict[int, np.ndarray]:
        """
        Edit only keyframes (for use with temporal propagation).

        Args:
            frames: List of frames
            control_tensors: List of control tensors
            keyword: Text prompt
            keyframe_interval: Edit every N-th frame
            **edit_kwargs: Additional arguments for edit_frame

        Returns:
            Dict mapping frame index to edited frame
        """
        keyframe_indices = list(range(0, len(frames), keyframe_interval))

        edited_keyframes = {}

        for i in tqdm(keyframe_indices, desc="Editing keyframes"):
            edited = self.edit_frame(
                frames[i],
                control_tensors[i],
                keyword,
                **edit_kwargs
            )
            edited_keyframes[i] = edited

        return edited_keyframes

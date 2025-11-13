"""
Keyword/Product Heatmap Generator

Uses CLIPSeg to generate keyword localization heatmaps.
"""

import os
import numpy as np
from PIL import Image
from typing import Union, List, Optional
import torch
from torch import nn
import cv2


class KeywordHeatmapGenerator:
    """Generates keyword/product heatmaps using CLIPSeg."""

    def __init__(
        self,
        model_name: str = "CIDAS/clipseg-rd64-refined",
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the keyword heatmap generator.

        Args:
            model_name: HuggingFace model identifier for CLIPSeg
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            threshold: Threshold for creating binary mask (0-1)
        """
        self.model_name = model_name
        self.threshold = threshold

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._load_model()

    def _load_model(self):
        """Load CLIPSeg model and processor."""
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        except ImportError:
            raise ImportError(
                "transformers package is required. "
                "Install with: pip install transformers"
            )

        self.processor = CLIPSegProcessor.from_pretrained(self.model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Set normalization
        self.processor.image_processor.image_mean = IMAGENET_DEFAULT_MEAN
        self.processor.image_processor.image_std = IMAGENET_DEFAULT_STD

    def generate_keyword_heatmap(
        self,
        image: Union[np.ndarray, Image.Image],
        keyword: str,
        return_binary: bool = False
    ) -> np.ndarray:
        """
        Generate keyword heatmap for a single image.

        Args:
            image: Input image as numpy array (H, W, 3) or PIL Image
            keyword: Text prompt describing the keyword/product
            return_binary: If True, return binary mask; else return continuous heatmap

        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Prepare inputs
        prompts = [keyword]
        inputs = self.processor(
            text=prompts,
            images=[pil_image],
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get predictions and resize
        preds = nn.functional.interpolate(
            outputs.logits.unsqueeze(1),
            size=(pil_image.size[1], pil_image.size[0]),
            mode="bilinear"
        )

        # Convert to numpy
        heatmap = preds[0][0].cpu().numpy()

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        if return_binary:
            _, heatmap = cv2.threshold(heatmap, self.threshold, 1.0, cv2.THRESH_BINARY)

        return heatmap

    def generate_keyword_mask(
        self,
        image: Union[np.ndarray, Image.Image],
        keyword: str,
        soften: bool = False,
        soften_sigma: float = 3.0
    ) -> np.ndarray:
        """
        Generate keyword mask M_t.

        Args:
            image: Input image
            keyword: Text prompt
            soften: If True, apply Gaussian smoothing to create soft mask
            soften_sigma: Sigma for Gaussian smoothing

        Returns:
            Binary or soft mask (H, W) with values in [0, 1]
        """
        mask = self.generate_keyword_heatmap(image, keyword, return_binary=True)

        if soften:
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask.astype(np.float32), sigma=soften_sigma)
            mask = np.clip(mask, 0, 1)

        return mask

    def batch_generate_heatmaps(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        keyword: str,
        return_binary: bool = False
    ) -> List[np.ndarray]:
        """
        Generate keyword heatmaps for multiple images.

        Args:
            images: List of input images
            keyword: Text prompt (same for all images)
            return_binary: Whether to return binary masks

        Returns:
            List of heatmaps
        """
        heatmaps = []
        for image in images:
            heatmap = self.generate_keyword_heatmap(
                image, keyword, return_binary=return_binary
            )
            heatmaps.append(heatmap)
        return heatmaps

    def save_heatmap(
        self,
        heatmap: np.ndarray,
        save_path: str,
        colormap: str = 'gray'
    ):
        """
        Save heatmap to file.

        Args:
            heatmap: Heatmap array (H, W)
            save_path: Output file path
            colormap: Matplotlib colormap name
        """
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, heatmap, cmap=colormap)

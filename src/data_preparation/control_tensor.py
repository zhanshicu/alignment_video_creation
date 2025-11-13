"""
Control Tensor Builder

Constructs control tensors C_t = [M_t, S_t] or [M_t, S_t, A_t, K_t]
from attention and keyword heatmaps.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import torch


class ControlTensorBuilder:
    """Builds control tensors for conditioning the generative model."""

    def __init__(
        self,
        include_raw_maps: bool = False,
        normalize: bool = True,
    ):
        """
        Initialize control tensor builder.

        Args:
            include_raw_maps: If True, include A_t and K_t in control tensor
            normalize: If True, normalize all maps to [0, 1]
        """
        self.include_raw_maps = include_raw_maps
        self.normalize = normalize

    def compute_alignment_map(
        self,
        attention_map: np.ndarray,
        keyword_map: np.ndarray
    ) -> np.ndarray:
        """
        Compute instantaneous alignment map S_t = normalize(A_t ⊙ K_t).

        Args:
            attention_map: Attention heatmap A_t (H, W)
            keyword_map: Keyword heatmap K_t (H, W)

        Returns:
            Alignment map S_t (H, W) normalized to [0, 1]
        """
        # Element-wise multiplication
        alignment = attention_map * keyword_map

        # Normalize
        if self.normalize:
            alignment = self._normalize_map(alignment)

        return alignment

    def compute_keyword_mask(
        self,
        keyword_map: np.ndarray,
        threshold: float = 0.5,
        soften: bool = False,
        soften_sigma: float = 3.0
    ) -> np.ndarray:
        """
        Compute keyword mask M_t from keyword heatmap.

        Args:
            keyword_map: Keyword heatmap K_t (H, W)
            threshold: Threshold for binarization
            soften: If True, apply Gaussian smoothing
            soften_sigma: Sigma for smoothing

        Returns:
            Keyword mask M_t (H, W) in [0, 1]
        """
        # Binarize
        mask = (keyword_map > threshold).astype(np.float32)

        # Optionally soften
        if soften:
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask, sigma=soften_sigma)
            mask = np.clip(mask, 0, 1)

        return mask

    def compute_background_mask(
        self,
        keyword_mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute background mask B_t = 1 - M_t.

        Args:
            keyword_mask: Keyword mask M_t (H, W)

        Returns:
            Background mask B_t (H, W)
        """
        return 1.0 - keyword_mask

    def build_control_tensor(
        self,
        attention_map: np.ndarray,
        keyword_map: np.ndarray,
        keyword_mask: Optional[np.ndarray] = None,
        alignment_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Build control tensor C_t.

        Args:
            attention_map: Attention heatmap A_t (H, W)
            keyword_map: Keyword heatmap K_t (H, W)
            keyword_mask: Optional pre-computed keyword mask M_t
            alignment_map: Optional pre-computed alignment map S_t

        Returns:
            Control tensor C_t (H, W, C) where C=2 or 4
        """
        # Compute mask if not provided
        if keyword_mask is None:
            keyword_mask = self.compute_keyword_mask(keyword_map)

        # Compute alignment if not provided
        if alignment_map is None:
            alignment_map = self.compute_alignment_map(attention_map, keyword_map)

        # Stack channels
        if self.include_raw_maps:
            # C_t = [M_t, S_t, A_t, K_t]
            control_tensor = np.stack([
                keyword_mask,
                alignment_map,
                attention_map,
                keyword_map
            ], axis=-1)
        else:
            # C_t = [M_t, S_t]
            control_tensor = np.stack([
                keyword_mask,
                alignment_map
            ], axis=-1)

        return control_tensor.astype(np.float32)

    def modify_alignment(
        self,
        alignment_map: np.ndarray,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Modify alignment map for experimental manipulation.

        S'_t = alpha * S_t

        Args:
            alignment_map: Original alignment map S_t
            alpha: Scaling factor (>1 for boost, <1 for reduction)

        Returns:
            Modified alignment map S'_t
        """
        modified = alignment_map * alpha
        return np.clip(modified, 0, 1)

    def create_experimental_variants(
        self,
        attention_map: np.ndarray,
        keyword_map: np.ndarray,
        alphas: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Create multiple experimental variants with different alignment levels.

        Args:
            attention_map: Attention heatmap A_t
            keyword_map: Keyword heatmap K_t
            alphas: Dict mapping variant name to alpha value

        Returns:
            Dict mapping variant name to control tensor
        """
        keyword_mask = self.compute_keyword_mask(keyword_map)
        base_alignment = self.compute_alignment_map(attention_map, keyword_map)

        variants = {}
        for name, alpha in alphas.items():
            modified_alignment = self.modify_alignment(base_alignment, alpha)
            control_tensor = self.build_control_tensor(
                attention_map,
                keyword_map,
                keyword_mask=keyword_mask,
                alignment_map=modified_alignment
            )
            variants[name] = control_tensor

        return variants

    def _normalize_map(self, map_array: np.ndarray) -> np.ndarray:
        """Normalize map to [0, 1]."""
        min_val = map_array.min()
        max_val = map_array.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(map_array)
        return (map_array - min_val) / (max_val - min_val)

    def to_torch(
        self,
        control_tensor: np.ndarray,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Convert control tensor to PyTorch tensor.

        Args:
            control_tensor: Numpy control tensor (H, W, C)
            device: Target device

        Returns:
            PyTorch tensor (1, C, H, W)
        """
        tensor = torch.from_numpy(control_tensor).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)

    def downsample_control_tensor(
        self,
        control_tensor: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Downsample control tensor for model input.

        Args:
            control_tensor: Control tensor (H, W, C)
            target_size: Target (H', W')

        Returns:
            Downsampled control tensor (H', W', C)
        """
        import cv2

        h, w, c = control_tensor.shape
        downsampled = np.zeros((*target_size, c), dtype=np.float32)

        for i in range(c):
            downsampled[:, :, i] = cv2.resize(
                control_tensor[:, :, i],
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return downsampled

    def calculate_alignment_score(
        self,
        attention_map: np.ndarray,
        keyword_mask: np.ndarray
    ) -> float:
        """
        Calculate alignment score (proportion of attention on keyword).

        Args:
            attention_map: Attention heatmap A_t
            keyword_mask: Keyword mask M_t

        Returns:
            Alignment score in [0, 1]
        """
        total_attention = np.sum(attention_map)
        if total_attention < 1e-8:
            return 0.0

        keyword_attention = np.sum(attention_map * keyword_mask)
        return keyword_attention / total_attention

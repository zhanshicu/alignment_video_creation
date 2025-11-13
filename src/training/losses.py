"""
Loss Functions for ControlNet Training

Implements:
- Diffusion loss: L_diff = ||ε̂ - ε||²
- Reconstruction loss: L_recon = ||Î_t - I_t||₁ + λ_LPIPS · LPIPS(Î_t, I_t)
- Background preservation: L_bg = λ_bg · ||(Î_t - I_t) ⊙ B_t||₁
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DiffusionLoss(nn.Module):
    """Standard diffusion denoising loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion loss.

        Args:
            noise_pred: Predicted noise (B, C, H, W)
            noise_target: Ground truth noise (B, C, H, W)

        Returns:
            Scalar loss
        """
        return F.mse_loss(noise_pred, noise_target)


class PerceptualLoss(nn.Module):
    """LPIPS perceptual loss."""

    def __init__(self, device: str = 'cuda'):
        super().__init__()
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: lpips not installed. Using L1 loss as fallback.")
            self.lpips_model = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [-1, 1]
            target: Target image (B, 3, H, W) in [-1, 1]

        Returns:
            Scalar loss
        """
        if self.lpips_model is not None:
            return self.lpips_model(pred, target).mean()
        else:
            # Fallback to L1
            return F.l1_loss(pred, target)


class ReconstructionLoss(nn.Module):
    """Image reconstruction loss with perceptual component."""

    def __init__(
        self,
        lambda_lpips: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize reconstruction loss.

        Args:
            lambda_lpips: Weight for LPIPS loss
            device: Device for LPIPS model
        """
        super().__init__()
        self.lambda_lpips = lambda_lpips
        self.perceptual_loss = PerceptualLoss(device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)

        Returns:
            Dict with 'total', 'l1', and 'lpips' losses
        """
        # L1 loss
        l1_loss = F.l1_loss(pred, target)

        # Perceptual loss
        lpips_loss = self.perceptual_loss(pred, target)

        # Total
        total = l1_loss + self.lambda_lpips * lpips_loss

        return {
            'total': total,
            'l1': l1_loss,
            'lpips': lpips_loss
        }


class BackgroundPreservationLoss(nn.Module):
    """Loss to preserve non-keyword regions."""

    def __init__(self, lambda_bg: float = 1.0):
        """
        Initialize background preservation loss.

        Args:
            lambda_bg: Weight for background preservation
        """
        super().__init__()
        self.lambda_bg = lambda_bg

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        background_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute background preservation loss.

        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            background_mask: Background mask B_t (B, 1, H, W) in [0, 1]

        Returns:
            Scalar loss
        """
        # Compute difference
        diff = torch.abs(pred - target)

        # Apply mask and average
        masked_diff = diff * background_mask
        loss = masked_diff.sum() / (background_mask.sum() + 1e-8)

        return self.lambda_bg * loss


class TotalLoss(nn.Module):
    """
    Total training loss combining all components.

    L = L_diff + λ_recon * L_recon + λ_bg * L_bg
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_lpips: float = 1.0,
        lambda_bg: float = 0.5,
        device: str = 'cuda',
        use_recon_loss: bool = True,
    ):
        """
        Initialize total loss.

        Args:
            lambda_recon: Weight for reconstruction loss
            lambda_lpips: Weight for LPIPS within reconstruction
            lambda_bg: Weight for background preservation
            device: Device for models
            use_recon_loss: Whether to include reconstruction loss (requires full denoising)
        """
        super().__init__()

        self.lambda_recon = lambda_recon
        self.lambda_bg = lambda_bg
        self.use_recon_loss = use_recon_loss

        self.diffusion_loss = DiffusionLoss()
        self.reconstruction_loss = ReconstructionLoss(lambda_lpips, device)
        self.background_loss = BackgroundPreservationLoss(lambda_bg)

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        pred_image: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        background_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            noise_pred: Predicted noise from diffusion model
            noise_target: Ground truth noise
            pred_image: Optional predicted image (if using recon loss)
            target_image: Optional target image
            background_mask: Optional background mask B_t

        Returns:
            Dict with loss components and total
        """
        losses = {}

        # Diffusion loss (always computed)
        l_diff = self.diffusion_loss(noise_pred, noise_target)
        losses['diffusion'] = l_diff
        total = l_diff

        # Reconstruction loss (optional, requires full denoising)
        if self.use_recon_loss and pred_image is not None and target_image is not None:
            recon_losses = self.reconstruction_loss(pred_image, target_image)
            losses['recon_total'] = recon_losses['total']
            losses['recon_l1'] = recon_losses['l1']
            losses['recon_lpips'] = recon_losses['lpips']
            total = total + self.lambda_recon * recon_losses['total']

            # Background preservation (requires reconstruction)
            if background_mask is not None:
                l_bg = self.background_loss(pred_image, target_image, background_mask)
                losses['background'] = l_bg
                total = total + l_bg

        losses['total'] = total
        return losses


class SyntheticEnhancementLoss(nn.Module):
    """
    Loss for Stage 1.5: Synthetic Keyword Enhancement

    Trains model to apply simple enhancements to keyword regions.
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_lpips: float = 1.0,
        device: str = 'cuda'
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.reconstruction_loss = ReconstructionLoss(lambda_lpips, device)
        self.diffusion_loss = DiffusionLoss()

    def apply_synthetic_enhancement(
        self,
        image: torch.Tensor,
        keyword_mask: torch.Tensor,
        enhancement_type: str = 'brightness',
        strength: float = 0.2
    ) -> torch.Tensor:
        """
        Apply synthetic enhancement to keyword region.

        Args:
            image: Input image (B, 3, H, W) in [-1, 1]
            keyword_mask: Keyword mask (B, 1, H, W)
            enhancement_type: Type of enhancement ('brightness', 'contrast', 'saturation')
            strength: Enhancement strength

        Returns:
            Enhanced image
        """
        enhanced = image.clone()

        if enhancement_type == 'brightness':
            # Increase brightness in keyword region
            adjustment = keyword_mask * strength
            enhanced = enhanced + adjustment
        elif enhancement_type == 'contrast':
            # Increase contrast
            mean = enhanced.mean(dim=1, keepdim=True)
            enhanced = enhanced + keyword_mask * (enhanced - mean) * strength
        elif enhancement_type == 'saturation':
            # Increase saturation
            gray = enhanced.mean(dim=1, keepdim=True)
            enhanced = enhanced + keyword_mask * (enhanced - gray) * strength

        # Clamp to valid range
        enhanced = torch.clamp(enhanced, -1, 1)

        return enhanced

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        pred_image: torch.Tensor,
        original_image: torch.Tensor,
        keyword_mask: torch.Tensor,
        enhancement_type: str = 'brightness'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute enhancement training loss.

        Args:
            noise_pred: Predicted noise
            noise_target: Target noise
            pred_image: Predicted reconstructed image
            original_image: Original image
            keyword_mask: Keyword mask
            enhancement_type: Type of enhancement to apply

        Returns:
            Dict with loss components
        """
        # Generate synthetic target
        enhanced_target = self.apply_synthetic_enhancement(
            original_image, keyword_mask, enhancement_type
        )

        # Diffusion loss
        l_diff = self.diffusion_loss(noise_pred, noise_target)

        # Reconstruction against enhanced target
        recon_losses = self.reconstruction_loss(pred_image, enhanced_target)

        total = l_diff + self.lambda_recon * recon_losses['total']

        return {
            'total': total,
            'diffusion': l_diff,
            'recon_total': recon_losses['total'],
            'recon_l1': recon_losses['l1'],
            'recon_lpips': recon_losses['lpips'],
        }

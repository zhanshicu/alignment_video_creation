"""
ControlNet-Style Adapter

Lightweight U-Net that processes control tensors and injects features
into Stable Diffusion via zero-initialized convolutions.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


class ZeroConv(nn.Module):
    """Zero-initialized convolution layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        # Initialize with zeros
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ControlNetBlock(nn.Module):
    """Single ControlNet encoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_attention: bool = False
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()

        # Residual connection
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.downsample = None

        # Optional self-attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                out_channels, num_heads=8, batch_first=True
            )
            self.norm_attn = nn.GroupNorm(32, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Conv block 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        # Conv block 2
        out = self.conv2(out)
        out = self.norm2(out)

        # Residual
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.act2(out)

        # Optional attention
        if self.use_attention:
            b, c, h, w = out.shape
            out_flat = out.view(b, c, h * w).permute(0, 2, 1)
            attn_out, _ = self.attention(out_flat, out_flat, out_flat)
            attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)
            out = out + self.norm_attn(attn_out)

        return out


class ControlNetAdapter(nn.Module):
    """
    ControlNet adapter for injecting control signals into Stable Diffusion.

    Takes control tensor C_t and produces multi-scale features that are
    injected into SD U-Net layers via zero-initialized convolutions.
    """

    def __init__(
        self,
        control_channels: int = 2,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (8,),
        sd_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
    ):
        """
        Initialize ControlNet adapter.

        Args:
            control_channels: Number of channels in control tensor (2 or 4)
            base_channels: Base number of channels
            channel_mult: Channel multipliers for each resolution
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions at which to use attention
            sd_channels: Channel dimensions of SD U-Net blocks for zero conv matching
        """
        super().__init__()

        self.control_channels = control_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks

        # Input convolution
        self.input_conv = nn.Conv2d(
            control_channels,
            base_channels,
            3,
            padding=1
        )

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.zero_convs = nn.ModuleList()

        current_channels = base_channels
        for i, mult in enumerate(channel_mult):
            out_channels = base_channels * mult
            use_attention = (2 ** i) in attention_resolutions

            for _ in range(num_res_blocks):
                block = ControlNetBlock(
                    current_channels,
                    out_channels,
                    stride=1,
                    use_attention=use_attention
                )
                self.encoder_blocks.append(block)
                current_channels = out_channels

            # Downsample (except last level)
            if i < len(channel_mult) - 1:
                downsample_block = ControlNetBlock(
                    current_channels,
                    current_channels,
                    stride=2,
                    use_attention=False
                )
                self.encoder_blocks.append(downsample_block)

        # Zero convolutions for injection
        # Match SD U-Net block dimensions
        for sd_ch in sd_channels:
            # Find closest matching encoder channel
            zero_conv = ZeroConv(current_channels, sd_ch)
            self.zero_convs.append(zero_conv)

        # Middle block
        self.middle_block = nn.Sequential(
            ControlNetBlock(current_channels, current_channels, stride=1, use_attention=True),
            ControlNetBlock(current_channels, current_channels, stride=1, use_attention=False),
        )

    def forward(
        self,
        control_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Forward pass through ControlNet adapter.

        Args:
            control_tensor: Control tensor (B, C, H, W)

        Returns:
            List of multi-scale features for injection into SD U-Net
        """
        # Input projection
        x = self.input_conv(control_tensor)

        # Collect features at different scales
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)

        # Middle block
        x = self.middle_block(x)
        features.append(x)

        # Apply zero convolutions
        zero_features = []
        for i, zero_conv in enumerate(self.zero_convs):
            if i < len(features):
                zero_features.append(zero_conv(features[i]))
            else:
                # Pad with zeros if needed
                zero_features.append(torch.zeros_like(zero_features[-1]))

        return zero_features

    def get_trainable_parameters(self):
        """Get trainable parameters (all except zero conv initial state)."""
        return self.parameters()


class ControlNetUNetWrapper(nn.Module):
    """
    Wrapper that combines SD U-Net with ControlNet adapter.

    Injects ControlNet features into SD U-Net blocks.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        controlnet: ControlNetAdapter,
        freeze_unet: bool = True
    ):
        """
        Initialize wrapper.

        Args:
            unet: Stable Diffusion U-Net (frozen)
            controlnet: ControlNet adapter (trainable)
            freeze_unet: Whether to freeze U-Net weights
        """
        super().__init__()

        self.unet = unet
        self.controlnet = controlnet

        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with ControlNet injection.

        Args:
            sample: Noisy latent (B, 4, H, W)
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings
            control_tensor: Control tensor (B, C, H', W')
            **kwargs: Additional arguments for U-Net

        Returns:
            Noise prediction
        """
        # Get ControlNet features
        control_features = self.controlnet(control_tensor)

        # Forward through U-Net with feature injection
        # Note: This is a simplified version. In practice, you would need to
        # modify the U-Net forward pass to accept and inject these features
        # at the appropriate layers.

        # For now, we'll use a hook-based approach
        injection_hooks = []

        def create_hook(feature):
            def hook(module, input, output):
                # Add control feature to output
                if isinstance(output, tuple):
                    output = (output[0] + feature[:, :, :output[0].shape[2], :output[0].shape[3]],) + output[1:]
                else:
                    output = output + feature[:, :, :output.shape[2], :output.shape[3]]
                return output
            return hook

        # Register hooks for injection
        # This is a placeholder - actual implementation would need to match
        # specific U-Net layers with control features
        blocks_to_inject = list(self.unet.down_blocks) + [self.unet.mid_block]

        for i, (block, feature) in enumerate(zip(blocks_to_inject[:len(control_features)], control_features)):
            hook = block.register_forward_hook(create_hook(feature))
            injection_hooks.append(hook)

        # Forward pass
        try:
            output = self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs
            )
        finally:
            # Remove hooks
            for hook in injection_hooks:
                hook.remove()

        return output

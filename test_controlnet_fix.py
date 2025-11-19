"""Test ControlNet dimension fix."""

import torch
import sys
sys.path.insert(0, '/home/user/alignment_video_creation')

from src.models.controlnet_adapter import ControlNetAdapter

def test_controlnet():
    """Test that ControlNet doesn't cause OOM with fixed attention resolutions."""

    print("Testing ControlNet with fixed attention resolutions...")

    # Create ControlNet with default (fixed) parameters
    controlnet = ControlNetAdapter(
        control_channels=2,
        base_channels=64,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(8,),  # Only use attention at 8x downsampled (64x64)
    )

    # Move to CPU for testing (or GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    controlnet = controlnet.to(device)

    # Create test input: batch=2, channels=2, height=512, width=512
    control_input = torch.randn(2, 2, 512, 512).to(device)

    print(f"Input shape: {control_input.shape}")
    print(f"Device: {device}")

    # Forward pass
    try:
        with torch.no_grad():
            outputs = controlnet(control_input)

        print(f"✓ Forward pass successful!")
        print(f"  Number of output features: {len(outputs)}")
        print(f"  Output shapes: {[o.shape for o in outputs]}")

        # Check memory usage
        if device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU memory allocated: {allocated:.2f} GB")
            print(f"  GPU memory reserved: {reserved:.2f} GB")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

if __name__ == '__main__':
    success = test_controlnet()
    if success:
        print("\n✓ ControlNet fix verified!")
    else:
        print("\n✗ ControlNet still has issues")
        sys.exit(1)

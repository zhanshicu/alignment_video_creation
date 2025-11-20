"""
GenAI v3: Zero-Shot Attention-Keyword Alignment Manipulation

No training required! Uses pre-trained generative models (InstructPix2Pix,
Stable Diffusion Inpainting) to manipulate attention-keyword alignment.

Key Features:
- Zero training - immediate results
- Fast inference - 2-3s per frame
- Simple API - easy to use
- Flexible control - adjust via text instructions
"""

from .zero_shot_manipulator import (
    ZeroShotAlignmentManipulator,
    load_scenes_from_paths,
    load_masks_from_paths,
    save_scenes,
)

from .temporal_smoother import (
    TemporalSmoother,
    apply_temporal_consistency_filter,
    create_crossfade,
)

__version__ = "1.0.0"
__all__ = [
    "ZeroShotAlignmentManipulator",
    "TemporalSmoother",
    "load_scenes_from_paths",
    "load_masks_from_paths",
    "save_scenes",
    "apply_temporal_consistency_filter",
    "create_crossfade",
]

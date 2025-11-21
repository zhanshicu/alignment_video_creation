"""
GenAI v3: Zero-Shot Background Manipulation for Attention Control

No training required! Uses SOTA pre-trained models (SDXL Inpainting)
to manipulate BACKGROUND to control attention on product.

Key Concept:
- Product stays UNCHANGED (authentic)
- Background is modified to increase/decrease attention on product
- "increase": Make background less distracting → attention on product
- "decrease": Make background more interesting → attention diverts

Usage:
    from GenAI_v3 import SceneManipulator

    manipulator = SceneManipulator()

    output = manipulator.manipulate(
        video_id="123456",
        scene_index=6,
        action="increase",
    )
"""

from .scene_manipulator import SceneManipulator

__version__ = "2.0.0"
__all__ = ["SceneManipulator"]

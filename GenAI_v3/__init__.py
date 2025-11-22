"""
GenAI v3: Automatic Background Manipulation for Attention Control

No training required! Uses SOTA pre-trained models:
- SDXL Inpainting for background manipulation
- SAM (Segment Anything) + DINO for automatic product detection

Key Concept:
- Product is AUTO-DETECTED using episodic memory (SAM + DINO)
- No keyword mask required!
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

try:
    from .product_detector import MainProductDetector
except ImportError:
    MainProductDetector = None

__version__ = "3.0.0"
__all__ = ["SceneManipulator", "MainProductDetector"]

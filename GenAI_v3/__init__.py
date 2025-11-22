"""
GenAI v3: Full Scene Background Manipulation for Attention Control

No training required! Uses SOTA pre-trained models:
- SDXL Inpainting for background manipulation
- SAM (Segment Anything) + DINO for automatic product detection
- PySceneDetect for real scene boundaries

Key Features:
- Product is AUTO-DETECTED using episodic memory (SAM + DINO)
- ENTIRE SCENES are manipulated (not just single frames)
- DRAMATIC background changes (not subtle)
- Smooth video output with keyframe interpolation
- Outputs both video AND frame

Usage:
    from GenAI_v3 import SceneManipulator

    manipulator = SceneManipulator()

    result = manipulator.manipulate(
        video_id="123456",
        scene_index=6,
        action="increase",
    )

    print(result.video_path)  # Full manipulated video
    print(result.frame_path)  # Sample frame
"""

from .scene_manipulator import SceneManipulator, ManipulationResult

try:
    from .product_detector import MainProductDetector
except ImportError:
    MainProductDetector = None

__version__ = "3.1.0"
__all__ = ["SceneManipulator", "ManipulationResult", "MainProductDetector"]

"""
GenAI v3: Consistent Video Background Manipulation

No training required! Uses video-native APIs for CONSISTENT manipulation.

Key Features:
- CONSISTENT background across ALL frames (no flicker!)
- Video-native APIs: Google Veo (Colab), Runway, or local SDXL
- ONE background generated, applied to all frames
- Product AUTO-DETECTED using SAM + DINO
- PySceneDetect for real scene boundaries
- Outputs both video AND frame

Backends:
- "google": Google Veo/Imagen via Vertex AI (best for Colab)
- "runway": Runway Gen-3 API
- "consistent": Local SDXL with single background generation
- "auto": Try Google first, fall back to consistent

Usage:
    from GenAI_v3 import SceneManipulator

    # On Google Colab
    manipulator = SceneManipulator(backend="google")

    # Or with local SDXL
    manipulator = SceneManipulator(backend="consistent")

    result = manipulator.manipulate(
        video_id="123456",
        scene_index=6,
        action="increase",
    )

    print(result.video_path)  # Consistent manipulated video
    print(result.frame_path)  # Sample frame
"""

from .scene_manipulator import SceneManipulator, ManipulationResult
from .video_backends import (
    VideoBackend,
    GoogleVeoBackend,
    RunwayBackend,
    ConsistentBackgroundBackend,
    get_backend,
)

try:
    from .product_detector import MainProductDetector
except ImportError:
    MainProductDetector = None

__version__ = "3.2.0"
__all__ = [
    "SceneManipulator",
    "ManipulationResult",
    "MainProductDetector",
    "VideoBackend",
    "GoogleVeoBackend",
    "RunwayBackend",
    "ConsistentBackgroundBackend",
    "get_backend",
]

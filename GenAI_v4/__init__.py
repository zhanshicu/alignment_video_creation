"""
GenAI v4: Manual Veo Workflow for Scene Manipulation

This version uses a manual workflow with Google Veo website:
1. Generate START and END frames locally (SDXL)
2. Upload to Veo website (Frame-to-Video mode)
3. Download Veo output
4. Replace scene in original video

Why this approach?
- Veo API is difficult to use directly
- Veo website is user-friendly
- Frame-to-Video mode gives precise control
- Results are more consistent than API

Workflow:
    from GenAI_v4 import FrameGenerator, SceneReplacer

    # Step 1: Generate frames for Veo
    generator = FrameGenerator()
    inputs = generator.generate_veo_inputs(
        video_id="123456",
        scene_index=6,
        action="increase",
    )

    # Step 2: Use Veo website manually
    # - Upload start_frame.png and end_frame.png
    # - Enter the provided prompt
    # - Download result as veo_output.mp4

    # Step 3: Replace scene in original video
    replacer = SceneReplacer()
    result = replacer.replace_scene(
        video_id="123456",
        scene_index=6,
        action="increase",
    )
"""

from .frame_generator import FrameGenerator, VeoInputs
from .scene_replacer import SceneReplacer, ReplacementResult

__version__ = "4.0.0"
__all__ = [
    "FrameGenerator",
    "VeoInputs",
    "SceneReplacer",
    "ReplacementResult",
]

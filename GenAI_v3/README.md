# GenAI v3: Consistent Video Background Manipulation

**Zero training required! No frame-by-frame inconsistency!**

## Key Innovation

**The Problem**: Frame-by-frame SDXL manipulation creates inconsistent, flickering backgrounds.

**The Solution**: Generate ONE background, apply to ALL frames → perfectly consistent video.

## Features

- **CONSISTENT background** across all frames (no flicker!)
- **Video-native APIs**: Google Veo (Colab), Runway, or local SDXL
- **Product AUTO-DETECTED** using SAM + DINO episodic memory
- **PySceneDetect** for real scene boundaries
- **Outputs both** video AND frame

## Backends

| Backend | Best For | Description |
|---------|----------|-------------|
| `"google"` | Google Colab | Uses Vertex AI (Imagen/Veo) |
| `"runway"` | Runway users | Uses Runway Gen-3 API |
| `"consistent"` | Local GPU | Single SDXL generation |
| `"auto"` | Default | Tries Google, falls back to consistent |

## Quick Start

### On Google Colab (Recommended)

```python
# Authenticate
from google.colab import auth
auth.authenticate_user()

from GenAI_v3 import SceneManipulator

# Initialize with Google backend
manipulator = SceneManipulator(
    video_dir="data/data_tiktok",
    backend="google",
)

# Manipulate scene
result = manipulator.manipulate(
    video_id="123456",
    scene_index=6,
    action="increase",
)

print(result.video_path)  # Consistent video
print(result.frame_path)  # Sample frame
```

### Local GPU

```python
from GenAI_v3 import SceneManipulator

manipulator = SceneManipulator(
    video_dir="data/data_tiktok",
    backend="consistent",  # Single SDXL generation
    device="cuda",
)

result = manipulator.manipulate(
    video_id="123456",
    scene_index=6,
    action="increase",
)
```

## How It Works

1. **Load video** and detect scenes (PySceneDetect)
2. **Auto-detect product** using SAM + DINO episodic memory
3. **Generate ONE background** from reference frame
4. **Apply SAME background** to ALL frames in scene
5. **Composite**: Product from original + Generated background
6. **Export** video and sample frame

## Actions

- **`"increase"`**: Background → **SOLID GRAY** → all attention on product
- **`"decrease"`**: Background → **VIBRANT/COLORFUL** → attention diverts

## Why This Approach?

### Consistency
- Old: Frame-by-frame generation = flickering, inconsistent
- New: Single generation = perfectly consistent background

### Speed
- Old: 30 frames = 30 SDXL runs (~5+ minutes)
- New: 1 SDXL run (~30 seconds)

### Product Preservation
- Product pixels come from ORIGINAL frames
- Only background is changed
- 100% authentic product appearance

## Dependencies

```bash
# Core
pip install torch diffusers transformers opencv-python pandas pillow

# Scene detection
pip install scenedetect[opencv]

# Product detection (optional)
pip install segment-anything

# Google Veo 3.1 backend (recommended)
pip install -U google-genai

# Runway backend (optional)
pip install runwayml
```

## Output Structure

```
outputs/genai_v3/
├── videos/
│   └── {video_id}_scene{N}_{action}.mp4
└── frames/
    └── {video_id}_scene{N}_{action}.png
```

## Performance

| Backend | Time/Scene | GPU Memory |
|---------|------------|------------|
| Google | ~10-30s | N/A (cloud) |
| Consistent | ~30s | ~10GB |
| Runway | ~20-40s | N/A (cloud) |

## Comparison

| Feature | Old (Frame-by-frame) | New (Consistent) |
|---------|---------------------|------------------|
| Background | Inconsistent | Perfectly consistent |
| Speed | Slow (N×SDXL) | Fast (1×SDXL) |
| Product | May vary | 100% original |
| Flicker | Yes | No |

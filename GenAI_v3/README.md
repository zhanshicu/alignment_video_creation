# GenAI v3: Automatic Background Manipulation

**Zero training required!** Uses SOTA models for automatic product detection and background manipulation.

## Key Features

- **Auto-detects main product** using SAM + DINO (no keyword mask needed!)
- **Episodic memory** - Tracks consistent elements across video frames
- **Background only modified** - Product stays authentic
- Uses **SDXL Inpainting** (state-of-the-art)

## How It Works

1. **Sample frames** across the video
2. **SAM** segments all objects in each frame
3. **DINO** tracks which segments appear consistently (episodic memory)
4. **Main product** = most frequent, prominent, consistent element
5. **Invert mask** → select background only
6. **SDXL** inpaints background to control attention

## Quick Start

```python
from GenAI_v3 import SceneManipulator

# Initialize (loads SDXL + SAM + DINO)
manipulator = SceneManipulator()

# Increase attention on auto-detected product
output = manipulator.manipulate(
    video_id="123456",
    scene_index=6,
    action="increase",
)
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_id` | Video ID | Required |
| `scene_index` | Scene number (1-indexed) | Required |
| `action` | `"increase"` or `"decrease"` | Required |
| `strength` | 0.0-1.0 (higher = stronger) | 0.8 |
| `use_keyword_mask` | Force use of keyword mask | False |

## Actions

- **`"increase"`**: Muted background → attention on product
- **`"decrease"`**: Vibrant background → attention diverts

## Data Structure

```
data/
├── data_tiktok/           # Original videos
│   └── {video_id}.mp4     # ← Required
└── valid_scenes.csv       # Optional (for keyword masks)
```

**Note**: Keyword masks are OPTIONAL. Product is auto-detected!

## Models

| Model | Purpose | Size |
|-------|---------|------|
| **SDXL Inpainting** | Background editing | ~7GB |
| **SAM ViT-H** | Object segmentation | ~2.5GB |
| **DINOv2** | Feature tracking | ~350MB |

## Performance

- ~60s per scene (including auto-detection)
- ~12GB GPU memory peak
- No training!

## Why Auto-Detection?

- **Keywords can be unreliable** - CLIPSeg might miss the actual product
- **Episodic memory** - Tracks what appears consistently
- **No manual labeling** - Works out of the box
- **More robust** - Finds the actual main content

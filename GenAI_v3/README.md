# GenAI v3: Background Manipulation for Attention Control

**Zero training required!** Uses SDXL (SOTA) to manipulate backgrounds and control attention.

## Key Concept

- **Product stays UNCHANGED** (authentic)
- **Background is modified** to control viewer attention
- `increase`: Muted background → more attention on product
- `decrease`: Vibrant background → less attention on product

## Quick Start

```python
from GenAI_v3 import SceneManipulator

# Initialize (loads SDXL, ~10GB)
manipulator = SceneManipulator()

# Increase attention on product in scene 6
output = manipulator.manipulate(
    video_id="123456",
    scene_index=6,
    action="increase",
)

# Output: outputs/genai_v3/123456_scene6_increase.mp4
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_id` | Video ID | Required |
| `scene_index` | Scene number (1-indexed) | Required |
| `action` | `"increase"` or `"decrease"` | Required |
| `strength` | 0.0-1.0 (higher = stronger) | 0.8 |
| `num_inference_steps` | Quality (20-50) | 30 |

## Data Structure

```
data/
├── data_tiktok/           # Original videos
│   └── {video_id}.mp4
├── video_scene_cuts/      # Scene images
│   └── {video_id}/
│       └── {video_id}-Scene-0xx-01.jpg
├── keyword_masks/         # Product masks
│   └── {video_id}/
│       └── scene_{x}.png
└── valid_scenes.csv       # Scene metadata
```

## How It Works

1. Load scene image and keyword mask
2. **Invert mask** → select background only
3. SDXL inpainting on background:
   - `increase`: "simple, muted, out of focus background"
   - `decrease`: "vibrant, colorful, detailed background"
4. Replace scene in video with smooth blending
5. Export edited video

## Performance

- ~30-40s per scene
- ~10GB GPU memory
- No training!

## Why Background Manipulation?

- **Product authenticity**: No fake edits to the product
- **Attention control**: Background affects where viewers look
- **Clean A/B testing**: Same product, different context

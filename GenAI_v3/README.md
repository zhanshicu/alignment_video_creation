# GenAI v3: Full Scene Background Manipulation

**Zero training required!** Manipulates ENTIRE SCENES with DRAMATIC background changes.

## Key Features

- **FULL SCENE manipulation** - Not just single frames! Uses PySceneDetect
- **DRAMATIC changes** - Solid gray or psychedelic backgrounds
- **Auto-detects main product** using SAM + DINO episodic memory
- **Smooth video output** - Keyframe interpolation, no static images
- **Dual output** - Returns both video path AND frame path
- Uses **SDXL Inpainting** (state-of-the-art)

## How It Works

1. **Load video** with all frames
2. **PySceneDetect** finds real scene boundaries
3. **SAM + DINO** auto-detect main product (episodic memory)
4. **Manipulate keyframes** across entire scene with SDXL
5. **Interpolate** between keyframes for smooth video
6. **Composite** - Keep product from original, use manipulated background
7. **Export** both video and sample frame

## Quick Start

```python
from GenAI_v3 import SceneManipulator

# Initialize (loads SDXL + SAM + DINO + PySceneDetect)
manipulator = SceneManipulator()

# Increase attention on auto-detected product
result = manipulator.manipulate(
    video_id="123456",
    scene_index=6,
    action="increase",
)

# Both video and frame are output!
print(result.video_path)        # Full manipulated video
print(result.frame_path)        # Sample frame
print(result.frames_manipulated) # How many frames changed
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_id` | Video ID | Required |
| `scene_index` | Scene number (1-indexed) | Required |
| `action` | `"increase"` or `"decrease"` | Required |
| `strength` | 0.0-1.0 (higher = more dramatic) | 0.95 |
| `num_inference_steps` | Quality (30-50) | 40 |
| `keyframe_interval` | Process every Nth frame | 10 |
| `use_keyword_mask` | Force use of keyword mask | False |

## Actions

- **`"increase"`**: Background → **SOLID GRAY/PLAIN** → all attention on product
- **`"decrease"`**: Background → **VIBRANT/PSYCHEDELIC** → attention diverts

## Output Structure

```
outputs/genai_v3/
├── videos/                           # Full manipulated videos
│   └── {video_id}_scene{N}_{action}.mp4
└── frames/                           # Sample frames
    └── {video_id}_scene{N}_{action}.png
```

## Data Structure

```
data/
├── data_tiktok/           # Original videos
│   └── {video_id}.mp4     # ← Required
└── valid_scenes.csv       # Optional (for keyword masks)
```

**Note**: Keyword masks are OPTIONAL. Product is auto-detected!

## Dependencies

```bash
pip install torch diffusers transformers
pip install segment-anything  # SAM
pip install scenedetect[opencv]  # PySceneDetect
pip install opencv-python pandas pillow
```

## Models

| Model | Purpose | Size |
|-------|---------|------|
| **SDXL Inpainting** | Background editing | ~7GB |
| **SAM ViT-H** | Object segmentation | ~2.5GB |
| **DINOv2** | Feature tracking | ~350MB |

## Performance

- **~2-5 min per scene** (depends on scene length)
- **~12-15GB GPU memory** peak
- **No training!**

## Why This Approach?

### Full Scene Manipulation
- Single frame insertion looks fake (static image in video)
- PySceneDetect finds real scene cuts
- Keyframe interpolation creates smooth transitions

### Dramatic Changes
- Subtle changes don't affect attention
- Solid gray vs psychedelic is clearly different
- A/B testing needs visible contrast

### Auto-Detection
- Keywords can be unreliable
- Episodic memory tracks consistent elements
- No manual labeling required

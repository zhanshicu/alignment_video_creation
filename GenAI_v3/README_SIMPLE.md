# Simple Scene Manipulation (GenAI v3)

**The simplest way to manipulate attention-keyword alignment - NO TRAINING!**

---

## 🎯 What This Does

Manipulate **one specific scene** in a video to increase or decrease attention-keyword alignment:

1. Input: `video_id="123456"`, `scene_index=6`, `action="increase"`
2. Output: Edited video with scene 6 modified to draw more attention to the keyword

---

## ⚡ Quick Start (3 Lines of Code!)

```python
from GenAI_v3 import SceneManipulator

# Initialize (downloads model once, ~5GB)
manipulator = SceneManipulator()

# Increase alignment for scene 6
output = manipulator.manipulate_and_replace(
    video_id="123456",
    scene_index=6,
    action="increase",  # or "decrease"
)

print(f"Done! {output}")
```

**That's it!** No training, no complex configuration.

---

## 📁 Data Structure

Same as before - uses existing data:

```
data/
├── data_tiktok/               # Original videos
│   └── {video_id}.mp4         # ← Your videos
│
├── video_scene_cuts/          # Scene images
│   └── {video_id}/
│       └── {video_id}-Scene-0xx-01.jpg
│
├── keyword_masks/             # Keyword masks
│   └── {video_id}/
│       └── scene_{x}.png
│
└── valid_scenes.csv           # Scene metadata
```

---

## 🔄 How It Works

### Stage 1: Load Video
```python
# Load original video from data/data_tiktok/{video_id}.mp4
video_frames, fps = load_video(video_id)
```

### Stage 2: Identify Scene Location
```python
# Find where scene 6 appears in the video (frame range)
start_frame, end_frame = find_scene_location(video_id, scene_index=6)
```

### Stage 3: Manipulate Scene
```python
# Load scene image and manipulate it
scene_image = load_scene_image(video_id, scene_index=6)
manipulated = manipulate_frame(scene_image, action="increase")
```

### Stage 4: Replace in Video
```python
# Replace frames in video with smooth blending
video_frames[start_frame:end_frame] = manipulated
```

### Stage 5: Export
```python
# Save edited video
export_video(video_frames, output_path)
```

---

## 🎨 Examples

### Example 1: Increase Alignment (Basic)

```python
manipulator = SceneManipulator()

output = manipulator.manipulate_and_replace(
    video_id="123456",
    scene_index=6,
    action="increase",
)
```

**Effect**: Makes the keyword more visually prominent in scene 6

---

### Example 2: Decrease Alignment

```python
output = manipulator.manipulate_and_replace(
    video_id="123456",
    scene_index=3,
    action="decrease",
)
```

**Effect**: Makes the keyword blend into the background in scene 3

---

### Example 3: Custom Strength

```python
output = manipulator.manipulate_and_replace(
    video_id="123456",
    scene_index=5,
    action="increase",
    boost_strength=1.2,  # Subtle (default is 1.5)
    num_inference_steps=30,  # Higher quality (default is 20)
)
```

**Boost Strength Guide**:
- `1.2`: Subtle increase
- `1.5`: Moderate increase (default)
- `2.0`: Strong increase
- `0.5`: Reduction (for "decrease")

---

### Example 4: Multiple Scenes

```python
output = manipulator.manipulate_multiple_scenes(
    video_id="123456",
    scene_actions={
        3: "increase",   # Boost scene 3
        6: "increase",   # Boost scene 6
        8: "decrease",   # Reduce scene 8
    },
)
```

---

## 🔧 Configuration Options

### Scene Detection Method

```python
# Fast (default): Estimate scene location from scene count
manipulator = SceneManipulator(use_scene_detection=False)  # ~instant

# Accurate: Use automatic scene detection
manipulator = SceneManipulator(use_scene_detection=True)   # ~2-3s
```

### Quality vs Speed

```python
# Fast (lower quality)
num_inference_steps=10      # ~15s per scene

# Balanced (recommended)
num_inference_steps=20      # ~25s per scene

# High quality
num_inference_steps=50      # ~60s per scene
```

### Blending Smoothness

```python
# Abrupt transition
blend_frames=3

# Smooth (default)
blend_frames=5

# Very smooth
blend_frames=10
```

### Method Selection

```python
# InstructPix2Pix (recommended - simpler, no masks needed)
manipulator = SceneManipulator(method="instruct_pix2pix")

# Inpainting (more precise, requires keyword masks)
manipulator = SceneManipulator(method="inpainting")
```

---

## 🐛 Debugging: Preview Scene Location

Not sure which frames will be edited? Preview it:

```python
manipulator.preview_scene_location(
    video_id="123456",
    scene_index=6,
)
```

This shows:
- Reference scene image
- Video frames before/during/after the scene
- Frame numbers and timing

Use this to verify scene detection is working correctly!

---

## 📊 Performance

| Task | Time | GPU Memory |
|------|------|------------|
| Model initialization | ~30s (first run) | 5-6 GB |
| Scene manipulation | ~25s | 2-3 GB |
| Video loading | ~5s | ~1 GB |
| Video export | ~10s | ~1 GB |
| **Total per scene** | **~40s** | **~6-8 GB** |

**Much faster than training** (which takes hours + 20GB GPU)!

---

## 🎯 Use Cases

### Use Case 1: A/B Testing

Create two versions of the same video:

```python
# Version A: Original
# (no changes)

# Version B: Boost scene 6
manipulator.manipulate_and_replace(
    video_id="video1",
    scene_index=6,
    action="increase",
)

# Deploy both, measure CTR/CVR
```

### Use Case 2: Temporal Patterns

Test if timing matters:

```python
# Early boost
manipulator.manipulate_and_replace(
    video_id="video1",
    scene_index=2,  # Early scene
    action="increase",
)

# Late boost
manipulator.manipulate_and_replace(
    video_id="video1",
    scene_index=8,  # Late scene
    action="increase",
)

# Compare performance
```

### Use Case 3: Strength Comparison

Test different boost levels:

```python
for strength in [1.2, 1.5, 2.0]:
    manipulator.manipulate_and_replace(
        video_id="video1",
        scene_index=6,
        action="increase",
        boost_strength=strength,
        output_path=f"outputs/video1_scene6_boost{strength}.mp4",
    )
```

---

## 🛠️ Troubleshooting

### Issue 1: Video Not Found

**Error**: `FileNotFoundError: Video not found: data/data_tiktok/123456.mp4`

**Solution**: Check that your videos are in the correct directory
```python
# List available videos
import os
videos = os.listdir("../data/data_tiktok")
print(videos)
```

---

### Issue 2: Scene Not Found

**Error**: `Scene 6 not found for video 123456. Available scenes: [1, 2, 3, 4, 5]`

**Solution**: This video only has 5 scenes. Use a different scene index.

---

### Issue 3: Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
```python
# Use CPU (slower but works)
manipulator = SceneManipulator(device="cpu")

# Or reduce inference steps
num_inference_steps=10  # Instead of 20
```

---

### Issue 4: Scene Detection Inaccurate

**Problem**: Wrong frames are being edited

**Solution**: Enable automatic scene detection
```python
manipulator = SceneManipulator(use_scene_detection=True)
```

Or preview first:
```python
manipulator.preview_scene_location(video_id, scene_index)
```

---

## 📚 API Reference

### SceneManipulator

```python
class SceneManipulator:
    def __init__(
        self,
        valid_scenes_file: str = "../data/valid_scenes.csv",
        video_dir: str = "../data/data_tiktok",
        output_dir: str = "../outputs/genai_v3/manipulated_videos",
        method: Literal["instruct_pix2pix", "inpainting"] = "instruct_pix2pix",
        device: str = "cuda",
        use_scene_detection: bool = False,
    )

    def manipulate_and_replace(
        self,
        video_id: str,
        scene_index: int,
        action: Literal["increase", "decrease"],
        keyword: Optional[str] = None,
        boost_strength: Optional[float] = None,
        num_inference_steps: int = 20,
        blend_frames: int = 5,
        output_path: Optional[str] = None,
    ) -> str

    def manipulate_multiple_scenes(
        self,
        video_id: str,
        scene_actions: dict,  # {scene_index: action}
        keyword: Optional[str] = None,
        num_inference_steps: int = 20,
        blend_frames: int = 5,
        output_path: Optional[str] = None,
    ) -> str

    def preview_scene_location(
        self,
        video_id: str,
        scene_index: int,
        output_dir: Optional[str] = None,
    )
```

---

## 🎓 Tutorial: Step-by-Step

### Step 1: Find Available Videos

```python
import pandas as pd

scenes_df = pd.read_csv("../data/valid_scenes.csv")
print(scenes_df[['video_id', 'scene_number', 'keyword']].head(20))
```

### Step 2: Pick a Video and Scene

```python
video_id = "123456"  # From Step 1
scene_index = 6       # Choose a scene number
```

### Step 3: Preview (Optional)

```python
manipulator.preview_scene_location(video_id, scene_index)
```

### Step 4: Manipulate

```python
output = manipulator.manipulate_and_replace(
    video_id=video_id,
    scene_index=scene_index,
    action="increase",
)
```

### Step 5: Check Output

```python
print(f"Original: data/data_tiktok/{video_id}.mp4")
print(f"Edited:   {output}")

# Play videos side-by-side to compare
```

---

## 🚀 Next Steps

1. **Run the workflow**: Open `simple_workflow.ipynb`
2. **Manipulate a scene**: Try the examples above
3. **Deploy for testing**: Upload edited videos
4. **Collect metrics**: Track CTR, CVR, engagement
5. **Analyze results**: Which scenes/strengths work best?

---

**Version**: 1.0
**Status**: Ready to use! 🎉
**Training Required**: ❌ None!

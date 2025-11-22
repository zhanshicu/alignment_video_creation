# GenAI v4: Manual Veo Workflow

**3-Step Process** for scene manipulation using Google Veo website.

## Why This Approach?

- Veo API is complex and has limitations
- Veo website is user-friendly with "Frame to Video" mode
- Gives precise control over background transitions
- More consistent results

## Quick Start

```python
from GenAI_v4 import FrameGenerator, SceneReplacer

# STEP 1: Generate frames for Veo
generator = FrameGenerator()
inputs = generator.generate_veo_inputs(
    video_id="123456",
    scene_index=6,
    action="increase",
)
# Creates: start_frame.png, end_frame.png, veo_instructions.txt

# STEP 2: Use Veo website (manual)
# - Upload start_frame.png and end_frame.png
# - Enter the provided prompt
# - Download result as veo_output.mp4

# STEP 3: Replace scene
replacer = SceneReplacer()
result = replacer.replace_scene(
    video_id="123456",
    scene_index=6,
    action="increase",
)
print(result.output_video_path)
```

## Workflow Details

### Step 1: Generate Veo Inputs

The `FrameGenerator` creates:

| File | Description |
|------|-------------|
| `start_frame.png` | Beginning of transition (original or slight mod) |
| `end_frame.png` | End of transition (target background) |
| `product_mask.png` | Auto-detected product region |
| `veo_instructions.txt` | Full instructions for Veo website |

**Key Parameters:**
```python
inputs = generator.generate_veo_inputs(
    video_id="123456",
    scene_index=6,              # 1-indexed scene number
    action="increase",          # "increase" (plain bg) or "decrease" (vibrant bg)
    transition_style="dramatic" # "smooth" or "dramatic"
)
```

### Step 2: Veo Website (Manual)

1. **Open**: https://labs.google/fx/tools/video-fx

2. **Select**: "Frame to Video" mode

3. **Upload Frames**:
   - START frame → First frame
   - END frame → Last frame

4. **Enter Prompt** (from veo_instructions.txt):

   For `action="increase"`:
   ```
   Smooth transition of background becoming simpler and more muted.
   The central subject remains perfectly still and unchanged.
   Background gradually transforms to plain neutral gray.
   Soft, professional product photography lighting.
   Camera is completely static. No movement of the main subject.
   ```

   For `action="decrease"`:
   ```
   Smooth transition of background becoming more vibrant and colorful.
   The central subject remains perfectly still and unchanged.
   Background gradually transforms with rich colors and subtle patterns.
   Dynamic lighting effects in background only.
   Camera is completely static. No movement of the main subject.
   ```

5. **Configure Settings**:
   | Setting | Value |
   |---------|-------|
   | Duration | Match scene duration |
   | Aspect Ratio | Match video (16:9 or 9:16) |
   | Motion | Minimal / Static camera |
   | Style | Photorealistic |

6. **Generate** and **Download** as `veo_output.mp4`

7. **Save** to: `outputs/genai_v4/veo_inputs/{video}_scene{N}_{action}/veo_output.mp4`

### Step 3: Replace Scene

The `SceneReplacer` handles:

- **PySceneDetect** - Finds exact scene boundaries
- **Frame resampling** - Matches Veo output to scene length
- **Color matching** - Histogram-based color alignment
- **Temporal blending** - Smooth transitions at boundaries

```python
result = replacer.replace_scene(
    video_id="123456",
    scene_index=6,
    action="increase",
    blend_frames=5,      # Frames for boundary blending
    match_colors=True,   # Apply color matching
)
```

## Output Structure

```
outputs/genai_v4/
├── veo_inputs/
│   └── {video}_scene{N}_{action}/
│       ├── start_frame.png      # Upload to Veo
│       ├── end_frame.png        # Upload to Veo
│       ├── product_mask.png     # Reference
│       ├── veo_instructions.txt # Full instructions
│       └── veo_output.mp4       # YOU download this
└── final_videos/
    ├── {video}_scene{N}_{action}_final.mp4
    └── {video}_scene{N}_{action}_sample.png
```

## Dependencies

```bash
pip install torch diffusers transformers opencv-python pandas pillow
pip install scenedetect[opencv]
pip install segment-anything  # Optional for product detection
```

## Tips for Best Results

### Frame Generation
- Use `transition_style="dramatic"` for visible changes
- Product is auto-detected (SAM + DINO)
- END frame shows the target background state

### Veo Website
- **Keep motion minimal** - product should stay still
- **Match duration** - check scene info in instructions
- **Note the seed** - for regeneration if needed
- **Try multiple generations** - Veo can be inconsistent

### Scene Replacement
- `blend_frames=5` gives smooth transitions
- `match_colors=True` helps with color consistency
- Check sample frame before full export

## Comparison with Previous Versions

| Version | Approach | Pros | Cons |
|---------|----------|------|------|
| v3 | API-based | Automated | API complexity, inconsistent |
| **v4** | **Manual Veo** | **User control, consistent** | **Manual step required** |

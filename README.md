# Attention-Keyword-Aligned Video Ad Manipulation Framework

A framework for generating controlled variants of video advertisements by manipulating **frame-level, region-local alignment** between **attention heatmaps** and **keyword heatmaps**, without requiring training from scratch.

## Overview

This framework implements the design documented in `Framework Document.md`. It enables researchers to:

1. **Generate attention and keyword heatmaps** for video frames
2. **Train a ControlNet adapter** to understand and manipulate these heatmaps
3. **Create experimental variants** with different alignment profiles
4. **Edit videos** with temporal consistency for experimental studies

### Key Features

- ✅ No training from scratch - uses pretrained Stable Diffusion
- ✅ Local, frame-specific manipulations only
- ✅ Minimal disruption to non-keyword regions
- ✅ Temporal consistency through optical flow or Rerender A Video integration
- ✅ Multiple experimental variants (early/middle/late boost, reduction, placebo)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preparation                          │
│  Video → Patches → LLaVA → Attention Heatmap (A_t)         │
│  Video → CLIPSeg → Keyword Heatmap (K_t)                    │
│  Control Tensor: C_t = [M_t, S_t, (A_t, K_t)]              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   ControlNet Training                        │
│  Frozen SD + Trainable ControlNet Adapter                   │
│  Losses: L_diff + L_recon + L_bg                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Experimental Variants                        │
│  Baseline | Early Boost | Middle Boost | Late Boost         │
│  Full Boost | Reduction | Placebo                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Video Editing                              │
│  Keyframe Editing + Temporal Consistency                    │
│  Output: Multiple video variants for experiments            │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended: >= 16GB VRAM)
- PyTorch >= 2.0

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd meaning_alignment_tiktok

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced features
pip install peft  # For LoRA fine-tuning
pip install xformers  # For memory-efficient attention
```

## Quick Start

### 1. Data Preparation

Generate attention and keyword heatmaps from your video:

```bash
python examples/01_prepare_data.py
```

This script:
- Extracts circular patches from video frames
- Saves patches for LLaVA semantic scoring
- Generates keyword heatmaps using CLIPSeg
- Builds control tensors C_t = [M_t, S_t]

**Note:** You need to run your fine-tuned LLaVA model separately to score the patches. See `workflow.ipynb` for reference.

### 2. Training

Train the ControlNet adapter:

```bash
python examples/02_train_model.py
```

Configuration options in `configs/default_config.yaml`:
- Model architecture
- Training hyperparameters
- Loss weights (λ_recon, λ_lpips, λ_bg)
- Data paths

### 3. Create Experimental Variants

Generate different alignment manipulation conditions:

```bash
python examples/03_create_variants.py
```

Creates:
- **Baseline**: No manipulation
- **Early/Middle/Late Boost**: Increased alignment in specific temporal windows
- **Full Boost**: Increased alignment throughout
- **Reduction**: Decreased alignment
- **Placebo**: Manipulation outside keyword region

### 4. Edit Videos

Apply edits with temporal consistency:

```bash
python examples/04_edit_video.py
```

Options:
- Keyframe-based editing with optical flow propagation
- Integration with Rerender A Video for advanced consistency
- Side-by-side comparison generation

## Project Structure

```
meaning_alignment_tiktok/
├── src/
│   ├── data_preparation/
│   │   ├── attention_heatmap.py      # Attention heatmap generation
│   │   ├── keyword_heatmap.py        # CLIPSeg keyword localization
│   │   └── control_tensor.py         # Control tensor construction
│   ├── models/
│   │   ├── controlnet_adapter.py     # ControlNet architecture
│   │   └── stable_diffusion_wrapper.py  # SD + ControlNet integration
│   ├── training/
│   │   ├── trainer.py                # Training loop
│   │   ├── losses.py                 # Loss functions
│   │   └── dataset.py                # Data loading
│   ├── video_editing/
│   │   ├── video_editor.py           # Frame-level editing
│   │   ├── temporal_consistency.py   # Temporal propagation
│   │   └── experimental_variants.py  # Variant generation
│   └── utils/
│       ├── video_utils.py            # Video I/O
│       ├── visualization.py          # Visualization tools
│       └── metrics.py                # Evaluation metrics
├── configs/
│   └── default_config.yaml           # Configuration file
├── examples/
│   ├── 01_prepare_data.py
│   ├── 02_train_model.py
│   ├── 03_create_variants.py
│   └── 04_edit_video.py
├── workflow.ipynb                     # Original workflow reference
├── Framework Document.md              # Detailed framework design
└── requirements.txt
```

## Configuration

Edit `configs/default_config.yaml` to customize:

### Model Settings
```yaml
model:
  sd_model_name: "runwayml/stable-diffusion-v1-5"
  controlnet:
    control_channels: 2  # 2 for [M_t, S_t], 4 for full
    base_channels: 64
```

### Training Settings
```yaml
training:
  batch_size: 4
  learning_rate: 1.0e-4
  lambda_recon: 1.0
  lambda_lpips: 1.0
  lambda_bg: 0.5
```

### Experimental Settings
```yaml
experiments:
  boost_alpha: 1.5      # Alignment boost factor
  reduction_alpha: 0.5  # Alignment reduction factor
  window_type: "thirds" # Temporal window division
```

## Data Format

### Input Structure
```
data/
├── frames/
│   └── video_id/
│       ├── frame_00000.png
│       ├── frame_00001.png
│       └── ...
├── attention_heatmaps/
│   └── video_id/
│       ├── frame_00000.png
│       └── ...
├── keyword_heatmaps/
│   └── video_id/
│       ├── frame_00000.png
│       └── ...
└── keywords.json
```

### keywords.json Format
```json
{
  "video_id_1": "jewelry",
  "video_id_2": "running shoes",
  "video_id_3": "lipstick"
}
```

## Loss Functions

### 1. Diffusion Loss
```
L_diff = ||ε̂ - ε||²
```
Standard denoising objective for diffusion models.

### 2. Reconstruction Loss
```
L_recon = ||Î_t - I_t||₁ + λ_LPIPS · LPIPS(Î_t, I_t)
```
Combines L1 loss with perceptual LPIPS loss.

### 3. Background Preservation Loss
```
L_bg = λ_bg · ||(Î_t - I_t) ⊙ B_t||₁
```
Ensures minimal changes outside keyword regions.

### Total Loss
```
L = L_diff + λ_recon · L_recon + λ_bg · L_bg
```

## Experimental Variants

The framework generates the following variants for controlled experiments:

| Variant | Description | Use Case |
|---------|-------------|----------|
| **Baseline** | No manipulation | Control condition |
| **Early Boost** | Increased alignment in early frames (0-33%) | Test timing effects |
| **Middle Boost** | Increased alignment in middle frames (33-66%) | Peak attention manipulation |
| **Late Boost** | Increased alignment in late frames (66-100%) | Recency effects |
| **Full Boost** | Increased alignment throughout | Maximum effect |
| **Reduction** | Decreased alignment in middle | Negative control |
| **Placebo** | Manipulation outside keyword | Control for editing artifacts |

## Integration with Rerender A Video

For advanced temporal consistency, integrate with [Rerender A Video](https://github.com/williamyang1991/Rerender_A_Video):

```python
from src.video_editing import TemporalConsistencyWrapper

wrapper = TemporalConsistencyWrapper(method="rerender")
wrapper.save_for_rerender(
    original_frames, edited_keyframes,
    output_dir="rerender_input",
    video_name="video_001"
)
```

Then run Rerender A Video on the exported frames.

## Evaluation Metrics

The framework includes several evaluation metrics:

- **Alignment Score**: Proportion of attention on keyword region
- **Temporal Consistency**: Frame-to-frame correlation
- **Spatial Overlap**: IoU and Dice between attention and keyword
- **Variant Effects**: Statistical comparison between variants

```python
from src.utils import calculate_alignment_metrics

metrics = calculate_alignment_metrics(
    attention_maps, keyword_maps, keyword_masks
)
print(f"Mean alignment: {metrics['mean_alignment']:.4f}")
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use `mixed_precision: true`
- Reduce image resolution in data config

### Poor Temporal Consistency
- Increase `keyframe_interval`
- Use `temporal_method: "rerender"` for better results
- Apply temporal smoothing with larger `blend_window`

### Training Instability
- Lower `learning_rate`
- Adjust loss weights (`lambda_recon`, `lambda_bg`)
- Enable gradient clipping

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{attention_keyword_alignment,
  title={Attention-Keyword-Aligned Video Ad Manipulation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/meaning_alignment_tiktok}
}
```

## References

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Rerender A Video](https://github.com/williamyang1991/Rerender_A_Video)
- [CLIPSeg](https://github.com/timojl/clipseg)
- Henderson, T. R. (2017). Semantic richness and attention in visual cognition.

## License

[Add your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

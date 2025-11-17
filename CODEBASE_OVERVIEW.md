# Attention-Keyword-Aligned Video Ad Manipulation Framework

## Overview

This is a research platform for creating experimental variants of TikTok-style video advertisements by systematically manipulating the alignment between viewer attention and product placement. The framework enables causal inference about what drives viewer engagement through controlled experiments.

**Key Innovation**: Uses ControlNet + Stable Diffusion to generate counterfactual video versions where attention-product alignment is manipulated while keeping everything else constant.

## What Problem Does This Solve?

In video advertising, understanding the causal relationship between viewer attention and product placement is challenging. This framework allows researchers to:

1. **Create Controlled Experiments**: Generate multiple variants of the same video with different attention-product alignments
2. **Test Causal Hypotheses**: Measure how alignment affects engagement metrics (CTR, CVR, watch time)
3. **Maintain Visual Consistency**: Edit videos while preserving temporal coherence and visual quality
4. **Scale Research**: Process large datasets with pre-computed alignment scores

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Data                                │
│  • Video files or frames                                     │
│  • Alignment scores (CSV)                                    │
│  • Keywords/Product descriptions (CSV)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Data Preparation                             │
│  • Extract patches from frames                               │
│  • Generate attention heatmaps (LLaVA semantic scoring)      │
│  • Generate keyword heatmaps (CLIPSeg segmentation)          │
│  • Build control tensors: C_t = [M_t, S_t]                  │
│    - M_t: Keyword Mask (where product appears)              │
│    - S_t: Alignment Map (attention × keywords)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              ControlNet Training                             │
│  • Frozen Stable Diffusion backbone                          │
│  • Trainable ControlNet adapter (~50-100M params)            │
│  • Multi-component loss:                                     │
│    - L_diff: Diffusion loss                                  │
│    - L_recon: Reconstruction quality                         │
│    - L_bg: Background preservation                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Experimental Variant Generation                   │
│  Creates 7 variants with modified alignment:                 │
│  1. Baseline (no change)                                     │
│  2. Early Boost (1.5x alignment in first 33%)                │
│  3. Middle Boost (1.5x alignment in middle 33%)              │
│  4. Late Boost (1.5x alignment in last 33%)                  │
│  5. Full Boost (1.5x throughout)                             │
│  6. Reduction (0.5x in middle)                               │
│  7. Placebo (modification outside keyword region)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Video Editing                              │
│  • Edit keyframes with trained ControlNet                    │
│  • Propagate edits via optical flow                          │
│  • Maintain temporal consistency                             │
│  • Generate side-by-side comparisons                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Output: 7 Video Variants                    │
│  Ready for A/B testing and causal analysis                   │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

**Stable Diffusion (Frozen)**:
- VAE encoder/decoder for latent representation
- U-Net denoiser with multi-scale features
- Text encoder for keyword conditioning
- No training - only inference

**ControlNet Adapter (Trainable)**:
- Lightweight U-Net processing control tensors
- Produces multi-scale feature injections
- Connected via zero-initialized convolutions
- Learns to manipulate attention-product alignment

**Feature Injection**:
```
H_SD^(l) ← H_SD^(l) + Conv_0(F_control^(l))
```
Where Conv_0 starts at zero, allowing gradual learning

## Directory Structure

```
alignment_video_creation/
├── data/
│   ├── alignment_score.csv      # Pre-computed alignment metrics (175,934 rows)
│   └── keywords.csv              # Video keywords/products (107,206 rows)
│
├── src/                          # Main source code (~4,820 lines)
│   ├── data_preparation/         # Heatmap and tensor generation
│   ├── models/                   # ControlNet + SD integration
│   ├── training/                 # Training pipeline and losses
│   ├── video_editing/            # Video manipulation and consistency
│   └── utils/                    # Utilities and metrics
│
├── src_alignment/clipseg/        # CLIPSeg model implementation
│
├── examples/                     # Runnable example scripts
│   ├── 01_prepare_data.py        # Data preprocessing
│   ├── 02_train_model.py         # Train ControlNet
│   ├── 03_create_variants.py    # Generate 7 experimental variants
│   └── 04_edit_video.py          # Edit videos with consistency
│
├── notebooks/                    # Interactive workflows
│   ├── finetune_and_inference_v2.ipynb  # Updated CSV-based workflow
│   └── finetune_and_inference.ipynb     # Original heatmap workflow
│
├── configs/
│   └── default_config.yaml       # Configuration parameters
│
└── documentation/
    ├── README.md                 # Framework documentation
    └── Framework Document.md     # Detailed technical design
```

## Data Format

### alignment_score.csv
Pre-computed alignment metrics for video scenes:

| Column | Description |
|--------|-------------|
| video id | Unique video identifier |
| Scene Number | Scene index within video |
| attention_proportion | Alignment score [0, 1] |
| start_time | Scene start timestamp |
| end_time | Scene end timestamp |
| CTR_mean | Click-through rate |
| CVR_mean | Conversion rate |
| Clicks_mean | Average clicks |
| Conversion_mean | Average conversions |
| Remain_mean | Average watch time |
| contrast | Visual contrast metric |
| brightness | Visual brightness metric |
| industry | Product category |

**Example**:
```csv
video id,Scene Number,attention_proportion,start_time,end_time,CTR_mean,industry
7123456789,0,0.73,0.0,1.5,0.042,Children's Apparel
7123456789,1,0.85,1.5,3.0,0.042,Children's Apparel
```

### keywords.csv
Video-to-product keyword mapping:

| Column | Description |
|--------|-------------|
| _id | Video ID |
| keyword_list[0] | Primary product keyword |

**Example**:
```csv
_id,keyword_list[0]
7123456789,four-leaf clover necklace
7234567890,diamond rings
```

## Workflow Example

### Step 1: Data Preparation

```python
# examples/01_prepare_data.py
from src.data_preparation import AttentionHeatmapGenerator, KeywordHeatmapGenerator

# Extract patches and generate attention heatmaps
attention_gen = AttentionHeatmapGenerator(
    visual_angles=[3, 7],  # degrees
    smoothing_sigma=5.0,
    gamma=3.0
)
attention_heatmaps = attention_gen.process_video(video_path)

# Generate keyword heatmaps using CLIPSeg
keyword_gen = KeywordHeatmapGenerator(model_name="CIDAS/clipseg-rd64-refined")
keyword_heatmaps = keyword_gen.generate(frames, keyword="necklace")

# Build control tensors
control_tensors = build_control_tensor(attention_heatmaps, keyword_heatmaps)
```

**Output**: Control tensors C_t = [M_t, S_t] for each frame

### Step 2: Train ControlNet

```python
# examples/02_train_model.py
from src.training import ControlNetTrainer
from src.models import ControlNetAdapter, StableDiffusionWrapper

# Initialize model
model = StableDiffusionWrapper(
    sd_model="runwayml/stable-diffusion-v1-5",
    controlnet=ControlNetAdapter(control_channels=2)
)

# Train
trainer = ControlNetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)
trainer.train(epochs=100)
```

**Output**: Trained ControlNet checkpoint

### Step 3: Generate Experimental Variants

```python
# examples/03_create_variants.py
from src.video_editing import ExperimentalVariantsV2

# Load alignment data
alignment_df = pd.read_csv("data/alignment_score.csv")
keywords_df = pd.read_csv("data/keywords.csv")

# Create variant generator
variant_gen = ExperimentalVariantsV2(
    alignment_df=alignment_df,
    keywords_df=keywords_df,
    boost_alpha=1.5,
    reduction_alpha=0.5
)

# Generate all 7 variants
variants = variant_gen.create_all_variants(video_id="7123456789")
```

**Output**: 7 modified control tensors for different experimental conditions

### Step 4: Edit Videos

```python
# examples/04_edit_video.py
from src.video_editing import VideoEditor, TemporalConsistency

# Load trained model
editor = VideoEditor(model_checkpoint="checkpoints/best_model.pt")

# Edit video with each variant
for variant_name, control_tensors in variants.items():
    edited_frames = editor.edit_video(
        original_frames=frames,
        control_tensors=control_tensors,
        keyword="necklace"
    )

    # Apply temporal consistency
    consistent_frames = TemporalConsistency.propagate(edited_frames)

    # Save video
    save_video(consistent_frames, f"output/{variant_name}.mp4")
```

**Output**: 7 edited video files ready for A/B testing

## Experimental Variants Explained

Each variant manipulates the alignment score to test different hypotheses:

| Variant | Manipulation | Research Question |
|---------|--------------|-------------------|
| **Baseline** | No change | Control condition |
| **Early Boost** | 1.5× alignment in first 33% of frames | Does early attention matter more? |
| **Middle Boost** | 1.5× alignment in middle 33% | Is peak attention critical? |
| **Late Boost** | 1.5× alignment in last 33% | Do recency effects dominate? |
| **Full Boost** | 1.5× throughout entire video | Maximum alignment effect |
| **Reduction** | 0.5× in middle frames | Negative control: does less alignment hurt? |
| **Placebo** | Modification outside keyword region | Controls for editing artifacts |

### Visual Example

```
Original Alignment:  [0.3, 0.4, 0.5, 0.6, 0.7, 0.6]
Early Boost:         [0.45, 0.6, 0.5, 0.6, 0.7, 0.6]  ← First 33% boosted
Middle Boost:        [0.3, 0.4, 0.75, 0.9, 0.7, 0.6]  ← Middle 33% boosted
Late Boost:          [0.3, 0.4, 0.5, 0.6, 1.0, 0.9]   ← Last 33% boosted
Full Boost:          [0.45, 0.6, 0.75, 0.9, 1.0, 0.9] ← All frames boosted
Reduction:           [0.3, 0.4, 0.25, 0.3, 0.7, 0.6]  ← Middle reduced
```

## Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
model:
  sd_model_name: "runwayml/stable-diffusion-v1-5"
  controlnet:
    control_channels: 2      # [M_t, S_t]
    base_channels: 64
    num_res_blocks: 2

training:
  batch_size: 4
  learning_rate: 1.0e-4
  num_epochs: 100
  gradient_clip: 1.0

  # Loss weights
  lambda_recon: 1.0          # Reconstruction quality
  lambda_lpips: 1.0          # Perceptual similarity
  lambda_bg: 0.5             # Background preservation

data:
  video_size: [512, 512]
  num_frames: 16
  scene_based: true          # Use scene-level alignment scores

experiments:
  boost_alpha: 1.5           # Boost multiplier
  reduction_alpha: 0.5       # Reduction multiplier
  variants:
    - baseline
    - early_boost
    - middle_boost
    - late_boost
    - full_boost
    - reduction
    - placebo
```

## Loss Functions

The model is trained with three complementary losses:

### 1. Diffusion Loss (L_diff)
```python
L_diff = MSE(predicted_noise, actual_noise)
```
Ensures the model learns proper diffusion denoising.

### 2. Reconstruction Loss (L_recon)
```python
L_recon = L1(edited_frame, target_frame) + LPIPS(edited_frame, target_frame)
```
Combines pixel-level accuracy (L1) with perceptual quality (LPIPS).

### 3. Background Preservation Loss (L_bg)
```python
L_bg = MSE((1 - M_t) ⊙ edited_frame, (1 - M_t) ⊙ original_frame)
```
Penalizes changes outside the keyword region, maintaining visual consistency.

### Total Loss
```python
L_total = L_diff + λ_recon × L_recon + λ_bg × L_bg
```

## Research Applications

### Experimental Design Process

1. **Generate Variants**: Create 7 versions of each test video
2. **Deploy**: Upload to A/B testing platform (e.g., TikTok Ads Manager)
3. **Measure**: Track engagement metrics
   - Click-Through Rate (CTR)
   - Conversion Rate (CVR)
   - Watch Time
   - Engagement Rate
4. **Analyze**: Compare variants to establish causal effects

### Key Research Questions

- **Causal Effect**: Does attention-product alignment causally affect engagement?
- **Dose-Response**: Is there a linear relationship between alignment and clicks?
- **Timing Effects**: When is alignment most critical (early/middle/late)?
- **Heterogeneity**: Do effects vary by product category or industry?
- **Interaction Effects**: How does alignment interact with other visual features (contrast, brightness)?

### Example Analysis

```python
import pandas as pd
from scipy import stats

# Load experimental results
results = pd.read_csv("experiment_results.csv")

# Compare baseline vs full_boost
baseline = results[results['variant'] == 'baseline']['CTR']
full_boost = results[results['variant'] == 'full_boost']['CTR']

# Statistical test
t_stat, p_value = stats.ttest_ind(full_boost, baseline)
lift = (full_boost.mean() - baseline.mean()) / baseline.mean() * 100

print(f"CTR Lift: {lift:.2f}%")
print(f"P-value: {p_value:.4f}")
```

## Key Features

- **Scalable**: Processes large video datasets with pre-computed scores
- **Modular**: Clean separation between data prep, training, and editing
- **Consistent**: Temporal propagation maintains visual coherence
- **Flexible**: Supports both heatmap and CSV-based workflows
- **Research-Ready**: Built-in variant generation for causal experiments
- **Well-Documented**: Comprehensive docs and example scripts
- **Production-Quality**: ~4,820 lines of tested Python code

## Technical Highlights

### CLIPSeg Integration
Zero-shot image segmentation for product localization:
```python
from src_alignment.clipseg import CLIPSegModel

model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")
keyword_mask = model.predict(frame, text_prompt="necklace")
```

### Attention Heatmap Generation
4-stage pipeline for semantic richness scoring:
1. Circular patch extraction at multiple visual angles
2. LLaVA semantic scoring of each patch
3. Gaussian smoothing + gamma correction
4. Normalization to [0, 1]

### Temporal Consistency
Optical flow-based propagation:
```python
from src.video_editing import TemporalConsistency

flow_engine = TemporalConsistency(method="raft")
consistent_frames = flow_engine.propagate(
    keyframes=edited_keyframes,
    original_frames=original_frames
)
```

## Performance Considerations

- **Memory**: Batch size of 4 requires ~16GB GPU memory
- **Training Time**: ~2-4 hours on NVIDIA A100 for 100 epochs
- **Inference Speed**: ~1-2 seconds per frame on GPU
- **Storage**: Control tensors are ~10MB per video

## Dependencies

**Core**:
- PyTorch >= 2.0.0
- Diffusers >= 0.21.0
- Transformers >= 4.30.0
- OpenCV, NumPy, Pillow
- Pandas, scikit-learn

**Optional**:
- xformers (memory-efficient attention)
- PEFT (LoRA fine-tuning)
- WandB (experiment tracking)

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data
python examples/01_prepare_data.py --video_dir data/videos --output_dir data/processed

# Train model
python examples/02_train_model.py --config configs/default_config.yaml

# Generate variants
python examples/03_create_variants.py --video_id 7123456789

# Edit videos
python examples/04_edit_video.py --checkpoint checkpoints/best_model.pt
```

## Recent Updates

The framework was recently updated (commit c3b7261) to:
- Work with CSV-based alignment scores instead of frame-by-frame heatmaps
- Support scene-based organization for better scalability
- Simplify control tensor construction for scalar alignment values
- Add `dataset_v2.py` and `experimental_variants_v2.py` for CSV workflow

## Citations

If you use this framework in your research, please cite:

```bibtex
@software{alignment_video_creation,
  title={Attention-Keyword-Aligned Video Ad Manipulation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/alignment_video_creation}
}
```

## License

[Your License Here]

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Summary**: This codebase is a sophisticated video manipulation research platform that enables causal experiments on viewer attention and engagement in video advertisements. It combines state-of-the-art AI models (Stable Diffusion, ControlNet, CLIPSeg) with rigorous experimental design to answer critical questions about what makes video ads effective.

# ControlNet Adapter Function in This Codebase

## Primary Purpose

The ControlNet adapter serves as a **trainable control mechanism** that translates attention-product alignment information into visual modifications in video frames, while the Stable Diffusion model remains frozen.

## The Problem It Solves

**Challenge**: You want to manipulate attention-product alignment in videos (make the product appear more/less aligned with viewer attention) WITHOUT:
- Retraining the entire Stable Diffusion model (billions of parameters, extremely expensive)
- Changing the overall video content, style, or semantics
- Creating visual artifacts or inconsistencies

**Solution**: Add a lightweight "adapter" that learns to inject alignment-based control signals into the frozen Stable Diffusion model.

## How It Works

### 1. Architecture Overview

```
Input Control Tensor                    Stable Diffusion U-Net (FROZEN)
C_t = [M_t, S_t]                       ┌──────────────────────┐
(2 channels)                           │   Encoder Block 1    │
    ↓                                  │   (320 channels)     │
┌──────────────────┐                   ├──────────────────────┤
│  Input Conv      │                   │   Encoder Block 2    │
│  2→64 channels   │                   │   (640 channels)     │
└────────┬─────────┘                   ├──────────────────────┤
         ↓                             │   Encoder Block 3    │
┌──────────────────┐                   │   (1280 channels)    │
│ ControlNet Block │                   ├──────────────────────┤
│   (64 channels)  │──→ZeroConv(320)──→│   + Injection        │
└────────┬─────────┘                   ├──────────────────────┤
         ↓                             │   Middle Block       │
┌──────────────────┐                   │   (1280 channels)    │
│ ControlNet Block │                   ├──────────────────────┤
│  (128 channels)  │──→ZeroConv(640)──→│   + Injection        │
└────────┬─────────┘                   ├──────────────────────┤
         ↓                             │   Decoder Blocks     │
┌──────────────────┐                   │                      │
│ ControlNet Block │                   │                      │
│  (256 channels)  │──→ZeroConv(1280)─→│   + Injection        │
└────────┬─────────┘                   └──────────────────────┘
         ↓                                       ↓
┌──────────────────┐                      Noise Prediction
│ ControlNet Block │
│  (512 channels)  │──→ZeroConv(1280)
└────────┬─────────┘
         ↓
┌──────────────────┐
│   Middle Block   │
│  (512 channels)  │
└──────────────────┘

Legend:
→ : Feature flow
──→: Zero-initialized injection (starts at 0, learns gradually)
```

### 2. Control Tensor Input

The ControlNet receives a 2-channel control tensor:

```python
# C_t has shape (B, 2, H, W)
C_t = [
    M_t,  # Channel 0: Keyword Mask (where product appears)
    S_t   # Channel 1: Alignment Map (attention × keyword overlap)
]
```

**Example visualization**:
```
M_t (Keyword Mask):        S_t (Alignment Map):
┌───────────────┐          ┌───────────────┐
│ 0  0  0  0  0 │          │ 0.0 0.0 0.0   │
│ 0  1  1  1  0 │          │ 0.0 0.4 0.6   │  ← Product region with
│ 0  1  1  1  0 │ ×        │ 0.0 0.8 0.9   │    high attention
│ 0  1  1  1  0 │ Attention│ 0.0 0.5 0.7   │
│ 0  0  0  0  0 │          │ 0.0 0.0 0.0   │
└───────────────┘          └───────────────┘
```

### 3. Feature Extraction Process

```python
# From controlnet_adapter.py lines 189-210

def forward(self, control_tensor):
    # Step 1: Project from 2 channels to 64 base channels
    x = self.input_conv(control_tensor)  # (B, 2, H, W) → (B, 64, H, W)

    # Step 2: Process through encoder blocks
    features = []
    for block in self.encoder_blocks:
        x = block(x)           # ResNet-style blocks
        features.append(x)     # Collect multi-scale features

    # Step 3: Middle block for deepest processing
    x = self.middle_block(x)
    features.append(x)

    # Step 4: Apply zero-initialized convolutions
    zero_features = []
    for i, zero_conv in enumerate(self.zero_convs):
        # Convert to match SD U-Net dimensions (320, 640, 1280, 1280)
        zero_features.append(zero_conv(features[i]))

    return zero_features  # List of features to inject
```

### 4. Zero-Initialized Injection

**Key Innovation**: The injection starts at ZERO impact and gradually learns.

```python
# From controlnet_adapter.py lines 14-25

class ZeroConv(nn.Module):
    """Zero-initialized convolution layer."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, padding=0)

        # CRITICAL: Initialize weights and bias to ZERO
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
```

**Why this matters**:

```
Training Iteration 0:
┌─────────────┐
│ Control     │     ZeroConv     ┌─────────────┐
│ Features    │ ──→ (all zeros) ─→│ SD U-Net    │ = Original SD output
│             │                   │ (frozen)    │
└─────────────┘                   └─────────────┘

Training Iteration 100:
┌─────────────┐
│ Control     │     ZeroConv     ┌─────────────┐
│ Features    │ ──→ (learned)  ─→│ SD U-Net    │ = Modified output
│             │     small values  │ (frozen)    │   (controlled by alignment)
└─────────────┘                   └─────────────┘

Training Iteration 1000:
┌─────────────┐
│ Control     │     ZeroConv     ┌─────────────┐
│ Features    │ ──→ (learned)  ─→│ SD U-Net    │ = Fully controlled output
│             │     full strength │ (frozen)    │
└─────────────┘                   └─────────────┘
```

This prevents the ControlNet from disrupting SD's pre-trained knowledge at the start of training.

### 5. Feature Injection into Stable Diffusion

```python
# From stable_diffusion_wrapper.py lines 203-238

def forward_unet_with_control(self, noisy_latent, timestep, text_embeddings, control_tensor):
    # Get control features at multiple scales
    control_features = self.controlnet(control_tensor)
    # control_features = [f1, f2, f3, f4] with shapes matching SD layers

    # Resize control to match latent dimensions
    control_signal = control_features[0]
    control_signal = interpolate(control_signal, size=noisy_latent.shape[-2:])

    # Inject into U-Net forward pass
    noise_pred = self.unet(
        noisy_latent + control_signal,  # Additive injection
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    return noise_pred
```

## What It Does in Practice

### Scenario: Boosting Alignment

**Input Control Tensor** (Full Boost variant):
```python
# Original alignment: 0.5
# Boosted alignment: 0.75 (1.5x)

M_t = keyword_mask        # Binary mask of product location
S_t = M_t * 0.75          # Boosted alignment map
```

**ControlNet Processing**:
1. Encodes the boosted alignment information
2. Generates features that guide SD to:
   - Enhance visual prominence in product region
   - Increase visual saliency (contrast, brightness)
   - Improve attention-grabbing elements
3. Injects these modifications through zero-convs

**Result**: Video frame where product appears MORE aligned with attention patterns (more visually prominent in the specified region)

### Scenario: Reducing Alignment

**Input Control Tensor** (Reduction variant):
```python
# Original alignment: 0.5
# Reduced alignment: 0.25 (0.5x)

S_t = M_t * 0.25          # Reduced alignment map
```

**Result**: Product region becomes LESS visually prominent

## Training Process

```python
# From stable_diffusion_wrapper.py lines 240-294

def training_step(self, image, control_tensor, text_prompt):
    # 1. Encode clean image to latent space
    latent = self.vae.encode(image)  # (B, 4, H/8, W/8)

    # 2. Add noise (standard diffusion training)
    noise = torch.randn_like(latent)
    timesteps = random_timesteps()
    noisy_latent = add_noise(latent, noise, timesteps)

    # 3. Encode text prompt
    text_embeddings = self.text_encoder(text_prompt)

    # 4. Predict noise WITH ControlNet guidance
    noise_pred = self.forward_unet_with_control(
        noisy_latent,
        timesteps,
        text_embeddings,
        control_tensor  # ← ControlNet processes this
    )

    # 5. Compute diffusion loss
    loss = mse_loss(noise_pred, noise)

    # 6. Backprop ONLY updates ControlNet weights
    # SD U-Net remains frozen
    loss.backward()
```

**What's being learned**:
- How to translate alignment scores → visual modifications
- Which features in product regions drive attention
- How to modify brightness, contrast, visual saliency based on alignment

## Key Advantages

### 1. **Parameter Efficiency**
```
Stable Diffusion U-Net: ~860M parameters (FROZEN)
ControlNet Adapter:     ~50-100M parameters (TRAINABLE)
```
Only 5-10% of total parameters need training.

### 2. **Preserves SD Knowledge**
- SD already knows how to generate high-quality images
- ControlNet just adds "control knobs" on top
- No risk of catastrophic forgetting

### 3. **Interpretable Control**
```python
# You explicitly control what changes
control_tensor = [
    keyword_mask,      # WHERE to modify (spatial control)
    alignment_score    # HOW MUCH to modify (intensity control)
]
```

### 4. **Gradual Learning**
Zero-conv initialization means:
- Training starts from SD's original output
- Gradually learns modifications
- Stable training process

### 5. **Multi-Scale Control**
Features injected at multiple U-Net layers:
- Early layers: Global structure, composition
- Middle layers: Object details, textures
- Late layers: Fine details, edges

## Comparison with Alternatives

### Fine-tuning Entire SD Model
```
Pros: Maximum flexibility
Cons:
  - 860M parameters to train
  - Risk of catastrophic forgetting
  - Requires massive dataset
  - Expensive (~weeks on A100)
```

### ControlNet Adapter (This Approach)
```
Pros:
  - 50-100M parameters to train
  - Preserves SD knowledge
  - Can train on small dataset
  - Fast (~hours on A100)
  - Interpretable control

Cons:
  - Limited to types of control it was trained for
```

### Direct Pixel Editing
```
Pros: Simple, fast
Cons:
  - No semantic understanding
  - Creates artifacts
  - Not temporally consistent
```

## Summary

The ControlNet adapter is a **learned translator** that:

1. **Input**: Takes alignment information (keyword mask + alignment score)
2. **Processing**: Extracts multi-scale control features through a lightweight U-Net
3. **Output**: Injects features into frozen Stable Diffusion to guide generation
4. **Result**: Modifies video frames to match desired attention-product alignment while preserving visual quality

It's like having a "control panel" for Stable Diffusion where you can dial up/down the visual prominence of products based on alignment scores, without retraining the entire model.

The zero-initialization ensures stable training by starting from SD's original behavior and gradually learning the control modifications.

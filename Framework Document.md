# Framework Documentation: Attention–Keyword-Aligned Video Ad Manipulation

## 1. Introduction

This document describes a framework for generating controlled variants of video advertisements by manipulating **frame-level, region-local alignment** between **attention heatmaps** and **keyword heatmaps**. The framework does **not** incorporate CTR in training or generation. Instead, CTR or human responses may be assessed later in lab experiments using the generated stimuli.

The key design constraints are:

- **No training from scratch.**  
  All components must rely on existing SOTA generative models.

- **Local, frame-specific manipulations only.**  
  Only certain frames (e.g., middle section) should be edited.

- **Minimal disruption to the rest of the frame.**  
  Most of the video should remain visually identical to the original.

- **Keyword ≈ Product.**  
  In advertising, “keyword” refers to the product being shown; these are treated identically.

The framework uses:

- **Stable Diffusion (SD)** as a frozen backbone image generator.  
- **ControlNet-style adapters** to inject attention and keyword heatmaps.  
- **Rerender A Video** as a temporal consistency wrapper for video editing.

---

## 2. Data Preparation and Notation

### 2.1 Video and Frame Definitions

For a video advertisement \( v \) with \( T \) frames:

- \( I_t \in \mathbb{R}^{H \times W \times 3} \): RGB frame at time \( t \).
- Keyword text: \( w \) (e.g., "running shoes", "lipstick").

### 2.2 Heatmaps from the Existing Pipeline

From the user’s 4-stage attention–meaning pipeline:

1. **Attention / Meaning Heatmap**  
   \( A_t \in \mathbb{R}^{H \times W} \)  
   Produced via circular patch extraction → LLaVA semantic scoring → Gaussian smoothing + gamma correction.

2. **Keyword Heatmap**  
   \( K_t \in \mathbb{R}^{H \times W} \)  
   Produced via CLIPSeg given text prompt \( w \).

### 2.3 Derived Maps

- **Keyword Mask**  
  \( M_t \in [0, 1]^{H \times W} \)  
  Binarized or softened version of \( K_t \), indicating keyword region(s).

- **Instantaneous Alignment Map**  
  \[
  S_t = \text{normalize}(A_t \odot K_t)
  \]
  High values indicate strong attention on keyword pixels.

- **Background Mask**  
  \( B_t = 1 - M_t \), indicating non-keyword regions.

### 2.4 Control Tensor

The core per-frame conditioning tensor:

\[
C_t = [M_t,\ S_t] \in \mathbb{R}^{H \times W \times 2}
\]

Optionally include \( A_t \) and \( K_t \):

\[
C_t = [M_t,\ S_t,\ A_t,\ K_t] \in \mathbb{R}^{H \times W \times 4}.
\]

This control tensor will be downsampled for the generative model.

---

## 3. Architecture

### 3.1 High-Level Design

The system consists of:

- A **frame-level editor** (Stable Diffusion + ControlNet adapter).
- A **video-level wrapper** (Rerender A Video) for temporal consistency.
- Post-processing for analysis and experiment preparation.

### 3.2 Stable Diffusion Backbone (Frozen)

- Latent diffusion model with:
  - VAE encoder/decoder
  - U-Net denoiser
  - Text encoder for keyword \( w \)

These components remain **frozen** during training.

### 3.3 ControlNet-Style Adapter

A lightweight U-Net that:

1. Takes downsampled control maps \( C_t^\downarrow \) as input.
2. Produces multi-scale features \( F_t^{(l)} \).
3. Injects them into the corresponding SD U-Net layers via **zero-initialized convolutions**.

This gives SD the ability to react to:

- Keyword location (via \( M_t \)),
- Degree of attention alignment (via \( S_t \)).

### 3.4 Injection Mechanism

At each SD U-Net block \( l \):

\[
H_{\text{SD}}^{(l)} \leftarrow H_{\text{SD}}^{(l)} + \text{Conv}_{0}(F_t^{(l)}),
\]

where \( \text{Conv}_{0} \) is initialized with zeros so the adapter starts with no effect.

---

## 4. Training the Frame-Level Editor

### 4.1 Training Goals

- Teach the adapter to understand the structure of \( M_t \) and \( S_t \).
- Maintain visual fidelity when not instructed to change.
- Permit *local* modifications when control maps are modified at inference.
- No CTR-related training.

### 4.2 Stage 1: Identity / Reconstruction Training

For each tuple \( (I_t, C_t, w) \):

1. Encode frame:
   \[
   z_t = \text{VAE\_enc}(I_t)
   \]

2. Sample diffusion timestep \( k \), noise \( \epsilon \):
   \[
   z_t^{(k)} = \alpha_k z_t + \sigma_k \epsilon
   \]

3. Condition SD+ControlNet on:
   - Noisy latent \( z_t^{(k)} \),
   - Text embedding of \( w \),
   - Control maps \( C_t^\downarrow \).

4. Predict noise \( \hat{\epsilon} \) and optionally reconstruct \( \hat{I}_t \).

#### Loss Terms

- **Diffusion Loss**  
  \[
  L_{\text{diff}} = \| \hat{\epsilon} - \epsilon \|_2^2
  \]

- **Image Reconstruction Loss**  
  \[
  L_{\text{recon}} = \| \hat{I}_t - I_t \|_1 + \lambda_{\text{LPIPS}} \cdot \text{LPIPS}(\hat{I}_t, I_t)
  \]

- **Background Preservation**  
  \[
  L_{\text{bg}} = \lambda_{\text{bg}} \cdot \| (\hat{I}_t - I_t) \odot B_t \|_1
  \]

Total loss:

\[
L = L_{\text{diff}} + \lambda_{\text{recon}} L_{\text{recon}} + \lambda_{\text{bg}} L_{\text{bg}}
\]

Only the **ControlNet adapter (and optionally LoRA weights)** are trained.

### 4.3 Optional Stage 1.5: Synthetic Keyword Enhancement

To give the model intuition for “boosting” a keyword region, construct pseudo-targets \( I_t^+ \):

- Apply simple local edits to keyword region \( M_t \) (brightness, contrast, sharpness).
- Train using:
  \[
  L_{\text{recon}}^+ = \| \hat{I}_t - I_t^+ \|_1 + \lambda_{\text{LPIPS}} \text{LPIPS}(\hat{I}_t, I_t^+)
  \]

Mix identity and enhanced samples.

---

## 5. Video-Level Editing and Experimental Manipulation

### 5.1 Temporal Consistency Wrapper

Use a SOTA video editing framework such as **Rerender A Video**, which:

- Applies SD+ControlNet edits to **key frames**.
- Propagates edits to neighboring frames using:
  - Cross-frame patch correspondence,
  - Motion constraints,
  - Feature-space consistency.

This yields a smooth, coherent video.

### 5.2 Designing Temporal Conditions

Define experimental windows:

- \( T_{\text{early}} \)
- \( T_{\text{mid}} \)
- \( T_{\text{late}} \)

Construct variants:

- **Baseline**: no change; \( C'_t = C_t \).
- **Middle alignment boost**: for \( t \in T_{\text{mid}} \),
  \[
  S'_t = \alpha S_t, \quad \alpha > 1
  \]
  and  
  \[
  C'_t = [M_t, S'_t]
  \]

- **Placebo**: manipulate non-keyword region or misalign intentionally.

### 5.3 Key-Frame Editing + Propagation

1. Select key frames (e.g., every 4th frame in an edit window).
2. For each key frame:
   - Encode \( I_t \rightarrow z_t \),
   - Build modified \( C'_t \),
   - Run SD+ControlNet to obtain \( \hat{I}_t \).
3. Rerender A Video propagates edits to all frames.
4. Output fully edited video for each experimental condition.
# AG-XAI Brain Tumor Segmentation (Reference PyTorch Implementation)

This repository contains a **complete reference implementation** of the paper:

**Explainable Brain Tumor Segmentation via Attention-Guided Hybrid CNN–Transformer–Mamba Network**

It is organized to match the paper pipeline:
1. MRI preprocessing
2. Hybrid CNN–Transformer–State-Space encoder
3. Dual-branch attention decoder
4. Gated cross-scale fusion
5. Boundary-aware segmentation head
6. Explainability module
7. Composite-loss training and evaluation

> Note:
> - This is a **research-grade reference implementation** designed to be readable, extensible, and reproducible.
> - The `StateSpace2D` block is a **self-contained Mamba-inspired selective state-space module** implemented without external Mamba dependencies so the code remains portable.
> - If you want to replace it with an official Mamba kernel, you only need to swap the `StateSpace2D` class in `models/blocks.py`.

## Repository Structure

```text
ag_xai_project/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── datasets/
│   └── brain_tumor_dataset.py
├── engine/
│   ├── evaluator.py
│   ├── metrics.py
│   └── trainer.py
├── losses/
│   └── segmentation_losses.py
├── models/
│   ├── ag_xai_net.py
│   ├── blocks.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── explainability.py
│   └── fusion.py
├── utils/
│   ├── ema.py
│   ├── preprocess.py
│   ├── transforms.py
│   └── visualization.py
├── train.py
├── test.py
└── requirements.txt
```

## Installation

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Expected Dataset Layout

The code expects each dataset to be arranged in the following 2D-slice format:

```text
data/raw/
├── kaggle/
│   ├── images/
│   │   ├── sample_0001.png
│   │   └── ...
│   └── masks/
│       ├── sample_0001.png
│       └── ...
├── figshare/
│   ├── images/
│   └── masks/
└── brats2020/
    ├── images/
    └── masks/
```

If you start from 3D BraTS volumes, first export T1ce axial slices and their masks into `images/` and `masks/`.

## Splits

The dataset loader supports CSV split files with columns:

- `image_path`
- `mask_path`
- `dataset`
- `label` (optional)

Example:

```csv
image_path,mask_path,dataset,label
data/raw/kaggle/images/sample_0001.png,data/raw/kaggle/masks/sample_0001.png,kaggle,1
```

If no CSV file is supplied, the loader can scan `images/` and `masks/` directly.

## Main Features

### 1. Preprocessing
Implemented in `utils/preprocess.py`:
- Otsu-based skull stripping
- Morphological cleanup
- Largest connected component extraction
- N4 bias-field correction (if SimpleITK is available)
- Z-score normalization within the brain mask
- Resizing to 256×256

### 2. Hybrid Encoder
Implemented in `models/encoder.py`:
- Multi-stage CNN feature extraction
- Transformer bottleneck reasoning
- Selective state-space (Mamba-inspired) propagation

### 3. Attention-Guided Decoder
Implemented in `models/decoder.py`:
- Criss-cross / axial-style spatial attention
- Channel attention via squeeze-excitation
- Learnable spatial–channel fusion

### 4. Cross-Scale Fusion
Implemented in `models/fusion.py`:
- Attention-guided encoder–decoder interaction
- Dynamic gating between semantic and anatomical features

### 5. Explainability
Implemented in `models/explainability.py`:
- Encoder Grad-CAM
- Decoder attention aggregation
- Composite heatmap generation

### 6. Losses
Implemented in `losses/segmentation_losses.py`:
- Dice loss
- Tversky loss
- Boundary loss using signed distance transform
- Attention consistency loss
- Composite weighted loss

### 7. Training
Implemented in `train.py` and `engine/trainer.py`:
- AdamW
- Warmup + cosine decay
- Mixed precision
- EMA model tracking
- Early stopping on validation Dice

## Default Hyperparameters

The default configuration follows the paper’s setup closely:

- Input size: 256×256
- Batch size: 16
- Epochs: 200
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-5
- Loss weights:
  - Dice = 1.0
  - Tversky = 0.7
  - Boundary = 0.5
  - Attention consistency = 0.3

These can be edited in `configs/default.yaml`.

## Training

```bash
python train.py --config configs/default.yaml
```

## Testing

```bash
python test.py --config configs/default.yaml --checkpoint outputs/best_model.pt
```

## Outputs

Training creates an `outputs/` directory containing:
- `best_model.pt`
- `last_model.pt`
- `metrics.json`
- `train_log.csv`
- sample predictions and explanation maps

## Notes on Reproducibility

- Set a fixed seed in the config file.
- Use the same train/val/test split CSVs across runs.
- For exact reproduction, keep preprocessing enabled and use the same augmentation configuration.

## Suggested Next Extensions

- Replace `StateSpace2D` with official Mamba kernels
- Add multi-class BraTS training
- Add cross-dataset evaluation scripts
- Add WandB or TensorBoard logging
- Add MONAI-based medical metrics and transforms

## Citation

If you use this implementation in your research workflow, please cite the associated paper.

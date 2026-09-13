# Structure-Aware Semi-Supervised Part Segmentation for Crabs

A research-grade implementation of semi-supervised semantic segmentation for crab body
parts (body, left/right claw, left/right legs). The model combines a ResNet50-FPN
encoder, a part-aware adapter, a structure-aware part relation graph (GAT), a prototype
memory, and a multi-scale mask decoder, trained with an EMA teacher-student scheme.

## Features

- **ResNet50-FPN Encoder**: ImageNet-pretrained ResNet50 backbone with a feature pyramid
  producing a stride-4 feature map.
- **Part-Aware Adapter**: A per-part convolutional branch that turns shared features into
  per-part feature maps and graph node embeddings.
- **Structure-Aware Part Graph (GAT)**: A multi-head graph attention network over the five
  parts, guided by an anatomical adjacency matrix and edge types. Node outputs condition
  the part features through a FiLM-style scale/shift modulation.
- **Part Prototype Memory**: Momentum-updated class prototypes with a contrastive loss that
  pulls each part embedding toward its own prototype.
- **Multi-Scale Mask Decoder**: ASPP-style dilated convolutions followed by upsampling
  blocks to produce full-resolution per-part logits.
- **EMA Teacher-Student Semi-Supervised Learning**: An EMA teacher generates pseudo labels
  for unlabeled images; the student is trained on weak/strong augmented views.
- **Uncertainty Weighting**: Monte-Carlo dropout on the teacher estimates per-pixel
  uncertainty that down-weights unreliable pseudo labels.
- **Boundary-Aware Edge Loss**: Edge-weighted regularization that focuses consistency on
  interior regions.
- **Test-Time Augmentation**: Horizontal-flip TTA at inference with class reordering.

## Project Structure

```
code/
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── data_process.py      # Preprocessing: X-AnyLabeling JSON -> multi-class masks; split & resize
├── train.py             # Model, losses, metrics and the semi-supervised training pipeline
├── predict.py           # Inference, measurement report and TTA
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place labeled data (images together with their X-AnyLabeling `.json` files) and unlabeled
images in the directories configured in `config.yaml` (by default `../labeled` and
`../unlabeled`, relative to this `code/` directory).

Supported image extensions: `.jpg`, `.jpeg`, `.png` (case-insensitive).

Label names must match the configured class names:

```
body, left_claw, right_claw, left_legs, right_legs
```

## Usage

Run all commands from this `code/` directory, because the scripts read `config.yaml` by a
relative path.

### 1. Preprocess Data

```bash
python data_process.py
```

This parses the X-AnyLabeling JSON annotations into 5-channel binary masks, resizes images
and masks with aspect-preserving padding to `model.image_size`, splits the labeled data
into train/val sets, and copies resized unlabeled images. Outputs are written under
`data.processed_dir`:

```
processed/
├── train/   (images/, masks/, metadata.json)
├── val/     (images/, masks/, metadata.json)
└── unlabeled/ (images + filelist.txt)
```

### 2. Train

```bash
python train.py
```

Checkpoints (`best_model.pth`, `latest_checkpoint.pth`) are written to `data.output_dir`.
Training resumes automatically if `training.resume` points to an existing checkpoint.
Early stopping is applied when the monitored metric stops improving.

### 3. Inference

```bash
# Single image
python predict.py path/to/image.jpg --checkpoint ./outputs/best_model.pth

# Batch processing
python predict.py path/to/image_dir/ --checkpoint ./outputs/best_model.pth

# Custom save directory / disable TTA
python predict.py path/to/image.jpg --save_dir ./results --no-tta
```

For each image the script saves a segmentation overlay with a measurement report
(`*_result.png`), a per-part panel (`*_parts.png`), and, for batches, an
`all_measurements.csv`. Measurements include per-part area (cm²), perimeter (cm) and
left/right symmetry indices, assuming a fixed scale of `px_per_cm = 194.9`.

## Configuration

Edit `config.yaml` to adjust:

- **data**: `labeled_dir`, `unlabeled_dir`, `output_dir`, `processed_dir`, `train_ratio`, `random_seed`
- **model**: `image_size`, `encoder_embed_dim`, `num_classes`, `class_names`, graph dimensions (`graph_node_dim`, `graph_hidden_dim`, `graph_dropout`)
- **training**: batch size, epochs, learning rate, warmup, EMA decay, loss weights (`lambda_sup`, `lambda_consistency`, `lambda_graph`, `lambda_edge`), pseudo-label thresholds, `resume`, `ablation`
- **eval**: `val_interval`, `save_best`, `metric`

## Ablation Modes

Set `training.ablation` to one of the following. The modes are cumulative.

| Mode | Supervised | Prototype | Pseudo Labels | Graph | Boundary/Uncertainty |
|------|:----------:|:---------:|:-------------:|:-----:|:--------------------:|
| `baseline` | ✓ | | | | |
| `semi` | ✓ | ✓ | ✓ | ✓ | |
| `baseline_graph` | ✓ | ✓ | ✓ | ✓ | |
| `baseline_graph_sym` | ✓ | ✓ | ✓ | ✓ | |
| `baseline_graph_sym_refine` | ✓ | ✓ | ✓ | ✓ | |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ |

- `baseline`: fully supervised training only.
- `semi`: adds prototype contrastive learning and confidence-thresholded pseudo-label
  consistency on unlabeled data.
- `baseline_graph` / `baseline_graph_sym` / `baseline_graph_sym_refine`: enable the part
  relation graph, with symmetry-aware structure and the pseudo-label refinement hook.
- `full`: additionally enables the boundary-aware edge loss and MC-dropout uncertainty
  weighting.

## Loss Functions

```
Total = λ_sup · L_sup
      + λ_consistency · L_cons
      + λ_graph · L_graph
      + λ_edge · L_edge
      + 0.05 · L_proto
```

- **L_sup**: per-class weighted BCE + Dice loss on labeled data.
- **L_cons**: teacher-student consistency (KL divergence) on unlabeled data.
- **L_graph**: graph structure preservation loss (contrast + smoothness + bilateral pairing).
- **L_edge**: boundary-aware edge regularization.
- **L_proto**: prototype contrastive loss from the part prototype memory.

## Metrics

Validation reports mean IoU (mIoU), Dice and F1, plus per-class mIoU. The monitored metric
is selected by `eval.metric`.

## Citation

If you use this code for research, please cite our work (paper coming soon).

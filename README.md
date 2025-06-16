# Temporal Coronary Angiography Segmentation

This repository contains code for the thesis project **"Exploring Temporal Segmentation Consistency in X-ray Coronary Angiography Videos"**. It implements and compares deep learning models (ResNet50 UNet and SegFormer) for binary vessel segmentation in X-ray coronary angiography videos from the CADICA dataset, with a focus on evaluating temporal consistency.

## Project Goal

The objective is to compare convolutional and transformer based architectures under identical training conditions by evaluating:

- Spatial segmentation performance (Dice score, Intersection over Union, Precision, Recall)
- Temporal consistency (Mean Interframe Intersection over Union, Stability Rate)

All models are trained on individual frames and evaluated on sequences to assess coherence between consecutive predictions.

## Repository Structure

```
BEP-temporalsegmentation/
├── CADICA_prepared/           # Example dataset location (can be elsewhere)
│   └── test/
│       ├── A/                 # Input frames
│       └── B/                 # Ground truth masks
├── checkpoints/               # Model checkpoints (saved by Trainer)
├── checkpoints_debug/         # Checkpoints from debug runs
├── logs/                      # TensorBoard logs
├── logs_debug/                # Logs from debug runs
├── overlays_output/           # Visualization overlays from evaluate.py
├── scripts/
│   ├── train.py               # Training and testing script
│   └── evaluate.py            # Standalone evaluation script
├── src/
│   ├── datasets.py            # PyTorch dataset definitions
│   ├── models.py              # Model factory for convolutional and transformer models
│   ├── metrics.py             # Spatial and temporal metric implementations
│   ├── losses.py              # Loss functions such as Tversky
│   └── pl_module.py           # LightningModule and DataModule definitions
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and usage
```

Note: Update `--img_dir` and `--msk_dir` if your data is stored in a different location.

## Installation

### 1. Prerequisites

- Conda for environment management
- NVIDIA GPU with CUDA for training acceleration

### 2. Create Conda Environment

```bash
conda create -n angio_seg python=3.10 -y
conda activate angio_seg
```

### 3. Install PyTorch

Choose the command matching your CUDA version from the official PyTorch website. For example, for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Additional Dependencies

```bash
pip install -r requirements.txt
pip install pytorch-lightning
```

## Usage

Run all commands from the root directory of the repository.

### Training and Testing (train.py)

Use the training script to train a model and optionally evaluate it after training.

```bash
python scripts/train.py --img_dir <input_image_dir> --msk_dir <mask_dir> --arch <model_architecture> [other options]
```

You can specify either a convolutional model such as `resnet50_unet`, or a transformer model such as `segformer_b2` using the `--arch` argument.

Common arguments:

- `--img_dir`, `--msk_dir`: Paths to the input frames and corresponding ground truth masks
- `--arch`: Model architecture (e.g., `resnet50_unet` or `segformer_b2`)
- `--batch_size`, `--lr`, `--max_epochs`: Training hyperparameters
- `--tversky_alpha`, `--tversky_beta`: Weights for Tversky loss
- `--run_test`: Enables evaluation on the test set after training
- `--test_seq_len`: Number of frames to use in sequence-level evaluation
- `--log_dir`, `--checkpoint_dir`, `--exp_name`: Configure logging and model checkpoint saving

Outputs include:

- Trained model checkpoints in the specified checkpoint directory
- Training logs viewable with TensorBoard

**Limitation:** The default training configuration operates on individual frames. However, training with sequence lengths of 10 is recommended to improve temporal consistency, as noted in the thesis.

---

### Evaluation (evaluate.py)

Use the standalone evaluation script to assess a trained checkpoint and generate spatial and temporal metrics, along with visual overlays of predictions.

```bash
python scripts/evaluate.py --checkpoint_path <checkpoint_file> --img_dir <input_image_dir> --msk_dir <mask_dir> [other options]
```

You can use this script for any saved checkpoint, whether from a CNN or a Transformer model.

Common arguments:

- `--checkpoint_path`: Path to a trained checkpoint file (with `.ckpt` extension)
- `--img_dir`, `--msk_dir`: Input image and mask directories
- `--overlay_dir`: Directory where overlay visualizations will be saved (optional)
- `--test_seq_len`: Number of frames to use in each evaluation sequence
- `--num


## TensorBoard

To visualize the training progress, launch TensorBoard:

```bash
tensorboard --logdir tb_logs/
```

Open the URL displayed in the terminal (usually http://localhost:6006).

## Citation and Thesis

This repository supports the following bachelor thesis:

**Kaan Mutlu Celik (2025)**  
*Exploring Temporal Segmentation Consistency in X-ray Coronary Angiography Videos: A Comparative Analysis of Convolutional Neural Networks and Transformer Architectures*  
Eindhoven University of Technology

## License

This project is released under the MIT License. See the LICENSE file for details.

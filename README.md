
# GAN Implementation for MNIST Dataset

A PyTorch implementation of Generative Adversarial Networks (GANs) trained on the MNIST dataset to generate handwritten digits.

© 2024 Noushin Mirnezami. All Rights Reserved.

**IMPORTANT**: This code is provided for educational purposes only. No commercial use, reproduction, or distribution is permitted without explicit permission from the author. Unauthorized copying or plagiarism of this project is strictly prohibited.

## Project Overview

This project implements a GAN architecture to generate realistic handwritten digits by training two competing neural networks:
- A Generator that creates fake images from random noise
- A Discriminator that tries to distinguish between real and fake images

## Project Structure

```
GANs-MNIST/
├── data/                    # MNIST dataset storage
├── models/
│   ├── generator.py        # Generator network architecture
│   └── discriminator.py    # Discriminator network architecture
├── outputs/                # Generated images during training
├── utils.py               # Utility functions (noise generation, loss calculations)
├── train.py              # Main training script
├── requirements.txt      # Project dependencies
└── .gitignore           # Git ignore configuration
```

## Requirements

- Python 3.8+
- GPU environment (recommended)
- Dependencies:
  ```
  torch>=1.9.0
  torchvision>=0.10.0
  numpy>=1.19.5
  matplotlib>=3.4.3
  torchsummary>=1.5.1
  ```

## Model Architecture

### Generator
- Input: Random noise vector (dimension: 100)
- Architecture:
  - Linear(100 → 256) + LeakyReLU(0.2)
  - Linear(256 → 512) + LeakyReLU(0.2)
  - Linear(512 → 1024) + LeakyReLU(0.2)
  - Linear(1024 → 784) + Tanh
- Output: 28x28 grayscale image

### Discriminator
- Input: 28x28 grayscale image (784 dimensions)
- Architecture:
  - Linear(784 → 512) + LeakyReLU(0.2)
  - Linear(512 → 256) + LeakyReLU(0.2)
  - Linear(256 → 1)
- Output: Binary classification (real/fake)

## Training Details

- Batch Size: 128
- Learning Rate: 0.0002
- Optimizer: Adam (β1=0.5, β2=0.999)
- Epochs: 40
- Loss Function: Binary Cross Entropy with Logits

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python train.py
```

Generated images will be saved in the `outputs/` directory every 2 epochs.

## Training Process

The training alternates between:
1. Discriminator Training:
   - Train on real images (target ≈ 1.0)
   - Train on generated fake images (target ≈ 0.0)

2. Generator Training:
   - Generate fake images
   - Try to fool discriminator (target ≈ 1.0)

## Results

The model generates MNIST-like digits after training. Generated samples are saved as PNG files in the `outputs/` directory during training.

## Implementation Details

- Uses PyTorch's built-in MNIST dataset loader
- Implements label smoothing (0.8-1.0 for real, 0.0-0.2 for fake)
- Includes deterministic training setup with fixed random seeds
- Automatic GPU utilization when available


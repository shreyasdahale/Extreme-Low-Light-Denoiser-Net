### Extreme Low-Light Denoiser Net

This repository contains the implementation of a neural network designed to denoise low-light images, using a scaled-down version of the Pyramid Real Image Denoising Network (PriDNet). The network was built and trained using PyTorch.

#### Overview

PRIDNet, or Pyramid Real Image Denoising Network, is a specialized neural network architecture aimed at effectively denoising images captured in extremely low-light conditions. This network leverages a pyramid structure to progressively refine the denoised output at multiple scales, which is essential for capturing both fine details and global structures in the images. This hierarchical approach ensures that the network can manage noise effectively across various levels of detail in the image.

![PRIDNet Structure](download.png)

#### Architecture

The PRIDNet architecture is composed of several key components:

1. **Pyramid Decomposition**: The input image is decomposed into multiple sub-bands through a series of downsampling operations. This process helps in capturing multi-scale features, which are crucial for handling different types of noise.

2. **Feature Extraction and Denoising**: Each level of the pyramid contains convolutional layers that extract features and perform denoising. These layers are designed to work at different scales, allowing the network to focus on both global structures and fine details.

3. **Reconstruction**: The denoised features from each pyramid level are upsampled and combined to reconstruct the final denoised image. This step ensures that the image retains its original resolution while significantly reducing noise.

4. **Residual Layer**: An additional residual layer was added to the network to help retain the original image details while focusing on denoising, thus improving the overall reconstruction quality.

#### Implementation

The implementation in this repository is a scaled-down version of PriDNet, which retains the core ideas but simplifies the network to make it computationally more efficient and easier to train.

- **Framework**: PyTorch
- **Training**: The network was trained for 300 epochs, achieving a testing accuracy of approximately 18.93 dB.

### Getting Started

#### Prerequisites

- Python 3.x
- PyTorch

#### Training & Prediction:
   For this, run:
   ```bash
   python Main.py
   ```
   Adjust the parameters in the `.py` script as needed for your dataset and training configuration.
   The reconstructed images will appear in the `./test/predicted/` directory, they can be checked after prediction. The reconstructed images will be denoised and brightened.

### Results

After training for 300 epochs, the model achieved a testing accuracy of approximately 18.93 dB, indicating a significant reduction in noise for low-light images.

### References

- Original PRIDNet paper: [Pyramid Real Image Denoising Network](https://arxiv.org/pdf/1908.00273)

For more details, please refer to the [project repository](https://github.com/shreyasdahale/Extreme-Low-Light-Denoiser-Net).

#  CelebFaces GAN — Generating Realistic Celebrity Faces with TensorFlow

This repository contains an end-to-end implementation of a **Generative Adversarial Network (GAN)** trained on the **CelebA human face dataset**.  
The project demonstrates how to build, train, and monitor a GAN capable of generating realistic synthetic face images.


##  Features

-  Automated download of the **CelebA Dataset**  
-  Image preprocessing pipeline using `tf.data`  
-  Custom **Discriminator** and **Generator** networks implemented in Keras  
-  Full **GAN training loop** using class-based model subclassing  
-  Custom callbacks for:
  - Saving sample generated images after each epoch  
  - Saving model checkpoints to Google Drive  
  - Saving generator model snapshots  
-  Visualization utilities for inspecting generated faces  
-  Fully reproducible notebook workflow (Colab-friendly)


##  Project Architecture

### **1. Dataset Loading**
CelebA is downloaded automatically using the Kaggle API:
- Unzips and extracts images
- Creates a TensorFlow dataset pipeline
- Normalizes images to floating point values in the range `[0, 1]`

### **2. Discriminator**
A convolutional classifier that outputs a single scalar indicating “real vs fake”.

Key components:
- Conv layers with stride
- LeakyReLU activations
- Dense final layer with sigmoid activation

### **3. Generator**
Takes a **latent vector (100 dims)** and produces a **128×128 RGB face image** using:
- Dense projection + reshape
- Conv2DTranspose layers for upsampling
- BatchNorm + ReLU activations
- Final Conv layer with `sigmoid` output

### **4. GAN Model**
Implemented via subclassing:

Includes:

Separate optimizers for generator & discriminator

Binary cross-entropy losses

Custom metrics for monitoring both losses

5. Callbacks

Includes a custom GANMonitor that:

Generates a grid of sample images each epoch

Saves images to disk

Displays samples inline during training

Also includes:

Google Drive checkpoint saving

Periodic generator model saving

 File Structure
Celeb_Faces.ipynb     # Main notebook file
kaggle.json           # Kaggle API key (DO NOT upload to GitHub!)
/content/celeba       # Extracted dataset (ignored in repo)
generated/            # Saved output images (optional)
models/               # Saved generator weights (optional)

 Requirements

Python 3.8+

TensorFlow 2.x

Keras

NumPy

Matplotlib

Kaggle API (for dataset download)

Install dependencies:

pip install tensorflow keras numpy matplotlib kaggle

 Running the Notebook

Upload kaggle.json to the notebook directory

Ensure permission setup:

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json


Run all cells to:

Download and extract CelebA

Build models

Train GAN

Generate synthetic face images

Training generates sample outputs each epoch and stores checkpoints.
 Model Saving

The notebook saves:

Generator model files (.h5)

Checkpoints for GAN, discriminator, and generator

Generated sample images every epoch

These are saved either locally or in mounted Google Drive.

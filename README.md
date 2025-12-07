# Artificial-Face-Generator-DCGAN
# 🎨 CelebFaces GAN — Generating Realistic Celebrity Faces with TensorFlow

This repository contains an end-to-end implementation of a **Generative Adversarial Network (GAN)** trained on the **CelebA human face dataset**.  
The project demonstrates how to build, train, and monitor a GAN capable of generating realistic synthetic face images.

---

## 🚀 Features

- ✅ Automated download of the **CelebA Dataset**  
- ✅ Image preprocessing pipeline using `tf.data`  
- ✅ Custom **Discriminator** and **Generator** networks implemented in Keras  
- ✅ Full **GAN training loop** using class-based model subclassing  
- ✅ Custom callbacks for:
  - Saving sample generated images after each epoch  
  - Saving model checkpoints to Google Drive  
  - Saving generator model snapshots  
- ✅ Visualization utilities for inspecting generated faces  
- ✅ Fully reproducible notebook workflow (Colab-friendly)

---

## 🧠 Project Architecture

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

```python
class GAN(keras.Model):
    def train_step(self, data):
        ...

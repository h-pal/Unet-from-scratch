# 🧠 U-Net from Scratch for Image Segmentation

This repository provides a custom implementation of the U-Net architecture using PyTorch and a modular training pipeline built for image segmentation tasks. It leverages Hugging Face Datasets and TorchVision for data handling and augmentation, and Weights & Biases for experiment tracking.

---

## 📁 Repository Structure

| File         | Description                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------------- |
| `model.py`   | Contains a full implementation of the U-Net architecture including encoder and decoder modules.    |
| `trainer.py` | Training script that handles data loading, preprocessing, model training, validation, and logging. |

---

## 🧠 Model Architecture: `model.py`

The U-Net architecture is constructed from scratch and consists of:

### 🔹 Encoder (`Unet_Encoder`)

* Sequential convolutional blocks with ReLU activations.
* Each block doubles the number of channels.
* Downsampling is done with `MaxPool2d`.

### 🔹 Decoder (`Unet_Decoder`)

* Upsampling via `ConvTranspose2d`.
* Skip connections from the encoder are concatenated to preserve spatial information.
* Final `Conv2d` layer reduces the channel size to the number of segmentation classes.

### 🔹 Main Network (`Unet`)

* Integrates encoder and decoder.
* Takes `image_channels` (e.g., 3 for RGB) and `n_classes` (e.g., 2 for binary segmentation).
* Output is a class prediction map of reduced spatial resolution (due to unpadded convolutions).

---

## ⚙️ Training Pipeline: `trainer.py`

### 📦 Dataset

* Utilizes [farmaieu/plantorgans](https://huggingface.co/datasets/farmaieu/plantorgans) from Hugging Face.
* Automatically downloads and caches the dataset locally.
* Train/validation splits are handled using `datasets.load_dataset`.

### 🧰 Transforms

* Images are preprocessed using `torchvision.transforms.v2`:

  * Resize to 572x572
  * Random horizontal and vertical flips
  * Random rotations
  * Color jitter for brightness and contrast

* Masks are resized to match input dimensions.

### 🔧 Model Training

* **Loss**: `CrossEntropyLoss`
* **Optimizer**: `Adam`
* **Device**: Automatically selects `cuda` or `cpu`
* **Epoch-wise training** with:

  * Training and validation phases
  * Logging of losses to **Weights & Biases**
  * Saving the best model (`best_model.pt`) based on validation loss

---

## 🧪 Usage

### ✅ Install Dependencies

```bash
pip install torch torchvision datasets wandb
```

### ✅ Train the Model

```bash
python trainer.py
```

You can customize paths and hyperparameters directly in the `trainer.py` script.

---

## 📊 Experiment Tracking

This project integrates with [Weights & Biases](https://wandb.ai):

* Logs training and validation loss
* Automatically tracks system metrics
* Tracks best checkpoint

To use it:

1. Sign up at [https://wandb.ai](https://wandb.ai)
2. Run `wandb login`
3. All logs will sync under project `Practise`, run name `UNET-Trainer`

---

## 📝 Notes

* The U-Net architecture uses `padding=0` in all conv layers, which causes spatial resolution to shrink. Input images are therefore resized to 572x572 to account for this.
* Ensure masks are properly aligned after resize for accurate training.
* You can easily swap in other datasets or extend the model for multi-class segmentation.

---

# Vision Transformer (ViT) from Scratch on CIFAR-10

This project is a hands-on implementation of a **Vision Transformer (ViT)** trained from scratch on the CIFAR-10 dataset. It focuses on understanding and constructing ViT components manually while optimizing the training process on a real GPU environment.

👉 **[Notebook here](https://github.com/prithvirajhi/VisionTransformer/blob/main/FunViTImplementation.ipynb)**

---

## 🔧 Key Features

- **Patch Embedding Layer from Scratch**  
  Implemented using `nn.Conv2d` to both extract patches and project them to the embedding dimension.

- **Modular Vision Transformer**  
  Uses `nn.TransformerEncoder` for the main block, but can be swapped with a fully custom transformer.

- **Training Pipeline**  
  - Data augmentation: `RandomCrop`, `ColorJitter`, `RandomRotation`, `HorizontalFlip`
  - **Label Smoothing** added to cross-entropy loss for regularization
  - **Mixed Precision Training** using `torch.cuda.amp.autocast()` and `GradScaler` for faster and memory-efficient training
  - **Warmup + Cosine Annealing LR Scheduler**
  - **Early Stopping** to avoid overfitting
  - **AdamW Optimizer** for better weight decay behavior

- **~4M parameter ViT model** trained fully from scratch — no pretrained weights used.

---

## 🖥️ Training Environment

- **Platform:** [RunPod.io](https://runpod.io)
- **GPU:** NVIDIA RTX 4000 Ada (Medium tier)
- **Dataset:** CIFAR-10
- **Batch size:** 64
- **Optimizer:** `AdamW`
- **Scheduler:** Custom Warmup + `CosineAnnealingLR`
- **Epochs:** 400  
  *(Training stopped early at epoch 203 via early stopping)*

---

## 📈 Training Results

| Metric                    | Value         |
|---------------------------|---------------|
| Total Parameters          | ~3.99M        |
| Final Training Accuracy   | 64.02%        |
| Final Validation Accuracy | **70.27%**    |
| Epochs Trained            | 203 / 400     |
| Early Stopping Patience   | 10 epochs     |

---

## ⚙️ Bottlenecks & Solutions

| Bottleneck                           | Solution                                                                 |
|--------------------------------------|--------------------------------------------------------------------------|
| **GPU Underutilization**             | Tuned `DataLoader` with `num_workers=12`, `pin_memory=True`, `prefetch_factor=4`. Also tried various batch sizes |
| **Slow convergence**                 | Switched from `Adam` to `AdamW` for better regularization                |
| **Initial training instability**     | Added **Warmup Phase** before cosine annealing LR schedule               |
| **Overfitting on CIFAR-10**          | Used strong **data augmentation** and **Label Smoothing**                |
| **Long training time**               | Used `torch.cuda.amp` (`autocast` + `GradScaler`) for faster FP16 training |
| **Wasted epochs due to plateau**     | Implemented **EarlyStopping** with `patience=10`                         |

---

## 🔭 What’s Next

- Swap out PyTorch `nn.TransformerEncoder` with a **custom Transformer block** (e.g., from [NanoGPT](https://github.com/karpathy/nanoGPT))
- Scale up to **TinyImageNet** or ImageNet subsets
- Add **attention visualization** for model interpretability
- Explore **MixUp**, **CutMix**, and **RandAugment** strategies
- Export model to ONNX or TorchScript for deployment


---

## 📬 Feedback & Collaboration

This repo is a learning-first ViT implementation. Contributions, issues, and discussions are welcome!

Have an idea or want to extend this to larger datasets or custom transformer blocks? Let’s collaborate.

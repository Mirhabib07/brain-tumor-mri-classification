# Brain Tumor MRI Classification (PyTorch)

An end-to-end deep learning pipeline built with **PyTorch** for classifying brain MRI scans into four categories: Glioma tumor, Meningioma tumor, Pituitary tumor, and No tumor.

---

## Project Structure

```text
cnn-image-classifier/
│
├── dataset/             # Local dataset folder (Ignored in Git)
│   ├── Training/        # Training subfolders (glioma, meningioma, notumor, pituitary)
│   └── Testing/         # Testing subfolders (glioma, meningioma, notumor, pituitary)
│
├── config.py            # Global configurations, paths, hyperparameters, and class names
├── dataset.py           # PyTorch Dataset loaders, data augmentation, and transforms
├── model.py             # Custom Convolutional Neural Network (CNN) architecture
├── train.py             # Training loop, validation tracking, and model checkpoint saving
├── evaluate.py          # Full model evaluation (Test accuracy, Classification Report, Confusion Matrix)
├── predict.py           # Single-image or random test inference script
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
from pathlib import Path

# ---- Paths ----
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"          # change if your folder has another name
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

MODEL_PATH = BASE_DIR / "brain_cnn.pth"
PLOT_PATH = BASE_DIR / "accuracy_plot.png"
CM_PATH = BASE_DIR / "confusion_matrix.png"
# ---- Images ----
IMG_SIZE = 128          # every image is resized to 128 x 128
CHANNELS = 1            # 1 = grayscale (MRI scans), 3 = RGB

# ---- Training ----
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
SEED = 42               # makes results repeatable
VAL_SPLIT=0.2
NUM_WORKERS=0
# ---- Classes ----
# Order must match the label numbers: 0, 1, 2, 3
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ---- Segmentation ----
SEG_DIR = BASE_DIR / "seg_dataset" / "data"
SEG_MODEL_PATH = BASE_DIR / "models" / "unet.pth"
SEG_PLOT_PATH = BASE_DIR / "results" / "seg_training_plot.png"
SEG_PRED_PATH = BASE_DIR / "results" / "seg_prediction.png"
SEG_IMG_SIZE = 128
SEG_BATCH_SIZE = 16
SEG_EPOCHS = 30
SEG_LR = 1e-3
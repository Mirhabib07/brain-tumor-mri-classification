import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE,
    SEED, VAL_SPLIT, NUM_WORKERS, CLASS_NAMES,
)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),                      # scales pixels to 0-1
        transforms.Normalize([0.5], [0.5]),         # scales to -1..1
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return train_tf, eval_tf


def get_loaders():
    train_tf, eval_tf = get_transforms()

    full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    full_val = datasets.ImageFolder(TRAIN_DIR, transform=eval_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)

    assert full_train.classes == CLASS_NAMES, (
        f"Class order mismatch: {full_train.classes} vs {CLASS_NAMES}"
    )

    # Reproducible random split of the Training folder
    n = len(full_train)
    n_val = int(n * VAL_SPLIT)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    print("Train/val/test samples:",
          len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)    # expect [32, 1, 128, 128]
    print("Labels:", labels[:10])
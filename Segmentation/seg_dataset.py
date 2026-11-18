import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from config import SEG_DIR, SEG_IMG_SIZE, SEG_BATCH_SIZE, SEED, VAL_SPLIT

def find_pairs():
    pairs=[]
    for mask_path in sorted(SEG_DIR.rglob("*_mask.png")):
        img_path = mask_path.with_name(mask_path.name.replace("_mask", ""))
        if img_path.exists():
            pairs.append((img_path, mask_path))
    return pairs

def split_pairs():
    pairs=find_pairs()
    assert pairs, f"No image/mask pairs found in {SEG_DIR}"
    n = len(pairs)
    n_val = int(n * VAL_SPLIT)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()
    train_pairs = [pairs[i] for i in perm[n_val:]]
    val_pairs = [pairs[i] for i in perm[:n_val]]
    return train_pairs, val_pairs

def load_image(path):
    size=(SEG_IMG_SIZE, SEG_IMG_SIZE)
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float().div(255)
    return ((x-0.5)/0.5).unsqueeze(0)  

def load_mask(path):
    size=(SEG_IMG_SIZE, SEG_IMG_SIZE)
    mask = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)


class SegDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        x = load_image(img_path)
        y = load_mask(mask_path)
        if self.augment:
            if torch.rand(1) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
        return x, y
def get_seg_loaders():
    train_pairs, val_pairs = split_pairs()
    train_loader=DataLoader(SegDataset(train_pairs, augment=True), batch_size=SEG_BATCH_SIZE, shuffle=True)
    val_loader=DataLoader(SegDataset(val_pairs, augment=False), batch_size=SEG_BATCH_SIZE, shuffle=False)
    return train_loader, val_loader   

if __name__ == "__main__":
    train_loader, val_loader = get_seg_loaders()
    print("Train/val:", len(train_loader.dataset), len(val_loader.dataset))
    x, y = next(iter(train_loader))
    print(x.shape, y.shape, "mask values:", y.unique())
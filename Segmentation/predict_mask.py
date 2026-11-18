import sys
import random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import SEG_MODEL_PATH, SEG_PRED_PATH
from Segmentation.unet import UNet
from Segmentation.seg_dataset import split_pairs, load_image, load_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device))
    model.eval()

    true_mask = None
    if len(sys.argv) == 2:
        img_path = sys.argv[1]
    else:
        _, val_pairs = split_pairs()
        img_path, mask_path = random.choice(val_pairs)
        true_mask = load_mask(mask_path)
    print("Image:", img_path)

    x = load_image(img_path)
    with torch.no_grad():
        prob = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu()
    pred = (prob > 0.5).float()

    panels = [("MRI", (x[0] + 1) / 2, "gray")]
    if true_mask is not None:
        inter = (pred * true_mask[0]).sum()
        dice = (2 * inter + 1e-6) / (pred.sum() + true_mask[0].sum() + 1e-6)
        print(f"Dice for this image: {dice.item():.3f}")
        panels.append(("True mask", true_mask[0], "gray"))
    panels.append(("Predicted mask", pred, "gray"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(SEG_PRED_PATH, dpi=150)
    print(f"Saved {SEG_PRED_PATH.name}")


if __name__ == "__main__":
    main()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from config import SEG_EPOCHS, SEG_LR, SEG_MODEL_PATH, SEG_PLOT_PATH, SEED
from Segmentation.seg_dataset import get_seg_loaders
from Segmentation.unet import UNet, bce_dice_loss, dice_iou


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = dice_sum = iou_sum = n = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            d, j = dice_iou(logits, y)
            b = x.size(0)
            loss_sum += loss.item() * b
            dice_sum += d * b
            iou_sum += j * b
            n += b
    return loss_sum / n, dice_sum / n, iou_sum / n


def save_plot(history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_dice"], label="train")
    ax[0].plot(history["val_dice"], label="validation")
    ax[0].set_title("Dice score")
    ax[1].plot(history["train_loss"], label="train")
    ax[1].plot(history["val_loss"], label="validation")
    ax[1].set_title("Loss")
    for a in ax:
        a.set_xlabel("Epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(SEG_PLOT_PATH, dpi=150)
    plt.close()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = get_seg_loaders()
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=SEG_LR)

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best = 0.0

    for epoch in range(1, SEG_EPOCHS + 1):
        tl, td, ti = run_epoch(model, train_loader, device, optimizer)
        vl, vd, vi = run_epoch(model, val_loader, device)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_dice"].append(td)
        history["val_dice"].append(vd)

        print(f"Epoch [{epoch}/{SEG_EPOCHS}] | train loss {tl:.4f} dice {td:.3f} "
              f"| val loss {vl:.4f} dice {vd:.3f} IoU {vi:.3f}")

        if vd > best:
            best = vd
            torch.save(model.state_dict(), SEG_MODEL_PATH)
            print(f"--> Saved best model (val Dice {vd:.3f})")

        save_plot(history)

    print(f"Best validation Dice: {best:.3f}")


if __name__ == "__main__":
    main()
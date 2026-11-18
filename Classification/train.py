import matplotlib
matplotlib.use("Agg")   # save plots to file without opening a window
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from config import EPOCHS, LEARNING_RATE, MODEL_PATH, PLOT_PATH, SEED
from dataset import get_loaders
from model import BrainCNN


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def save_plot(history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, key, title in [(ax[0], "acc", "Accuracy"), (ax[1], "loss", "Loss")]:
        a.plot(history[f"train_{key}"], label="train")
        a.plot(history[f"val_{key}"], label="validation")
        a.set_title(title)
        a.set_xlabel("Epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def train_model():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_loaders()
    model = BrainCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch [{epoch}/{EPOCHS}] | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc*100:.2f}%")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"--> Saved best model with Val Acc: {va_acc*100:.2f}%")

        save_plot(history)   # updated every epoch, so Ctrl+C still leaves a plot

    # ---- Test evaluation with the best checkpoint ----
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    te_loss, te_acc = run_epoch(model, test_loader, criterion, device)
    print(f"Test Accuracy: {te_acc*100:.2f}%")


if __name__ == "__main__":
    train_model()
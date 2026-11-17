import torch
from sklearn.metrics import classification_report, confusion_matrix
from config import MODEL_PATH, CLASS_NAMES
from dataset import get_loaders
from model import BrainCNN


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data loaders (we only need test_loader here)
    _, _, test_loader = get_loaders()

    # Load trained model
    model = BrainCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating model on the entire test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate overall accuracy
    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = (correct / total) * 100

    print(f"\nOverall Test Accuracy: {accuracy:.2f}% ({correct}/{total} images correct)")

    # Detailed Classification Report (Precision, Recall, F1-Score per class)
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate_model()
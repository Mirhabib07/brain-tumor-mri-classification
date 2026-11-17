import torch
import random
from PIL import Image
from pathlib import Path
from config import MODEL_PATH, CLASS_NAMES, TEST_DIR
from dataset import get_transforms
from model import BrainCNN


def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = BrainCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Preprocess image
    _, eval_tf = get_transforms()
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    input_tensor = eval_tf(image).unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        conf, predicted_idx = torch.max(probabilities, 1)

    class_name = CLASS_NAMES[predicted_idx.item()]
    confidence = conf.item() * 100

    print(f"\nImage: {image_path}")
    print(f"Prediction: {class_name} ({confidence:.2f}% confidence)")
    return class_name, confidence


if __name__ == "__main__":
    # Gather all image files from all test subdirectories
    all_image_files = []
    for class_name in CLASS_NAMES:
        class_dir = TEST_DIR / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            all_image_files.extend(files)

    if all_image_files:
        # Pick a completely random image each time you run the script
        sample_path = str(random.choice(all_image_files))
        predict_image(sample_path)
    else:
        print(f"No test images found in {TEST_DIR}. Please check your dataset folder paths.")
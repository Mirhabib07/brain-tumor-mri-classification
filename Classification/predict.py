import sys
import random
import torch
from PIL import Image
from config import MODEL_PATH, CLASS_NAMES, TEST_DIR
from dataset import get_transforms
from model import BrainCNN


def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BrainCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    _, eval_tf = get_transforms()
    image = Image.open(image_path).convert("L")
    x = eval_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    print(f"\nImage: {image_path}")
    for i in probs.argsort(descending=True):
        print(f"  {CLASS_NAMES[i]}: {probs[i].item()*100:.1f}%")
    return CLASS_NAMES[probs.argmax().item()], probs.max().item() * 100


def random_test_image():
    files = []
    for name in CLASS_NAMES:
        d = TEST_DIR / name
        if d.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                files.extend(d.glob(ext))
    return random.choice(files) if files else None


if __name__ == "__main__":
    if len(sys.argv) == 2:
        predict_image(sys.argv[1])
    else:
        path = random_test_image()
        if path is None:
            print(f"No test images found in {TEST_DIR}.")
        else:
            print(f"True class (folder name): {path.parent.name}")
            predict_image(str(path))

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
DATA = Path(
    r"C:\Users\Shreya.G.B\agriculture-ml-internship"
    r"\disease_detection\data\plantvillage"
    r"\data_distribution_for_SVM\test"
)
MODEL_PATH = PROJECT / "models" / "agripredict_improved_model.keras"
LABEL_PATH = PROJECT / "models" / "agripredict_improved_class_names.json"

with open(LABEL_PATH, encoding="utf-8") as f:
    labels = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

folder_order = sorted(
    [p.name for p in DATA.iterdir() if p.is_dir()]
)

images = []
expected_indices = []
per_class_limit = 3

for folder in folder_order:
    folder_path = DATA / folder
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(folder_path.glob(pattern))

    for path in sorted(files)[:per_class_limit]:
        image = Image.open(path).convert("RGB").resize((128, 128))
        array = np.asarray(image, dtype=np.float32)
        images.append(array)
        expected_indices.append(folder_order.index(folder))

if not images:
    raise RuntimeError(f"No test images found in {DATA}")

x = np.stack(images)

probabilities = model.predict(x, batch_size=32, verbose=0)
predicted_indices = np.argmax(probabilities, axis=1)

correct = int(np.sum(predicted_indices == expected_indices))
total = len(expected_indices)

print("\n===== AGRIPREDICT AI VALIDATION =====")
print("Images tested:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)
print(f"Sample accuracy: {correct / total * 100:.2f}%")
print("\nFirst 10 results:")

for i in range(min(10, total)):
    actual = labels[expected_indices[i]]
    predicted = labels[int(predicted_indices[i])]
    status = "CORRECT" if predicted_indices[i] == expected_indices[i] else "INCORRECT"
    print(f"{i + 1}. {status} | Expected: {actual} | Predicted: {predicted}")

print("\nValidation finished.")

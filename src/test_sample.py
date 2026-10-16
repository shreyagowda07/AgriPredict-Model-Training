
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

project = Path(__file__).resolve().parent.parent

image_folder = Path(
    r"C:\Users\Shreya.G.B\agriculture-ml-internship"
    r"\disease_detection\data\plantvillage"
    r"\data_distribution_for_SVM\train\10"
)

model_path = project / "models" / "agripredict_improved_model.keras"
labels_path = project / "models" / "agripredict_improved_class_names.json"

images = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    images.extend(image_folder.glob(pattern))

if not images:
    raise FileNotFoundError(f"No images found in {image_folder}")

image_path = images[0]
print("Testing image:", image_path.name)
print("Expected class: Corn (maize) - healthy")

with open(labels_path, encoding="utf-8") as f:
    labels = json.load(f)

model = tf.keras.models.load_model(model_path)

image = Image.open(image_path).convert("RGB").resize((128, 128))
array = np.asarray(image, dtype=np.float32)
array = np.expand_dims(array, axis=0)

predictions = model.predict(array, verbose=0)[0]
top_indices = np.argsort(predictions)[-3:][::-1]

print("\n===== AGRIPREDICT AI TEST =====")
for index in top_indices:
    print(f"{labels[int(index)]}: {predictions[index] * 100:.2f}%")

print("\nTest finished. Compare the prediction with the expected class.")

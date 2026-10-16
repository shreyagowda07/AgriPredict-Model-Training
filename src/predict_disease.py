
from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

PROJECT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT / "models" / "agripredict_disease_model.keras"
LABELS_PATH = PROJECT / "models" / "agripredict_class_names.json"

IMAGE_SIZE = (128, 128)


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print('python src\\predict_disease.py "path\\to\\leaf_image.jpg"')
        return

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print("ERROR: Image file not found:", image_path)
        return

    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        print("ERROR: Model or class-label file is missing.")
        return

    print("Loading AgriPredict AI model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        print("ERROR: Model output and class labels do not match.")
        return

    image = load_img(image_path, target_size=IMAGE_SIZE)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    print("Analyzing leaf image...")
    predictions = model.predict(image_array, verbose=0)[0]

    best_indices = np.argsort(predictions)[-3:][::-1]

    print("\n========== AGRIPREDICT AI ==========")
    print("Image:", image_path.name)
    print("\nTop predictions:")

    for index in best_indices:
        print(f"{class_names[index]}: {predictions[index] * 100:.2f}%")

    print("\nNote: This is an experimental model.")
    print("Do not use these predictions as a confirmed diagnosis.")


if __name__ == "__main__":
    main()

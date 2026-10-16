
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT / "models" / "agripredict_disease_model.keras"
LABELS_PATH = PROJECT / "models" / "agripredict_class_names.json"
TEST_DIR = (
    PROJECT.parent
    / "disease_detection"
    / "data"
    / "plantvillage"
    / "data_distribution_for_SVM"
    / "test"
)

IMAGE_SIZE = (128, 128)
MAX_IMAGES = 100


def main():
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        raise FileNotFoundError("Model or class names file is missing.")

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test dataset folder not found: {TEST_DIR}")

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = tf.keras.models.load_model(MODEL_PATH)
    correct = 0
    tested = 0

    for class_id, class_name in enumerate(class_names):
        class_dir = TEST_DIR / str(class_id)
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp"
            }:
                continue

            try:
                image = tf.keras.utils.load_img(
                    image_path, target_size=IMAGE_SIZE
                )
                array = tf.keras.utils.img_to_array(image)
                array = np.expand_dims(array, axis=0)

                probabilities = model.predict(array, verbose=0)[0]
                predicted_id = int(np.argmax(probabilities))

                tested += 1
                if predicted_id == class_id:
                    correct += 1

                if tested >= MAX_IMAGES:
                    break

            except Exception as error:
                print(f"Skipped {image_path.name}: {error}")

        if tested >= MAX_IMAGES:
            break

    if tested == 0:
        print("No test images were evaluated. Check the dataset folder.")
        return

    accuracy = correct / tested * 100
    print("\n===== AGRIPREDICT AI TEST =====")
    print(f"Images tested: {tested}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {tested - correct}")
    print(f"Accuracy on this sample: {accuracy:.2f}%")
    print("Note: This is a small sample, not the full test-set accuracy.")


if __name__ == "__main__":
    main()

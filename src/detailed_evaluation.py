
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROJECT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT / "models" / "agripredict_disease_model.keras"
LABELS_PATH = PROJECT / "models" / "agripredict_class_names.json"
TEST_DIR = (
    PROJECT.parent / "disease_detection" / "data" / "plantvillage"
    / "data_distribution_for_SVM" / "test"
)

IMAGE_SIZE = (128, 128)
MAX_IMAGES_PER_CLASS = 50


def main():
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = tf.keras.models.load_model(MODEL_PATH)
    actual, predicted = [], []

    for class_id, class_name in enumerate(class_names):
        class_dir = TEST_DIR / str(class_id)
        if not class_dir.is_dir():
            continue

        count = 0
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
                probabilities = model.predict(
                    np.expand_dims(array, axis=0), verbose=0
                )[0]

                actual.append(class_id)
                predicted.append(int(np.argmax(probabilities)))
                count += 1

                if count >= MAX_IMAGES_PER_CLASS:
                    break
            except Exception as error:
                print("Skipped", image_path.name, "-", error)

    if not actual:
        print("No images evaluated. Check the test dataset path.")
        return

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    matrix = confusion_matrix(
        actual, predicted, labels=list(range(len(class_names)))
    )

    output_dir = PROJECT / "results"
    output_dir.mkdir(exist_ok=True)

    with (output_dir / "detailed_classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix, delimiter=",", fmt="%d"
    )

    accuracy = np.mean(np.array(actual) == np.array(predicted)) * 100
    print("Detailed evaluation completed!")
    print("Images evaluated:", len(actual))
    print(f"Sample accuracy: {accuracy:.2f}%")
    print("Saved: results/detailed_classification_report.json")
    print("Saved: results/confusion_matrix.csv")
    print("This is a capped sample, not full test-set evaluation.")


if __name__ == "__main__":
    main()

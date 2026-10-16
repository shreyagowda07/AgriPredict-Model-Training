
from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT / "results" / "training_report.json"
OUTPUT_FILE = PROJECT / "reports" / "model_evaluation.txt"


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        report = json.load(file)

    accuracy = report["test_accuracy"] * 100
    validation_accuracy = report["best_validation_accuracy"] * 100

    lines = [
        "AGRIPREDICT AI",
        "MODEL EVALUATION REPORT",
        "=" * 40,
        f"Model: {report['model']}",
        f"TensorFlow version: {report['tensorflow_version']}",
        f"Number of classes: {report['number_of_classes']}",
        f"Image size: {report['image_size'][0]} x {report['image_size'][1]}",
        f"Epochs completed: {report['epochs_completed']}",
        "",
        "EVALUATION RESULTS",
        "-" * 40,
        f"Best validation accuracy: {validation_accuracy:.2f}%",
        f"Test accuracy: {accuracy:.2f}%",
        f"Test loss: {report['test_loss']:.4f}",
        "",
        "INTERPRETATION",
        "-" * 40,
        "The model completed training and was evaluated on the test dataset.",
        "The current test accuracy is a baseline and needs improvement.",
        "Further training, data augmentation, and model tuning may help.",
        "",
        "LIMITATION",
        "-" * 40,
        "This is an experimental model. Predictions should not be treated",
        "as confirmed plant disease diagnoses.",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Evaluation report created successfully!")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

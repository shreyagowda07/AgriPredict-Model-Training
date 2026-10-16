
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT / "results" / "confusion_matrix.csv"
LABELS_PATH = PROJECT / "models" / "agripredict_class_names.json"
OUTPUT_PATH = PROJECT / "results" / "figures" / "confusion_matrix.png"


def main():
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    matrix = np.loadtxt(MATRIX_PATH, delimiter=",", dtype=int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("AgriPredict AI - Confusion Matrix")
    plt.xlabel("Predicted Disease")
    plt.ylabel("Actual Disease")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    print("Confusion matrix graph created successfully!")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

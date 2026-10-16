from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path.home() / "agriculture-ml-internship" / "disease_detection" / "data" / "plantvillage"
TRAIN_DIR = DATASET_DIR / "data_distribution_for_SVM" / "train"
TEST_DIR = DATASET_DIR / "data_distribution_for_SVM" / "test"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

def scan(folder):
    results = {}
    for class_dir in sorted(folder.iterdir()):
        if class_dir.is_dir():
            count = sum(1 for f in class_dir.rglob("*") if f.is_file() and f.suffix.lower() in EXTENSIONS)
            if count:
                results[class_dir.name] = count
    return results

if __name__ == "__main__":
    print("AGRIPREDICT AI - DATASET CHECK")
    print("Dataset:", DATASET_DIR)
    print("Train folder exists:", TRAIN_DIR.is_dir())
    print("Test folder exists:", TEST_DIR.is_dir())
    train = scan(TRAIN_DIR) if TRAIN_DIR.is_dir() else {}
    test = scan(TEST_DIR) if TEST_DIR.is_dir() else {}
    print("Training classes:", len(train))
    print("Training images:", sum(train.values()))
    print("Test classes:", len(test))
    print("Test images:", sum(test.values()))
    print("Sample training folders:", dict(list(train.items())[:8]))
    output = PROJECT_DIR / "results" / "dataset_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"train": train, "test": test}, indent=2), encoding="utf-8")
    print("Summary saved:", output)

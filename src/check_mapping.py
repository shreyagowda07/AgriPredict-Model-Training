
from pathlib import Path
from collections import Counter
import re

dataset = (
    Path.home()
    / "agriculture-ml-internship"
    / "disease_detection"
    / "data"
    / "plantvillage"
)

mapping = dataset / "data_distribution_for_SVM" / "train_mapping.txt"

if not mapping.exists():
    print("ERROR: Mapping file not found:", mapping)
    raise SystemExit(1)

counts = Counter()
unparsed = 0

for line in mapping.read_text(encoding="utf-8", errors="replace").splitlines():
    # Original filenames can contain spaces. Find the processed path
    # using its distinctive SVM/train/<number>/ pattern.
    match = re.search(r"SVM/train/(\d+)/", line)

    if not match:
        unparsed += 1
        continue

    folder_number = match.group(1)
    prefix = line[:match.start()]

    # Get the original disease folder from raw/color/<class>/
    original_match = re.search(r"raw/color/([^/]+)/", prefix)

    if not original_match:
        unparsed += 1
        continue

    disease = original_match.group(1)
    counts[(folder_number, disease)] += 1

print("\nAGRIPREDICT AI - MAPPING CHECK")
print("Mapping file:", mapping)
print("Mapped records:", sum(counts.values()))
print("Unparsed records:", unparsed)
print("\nNUMERIC FOLDER -> DISEASE LABEL")

for (folder, disease), count in sorted(
    counts.items(), key=lambda item: (int(item[0][0]), item[0][1])
):
    print(f"{folder} -> {disease} ({count} records)")

folder_labels = {}
for folder, disease in counts:
    folder_labels.setdefault(folder, set()).add(disease)

mixed = {
    folder: labels
    for folder, labels in folder_labels.items()
    if len(labels) > 1
}

print("\nNumeric folders found:", len(folder_labels))
print("Folders with multiple labels:", len(mixed))

for folder, labels in sorted(mixed.items(), key=lambda item: int(item[0])):
    print("CHECK FOLDER", folder, ":", ", ".join(sorted(labels)))

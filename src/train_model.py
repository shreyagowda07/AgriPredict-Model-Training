
from pathlib import Path
from collections import defaultdict
import json
import re

import tensorflow as tf

# --------------------------------------------------
# AGRIPREDICT AI - MODEL TRAINING
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parents[1]

DATASET = (
    Path.home()
    / "agriculture-ml-internship"
    / "disease_detection"
    / "data"
    / "plantvillage"
)

TRAIN_DIR = DATASET / "data_distribution_for_SVM" / "train"
TEST_DIR = DATASET / "data_distribution_for_SVM" / "test"
MAPPING_FILE = DATASET / "data_distribution_for_SVM" / "train_mapping.txt"

MODEL_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

print("\n========================================")
print("       AGRIPREDICT AI TRAINING")
print("========================================")
print("TensorFlow:", tf.__version__)

# Step 1: Check the dataset paths
for folder in (TRAIN_DIR, TEST_DIR):
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

if not MAPPING_FILE.is_file():
    raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")

# Step 2: Build numeric-folder-to-disease mapping
folder_labels = defaultdict(set)

with MAPPING_FILE.open(
    "r", encoding="utf-8", errors="replace"
) as file:
    for line in file:
        match = re.search(r"SVM/train/(\d+)/", line)
        if not match:
            continue

        folder_number = match.group(1)
        prefix = line[:match.start()]
        label_match = re.search(r"raw/color/([^/]+)/", prefix)

        if label_match:
            folder_labels[folder_number].add(label_match.group(1))

if not folder_labels:
    raise ValueError("No disease labels found in the mapping file.")

if any(len(labels) != 1 for labels in folder_labels.values()):
    raise ValueError("Some numeric folders have multiple disease labels.")

label_map = {
    folder: next(iter(labels))
    for folder, labels in folder_labels.items()
}

class_folders = sorted(
    [p.name for p in TRAIN_DIR.iterdir() if p.is_dir()],
    key=int
)

test_folders = sorted(
    [p.name for p in TEST_DIR.iterdir() if p.is_dir()],
    key=int
)

if len(class_folders) != 38:
    raise ValueError(f"Expected 38 training classes, found {len(class_folders)}")

if set(class_folders) != set(test_folders):
    raise ValueError("Training and test folders do not contain matching class IDs.")

if set(class_folders) != set(label_map):
    raise ValueError("Folder IDs do not match the verified disease mapping.")

class_names = [label_map[folder] for folder in class_folders]

if len(set(class_names)) != len(class_names):
    raise ValueError("Duplicate disease labels detected.")

print("Verified disease classes:", len(class_names))
print("Preparing training and validation images...")

# Step 3: Load training images and reserve 15% for validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    labels="inferred",
    label_mode="int",
    class_names=class_folders,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    labels="inferred",
    label_mode="int",
    class_names=class_folders,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=SEED
)

print("Preparing separate test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    labels="inferred",
    label_mode="int",
    class_names=class_folders,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Step 4: Build a convolutional neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),

    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Step 5: Save the best model during training
model_path = MODEL_DIR / "agripredict_disease_model.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Step 6: Evaluate on the separate test dataset
print("\nEvaluating on the separate test dataset...")
best_model = tf.keras.models.load_model(model_path)
test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)

# Step 7: Save disease labels in the model's output order
labels_path = MODEL_DIR / "agripredict_class_names.json"

with labels_path.open("w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

# Step 8: Save training and evaluation results
report = {
    "model": "AgriPredict AI CNN",
    "tensorflow_version": tf.__version__,
    "number_of_classes": len(class_names),
    "class_names": class_names,
    "image_size": list(IMAGE_SIZE),
    "epochs_completed": len(history.history["loss"]),
    "best_validation_accuracy": float(
        max(history.history["val_accuracy"])
    ),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

report_path = RESULTS_DIR / "training_report.json"

with report_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("\n========================================")
print("       TRAINING FINISHED")
print("========================================")
print("Model saved:", model_path)
print("Class labels saved:", labels_path)
print("Training report saved:", report_path)
print(f"Test accuracy: {test_accuracy:.2%}")

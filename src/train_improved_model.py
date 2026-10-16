
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project location
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Actual dataset location
DATA_DIR = Path(
    r"C:\Users\Shreya.G.B\agriculture-ml-internship\disease_detection\data\plantvillage\data_distribution_for_SVM"
)

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

print("\n===== AGRIPREDICT AI: IMPROVED TRAINING =====")
print("Training folder exists:", TRAIN_DIR.exists())
print("Testing folder exists:", TEST_DIR.exists())

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError("Training or testing folder was not found.")

# Load images from the numeric class folders
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names

if class_names != test_ds.class_names:
    raise ValueError("Training and testing class folders do not match.")

num_classes = len(class_names)
print("Number of classes:", num_classes)
print("Folder labels:", class_names)

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# MobileNetV2 transfer learning
base_model = keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(128, 128, 3))
x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nStarting training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)

print("\nEvaluating on the test dataset...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

model_path = MODEL_DIR / "agripredict_improved_model.keras"
labels_path = MODEL_DIR / "agripredict_improved_class_names.json"

model.save(model_path)

with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

report = {
    "model": "MobileNetV2 transfer learning",
    "image_size": list(IMG_SIZE),
    "classes": num_classes,
    "epochs": EPOCHS,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "class_names": class_names
}

with open(RESULTS_DIR / "improved_training_report.json", "w",
          encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n===== TRAINING COMPLETE =====")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("Model saved to:", model_path)
print("Class labels saved to:", labels_path)
print("Report saved to:", RESULTS_DIR / "improved_training_report.json")

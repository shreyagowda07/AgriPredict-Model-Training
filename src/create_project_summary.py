
from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
RESULTS = PROJECT / "results"

def main():
    REPORTS.mkdir(exist_ok=True)

    with (PROJECT / "models" / "agripredict_class_names.json").open(
        "r", encoding="utf-8"
    ) as file:
        classes = json.load(file)

    with (PROJECT / "results" / "training_report.json").open(
        "r", encoding="utf-8"
    ) as file:
        training = json.load(file)

    report_path = REPORTS / "final_project_summary.txt"

    report = f"""
AGRIPREDICT AI
AGRICULTURAL PLANT DISEASE DETECTION
FINAL PROJECT SUMMARY
====================================

1. PROJECT OVERVIEW
AgriPredict AI is a machine learning project designed to classify
plant leaf images into disease categories. It uses a Convolutional
Neural Network (CNN) to learn visual patterns from leaf images.

2. PROJECT OBJECTIVES
- Build an image classification model for plant diseases.
- Prepare and use a labelled image dataset.
- Train and evaluate a CNN model.
- Generate predictions for new leaf images.
- Save evaluation results for further analysis.

3. TECHNOLOGIES USED
- Python
- TensorFlow and Keras
- NumPy
- Scikit-learn
- Matplotlib and Seaborn
- PlantVillage image dataset

4. MODEL CONFIGURATION
- Model type: Convolutional Neural Network (CNN)
- Input image size: 128 x 128 pixels
- Number of classes: {len(classes)}
- Training epochs: 5

5. TRAINING AND TESTING
The model was trained using labelled plant leaf images and evaluated
on a separate test dataset.

Training report data:
{json.dumps(training, indent=2)}

6. DETAILED EVALUATION
The detailed evaluation script processed a capped sample of 1,837
test images and reported 39.19% sample accuracy. This result is
specific to that sample and is not the full test-set accuracy.

A confusion matrix and detailed classification report were saved
under the results folder.

7. PROJECT OUTPUTS
- models/agripredict_disease_model.keras
- models/agripredict_class_names.json
- results/training_report.json
- results/detailed_classification_report.json
- results/confusion_matrix.csv
- results/figures/confusion_matrix.png
- reports/model_evaluation.txt

8. LIMITATIONS
The current model has limited accuracy and may confuse visually
similar plant diseases. Predictions are experimental and should
not be treated as a definitive agricultural diagnosis.

9. FUTURE ENHANCEMENTS
- Increase training epochs and tune model parameters.
- Improve data balancing and image augmentation.
- Evaluate precision, recall, and F1-score for each class.
- Test with diverse real-world leaf photographs.
- Develop a user-friendly application for farmers.

====================================
END OF PROJECT SUMMARY
====================================
"""

    report_path.write_text(report.strip() + "\n", encoding="utf-8")
    print("Final project summary created successfully!")
    print("Saved to:", report_path)

if __name__ == "__main__":
    main()

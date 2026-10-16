# AgriPredict AI — Plant Disease Detection Using MobileNetV2

## 1. Project Overview

AgriPredict AI is a machine learning project that uses deep learning and transfer learning to classify plant leaf images into 38 plant health and disease categories. It aims to demonstrate how AI can support early plant disease identification.

## 2. Objectives

- Build an image classification model using MobileNetV2.
- Prepare and inspect the image dataset.
- Train and evaluate the model.
- Predict plant disease categories from leaf images.
- Display predicted classes and confidence scores.
- Provide a user-friendly interface for image-based prediction.

## 3. Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Pillow
- JSON
- Streamlit
- PlantVillage dataset

## 4. Model Details

- Model architecture: MobileNetV2
- Learning approach: Transfer learning
- Input image size: 128 × 128 pixels
- Image channels: RGB
- Number of classes: 38
- Training epochs: 5

## 5. Dataset

The project uses labelled plant leaf images from the PlantVillage dataset. The dataset includes multiple plant species and disease categories, including apples, grapes, potatoes, corn, and tomatoes.

## 6. Model Evaluation

The project includes model evaluation scripts, a confusion matrix, and classification reports.

The earlier CNN baseline recorded 55.12% test accuracy. The improved MobileNetV2 model has separate evaluation results; refer to the latest evaluation report before reporting its final accuracy.

Note: Evaluation results depend on the dataset split and evaluation procedure. The model should be tested on an independent dataset before making claims about real-world performance.

## 7. Project Structure

- `models/` — Trained models and class-name mappings.
- `src/` — Training, prediction, validation, and application scripts.
- `results/` — Evaluation reports, dataset summaries, and confusion matrix.
- `reports/` — Project summary and model evaluation reports.
- `README.md` — Project documentation.

## 8. How to Run the Project

### Install dependencies

Create and activate a Python virtual environment if required, then install the dependencies used by the application.

```cmd
python -m venv venv
venv\Scripts\activate
pip install streamlit tensorflow numpy pillow
```

### Launch the web application

From the project root directory, run:

```cmd
streamlit run src/app.py
```

Open the local URL displayed in the terminal to use the application.

## 9. Limitations

- Predictions may be incorrect for unfamiliar or real-world leaf images.
- Dataset images may not represent every field condition.
- Model confidence scores do not guarantee a correct diagnosis.
- The application is an experimental project, not a replacement for expert agricultural advice.

## 10. Future Enhancements

- Evaluate the model on an independent test dataset.
- Improve performance using data augmentation and model tuning.
- Add more plant species and disease categories.
- Improve the user interface and prediction explanations.
- Deploy the application for wider testing.

## Conclusion

AgriPredict AI demonstrates an AI-assisted plant disease classification workflow using MobileNetV2 and transfer learning. The project includes trained models, prediction scripts, evaluation results, and a Streamlit interface. It provides a foundation for further experimentation in agricultural AI.

**Disclaimer:** This model is for educational and experimental purposes. Always confirm important crop disease decisions with a qualified agricultural expert.

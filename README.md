\# AgriPredict AI — Plant Disease Detection Using CNN



\## 1. Project Overview



AgriPredict AI is a machine learning project that uses a Convolutional Neural Network (CNN) to classify plant leaf images into 38 plant health and disease categories. The project aims to demonstrate how deep learning can support early plant disease identification.



\## 2. Objectives



\- Build a CNN model for plant leaf classification.

\- Prepare and inspect the image dataset.

\- Train and evaluate the model.

\- Predict plant categories from new leaf images.

\- Display the top three predicted classes and their model confidence scores.



\## 3. Technologies Used



\- Python

\- TensorFlow and Keras

\- NumPy

\- Pillow

\- JSON

\- PlantVillage image dataset



\## 4. Model Details



\- Model: Convolutional Neural Network (CNN)

\- Input image size: 128 × 128 pixels

\- Image channels: RGB

\- Number of classes: 38

\- Training epochs: 5



\## 5. Dataset



The project uses the PlantVillage dataset, which contains labelled plant leaf images. The dataset is organised into training and testing sets. The model supports 38 classes covering plants such as apples, grapes, potatoes, corn, and tomatoes.



\## 6. Model Evaluation



\- Best validation accuracy: 55.64%

\- Test accuracy: 55.12%

\- Test loss: 1.5233



These results represent an initial baseline. Further training, data augmentation, and model improvements are needed to improve classification performance.



\## 7. How to Run the Project



Open Command Prompt in the project directory and run:



```cmd

python src\\predict\_disease.py "FULL\_PATH\_TO\_LEAF\_IMAGE.jpg"

```



Replace the example path with the actual path to a leaf image.



The trained model and class labels are stored in the `models` folder.



\## 8. Output



The prediction program displays the image filename and the top three predicted plant classes with their confidence scores.



\## 9. Limitations



The model is experimental and has not been validated for reliable field diagnosis. Predictions may be incorrect, particularly for images taken in real-world conditions.



\## 10. Future Enhancements



\- Improve model accuracy through data augmentation and model tuning.

\- Evaluate per-class performance using a confusion matrix and classification report.

\- Develop a user-friendly web interface for image uploads.

\- Test the model using leaf images collected outside the training dataset.



\## Conclusion



AgriPredict AI demonstrates an end-to-end plant image classification workflow, from dataset inspection and CNN training to evaluation and prediction. It provides a foundation for further development of AI-assisted plant disease detection.

### Detailed Sample Evaluation

A separate evaluation script processed a capped sample of 1,837 test images and reported 39.19% sample accuracy. This is a separate evaluation result and should not be confused with the 55.12% test accuracy reported during model training.

The project also generates a confusion matrix and a per-class classification report. The model is an experimental baseline and requires further improvement.

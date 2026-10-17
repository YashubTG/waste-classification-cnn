# ♻️ Waste Classification Using Deep Learning and Computer Vision

## 📌 Project Overview

Waste management is an important environmental challenge. Correctly identifying and separating waste materials can help improve recycling and reduce the amount of waste sent to landfills.

This project uses **Deep Learning and Computer Vision** to classify waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

A Convolutional Neural Network (CNN) is used to learn visual patterns from waste images and predict the category of an uploaded image.

The trained model is integrated into a Flask web application, allowing users to upload an image and receive a predicted waste category along with the model's confidence score.

## 🎯 Project Objectives

- Build an image classification system using deep learning.
- Apply image preprocessing and data augmentation.
- Train and evaluate a CNN model for waste classification.
- Develop a web interface using Flask.
- Make image predictions through a simple, user-friendly application.

## 🗂️ Dataset

The project uses a resized waste image dataset containing six classes:

| Class | Description |
|---|---|
| Cardboard | Cardboard waste |
| Glass | Glass waste |
| Metal | Metal waste |
| Paper | Paper waste |
| Plastic | Plastic waste |
| Trash | Other waste classified in the dataset |

The images are resized to **128 × 128 pixels** and normalized before being passed to the model.

Data augmentation is used during training to introduce variations in the training images and help the model generalize.

**Note:** The dataset is not included in this repository.

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- Computer Vision
- NumPy
- Pillow
- Flask
- Jupyter Notebook
- HTML and CSS
- GitHub

## 🧠 Model Development

The project follows an end-to-end machine learning workflow:

1. **Exploratory Data Analysis (EDA):** Examine the dataset, image dimensions, and class distribution.
2. **Data Preprocessing:** Resize images, normalize pixel values, and prepare training, validation, and test datasets.
3. **Data Augmentation:** Apply transformations to training images to increase visual variation.
4. **Model Training:** Train and experiment with a CNN-based image classification model.
5. **Model Evaluation:** Evaluate classification performance on a held-out test dataset.
6. **Deployment:** Integrate the trained model into a Flask application for image predictions.

## 📊 Model Performance

The selected model achieved approximately **59.47% test accuracy** in the current experiment.

The model is a work in progress, and further improvements may be possible through architecture tuning, improved data handling, and additional training experiments.

## 🌐 Web Application

The Flask application allows users to:

1. Upload an image of waste.
2. Submit the image for prediction.
3. View the predicted waste category.
4. View the model's confidence score.
5. See the uploaded image alongside the result.

## 📁 Project Structure

```text
waste-classification-cnn/
│
├── flask_app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── uploads/
│
├── 01_eda.ipynb
├── 02_Data_Preprocessing.ipynb
├── deployment.ipynb
├── waste_classifier.keras
├── requirements.txt
├── .gitignore
└── README.md

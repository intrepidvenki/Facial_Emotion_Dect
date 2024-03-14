# Facial Emotion Recognition using Convolutional Neural Networks

This repository contains code for building and training a Convolutional Neural Network (CNN) to recognize facial emotions. The dataset consists of facial images categorized into seven emotion classes: angry, disgust, fear, happy, neutral, sad, and surprise.

## Dataset

The dataset used for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).

## Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Model Architecture](#model-architecture)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Model Inference](#model-inference)
6. [Visualization](#visualization)

## Data Preprocessing

- The `createdataframe` function is used to create a DataFrame containing image paths and corresponding labels for both training and validation datasets.
- Images are loaded and preprocessed using the `extract_feature` function. Grayscale images are loaded and reshaped to the required input shape for the CNN model.

## Model Architecture

- A CNN model is built using Keras Sequential API.
- The model consists of convolutional layers followed by max-pooling and dropout layers to prevent overfitting.
- After flattening the feature maps, fully connected layers are added for classification.
- The output layer consists of seven units (one for each emotion class) with softmax activation.

## Model Training

- The model is compiled with the Adam optimizer and categorical cross-entropy loss function.
- Training data is normalized and one-hot encoded.
- Model training is carried out on the training dataset.

## Model Evaluation

- After training, the model summary is printed to inspect the architecture and parameters.
- Additionally, the model is evaluated on the test dataset to measure its performance.

## Model Inference

- A trained model is loaded from JSON and weights files.
- An image is passed through the model for inference, and the predicted emotion label is printed.

## Visualization

- The original image along with the predicted emotion label is displayed using matplotlib.

Happy coding!

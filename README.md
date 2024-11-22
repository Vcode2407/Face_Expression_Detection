# Facial Emotion Recognition Using CNN

## Project Overview
Facial emotion recognition is a crucial component of human-computer interaction, allowing machines to identify and respond to human emotions. This project leverages Convolutional Neural Networks (CNNs) to classify facial expressions into seven categories: angry, disgust, fear, happy, neutral, sad, and surprise. The model was trained using a comprehensive dataset, achieving an accuracy of 64.00%.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Environment and Libraries](#environment-and-libraries)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/aadityasinghal/facial-expression-dataset). It contains images labeled with seven different emotion categories.

## Environment and Libraries
The project was developed and executed in the Kaggle environment. The following libraries were used:
- `pandas`
- `numpy`
- `matplotlib`
- `keras`
- `tensorflow`
- `scikit-learn`

## Model Architecture
The Convolutional Neural Network (CNN) model includes the following layers:
- Convolutional Layers
- Activation Layers (ReLU)
- Pooling Layers (Max Pooling)
- Fully Connected Layers
- Output Layer (Softmax Activation)

Transfer learning techniques were also employed using pre-trained models like VGG16, ResNet-50, and MobileNet.

## Results
The model achieved a training accuracy of 98% and a validation accuracy of 64%. Detailed evaluation metrics, including precision, recall, and F1-score, are provided for each emotion category. The results indicate the model's capability to accurately classify facial expressions, with room for further refinement.

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/facial-emotion-recognition.git
   cd facial-emotion-recognition

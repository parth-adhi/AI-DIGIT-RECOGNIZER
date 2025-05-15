# AI-DIGIT-RECOGNIZER
# AI-Based Handwritten Digit Recognition

## Objective:
Build a neural network model to recognize handwritten digits using the MNIST dataset.

## Features:
- Trains a CNN on MNIST data
- Predicts digits from user-drawn input images
- Includes optional GUI support

## Files:
- `train_model.py`: Trains the CNN model
- `predict_digit.py`: Predicts digits from an image
- `model.h5`: Trained model (generated after running train_model.py)

## Requirements:
- Python 3.x
- TensorFlow
- Pillow (PIL)
- NumPy

## How to Run:
1. Train model using `python train_model.py`
2. Test model with digit image: `python predict_digit.py`

# Neural-Network-Digits-Detection
A neural network approach to detecting different types of digits in order to identify them.


This project implements and evaluates a digit classification system using both Logistic Regression and a Convolutional Neural Network (CNN) on the mfeat-pix.txt dataset. It includes functionality for data preprocessing, training, evaluation, and hyperparameter tuning using k-fold cross-validation.

Dataset:
File: mfeat-pix.txt
Content: 16x15 pixel grayscale images of handwritten digits (0–9) with 200 samples per digit.
The first 100 per digit are used for training, and the remaining 100 for testing.


The Structure of The Code:
1. Data Preprocessing
Reads mfeat-pix.txt and splits into training (1000 samples) and test (1000 samples) sets.
Normalizes data and reshapes to 16x15 for CNN input.
Wraps data into a custom PyTorch Dataset class.

2. Logistic Regression
Baseline classifier using scikit-learn's LogisticRegression.
Trains on flattened 240-length vectors.
Evaluates and prints train/test accuracy over 100 iterations.

3. CNN Architecture
Customizable depth: 1–3 convolutional layers.
MaxPooling and ReLU activations after each convolution.
Fully connected (dense) layer with dropout.
Cross-entropy loss and Adam optimizer used for training.

4. Training Functions
train_epoch: Trains one epoch.
test_model: Evaluates model on validation/test set.
kfold_cv: Performs k-fold cross-validation for CNN with specified hyperparameters.

5. Hyperparameter Tuning
Each of the following is tuned using k-fold CV: Learning rate, Number of convolutional layers, Number of feature maps, Size of dense layer.
Plots of training/validation loss are generated for each variation.

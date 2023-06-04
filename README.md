# FashionMNIST Classification with Deep Learning Models

This repository contains code for classifying the FashionMNIST dataset using three different deep learning models implemented in PyTorch. The models are designed to classify the images into their respective labels using different architectural configurations.

## Dataset

The FashionMNIST dataset is a popular benchmark dataset in the field of computer vision. It consists of 60,000 grayscale images of fashion products belonging to 10 different classes. The dataset is divided into a training set of 50,000 images and a test set of 10,000 images. Each image is a 28x28-pixel square.

## Requirements

To run the code in this repository, you need to have the following dependencies installed:

- PyTorch (version >= 1.7.0)
- NumPy
- Matplotlib

You can install the required packages by running the following command:


## Files

The repository contains the following files:

- `FashionMNIST_Image_classifier.ipynb`: Jupyter Notebook demonstrating the implementation and evaluation of all 3 deep learning models.
- `README.md`: This file, providing an overview of the repository.

## Models

The repository includes three different models for FashionMNIST classification:

1. Linear Model: This model consists of a single linear layer that directly maps the input image pixels to the output labels. It serves as a basic baseline model.

2. Nonlinear Model: This model extends the linear model by adding a non-linear activation layer (e.g., ReLU) after the linear layer. The non-linear activation helps the model capture more complex relationships between the input features.

3. Convolutional Model: This model utilizes convolutional layers to extract hierarchical features from the input images. It includes a combination of convolutional layers, pooling layers, and fully connected layers to learn discriminative representations.

Each model is implemented in a separate Jupyter Notebook for easy understanding and comparison of the results.

## Usage

To run the notebooks and train the models, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies as mentioned in the "Requirements" section.
3. Open the desired Jupyter Notebook (e.g., `linear_model.ipynb`) using Jupyter or JupyterLab.
4. Execute the code cells in the notebook to train and evaluate the corresponding model.

You can experiment with different hyperparameters, architectures, or optimization techniques to further improve the model's performance.

## Results

The notebooks display the results of training and evaluating the models on the FashionMNIST dataset. The evaluation metrics include accuracy, and  confusion matrices. You can compare the performance of the different models and analyze the effect of architectural choices on the model's accuracy and convergence.

## Conclusion

This repository provides a comprehensive overview of three different deep learning models for FashionMNIST classification. It serves as a starting point for understanding and experimenting with deep learning architectures on image classification tasks. Feel free to explore and modify the code to suit your requirements.

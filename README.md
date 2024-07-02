# Handwritten Digit Recognition Using MNIST Dataset

Handwritten Digit Recognition project predicts the images and labels of digits from 0-9. In this project, a neural network model is built with tensorflow, keras, and python programming language. The dataset used is MNIST dataset which is a collection of 28 x 28 images of handwritten digits from 0-9.


### Steps to build Neural Network Model
1. Importing the required libraries
   1. Tensorflow and matplotlib are imported
2. Loading dataset
   1. MNIST dataset is loaded into train_images, train_labels, test_images, test_labels
3. Scaling the dataset
   1. Scaling dataset is important because it reduces the wide range of differences between data in features
4. Build the neural network model
   1. The input layer is built using Flatten()
   2. The hidden layers are built using 128 neurons and rectified linear unit as activation function to generate activation numbers
   3. The output layer is constructed using 10 neurons with Softmax activation function to generate the output as the possibility of being the number within the range of 0-9 is predicted.
5. Compile the model
   1. Calculate loss (degree of inaccuracy of the model) using sparse_categorical_crossentropy and accuracy as metrics
6. Train the model with 3 epochs 
7. Evaluate the model
8. Predict the test dataset 
   1. Predict the first image and label of the digit in the test images.

References
1. https://www.youtube.com/watch?v=wQ8BIBpya2k
   
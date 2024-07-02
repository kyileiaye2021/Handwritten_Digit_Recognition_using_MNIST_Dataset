# Handwritten Digit Recognition 
# Basic Neural Network Construction
# Using MNIST Dataset

#Importing necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist # 28 x 28 images of hand-written digits of 0-9

#Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Scaling the dataset
train_images = tf.keras.utils.normalize(train_images)
test_images = tf.keras.utils.normalize(test_images)

#Define the model (Specify how to compute output)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) #adding input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #adding first hidden layer (Dense is used with 128 neurons and relu activation)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #adding second hidden layer (Dense is used with 128 neurons and relu activation)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #adding output layer

#Compile the model (Specify cost and loss)
model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy']) #reduce the loss (degree of inaccuracy of the model)

#Fitting the model (Train on data to minimize cost)
model.fit(train_images, train_labels, epochs=3) 

#Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

#Predicting first image of the test_image list
prediction = model.predict(test_images)
plt.imshow(test_images[1])
plt.title("First Image of Test Image")
plt.show()

print(f"Predicted label: {prediction[1].argmax()}")
print(f"Actual label: {test_labels[1]}")
print()
#plt.imshow(train_images[0], cmap = plt.cm.binary)
#plt.show()
#print(train_images[0])
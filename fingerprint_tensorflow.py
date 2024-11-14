'''
This is a second attempt using tensforflow to define the model instead of cv2 and pytorch
Author Aidan LaFond 
References 
https://medium.com/ai-techsystems/fingerprint-pattern-classification-using-deep-learning-9eb93757df11
https://www.tensorflow.org/guide
'''

import tensorflow as tf
from keras import layers, models
import numpy as np
import os
from skimage import io
from sklearn.model_selection import train_test_split

# Load images from directories
def load_images_from_directory(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image = io.imread(os.path.join(directory, filename), as_gray=True)
            images.append(image)
            labels.append(filename)
    return np.array(images), labels

# Preprocess images
def preprocess_images(images):
    images = images / 255.0  # Normalize to [0, 1]
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    return images

# Load and preprocess images
ref_images, ref_labels = load_images_from_directory('NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/reference')
sub_images, sub_labels = load_images_from_directory('NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/subject')

ref_images = preprocess_images(ref_images)
sub_images = preprocess_images(sub_images)

# Split data into training and testing sets
ref_train, ref_test, sub_train, sub_test = train_test_split(ref_images, sub_images, test_size=0.2, random_state=42)

# Define the neural network model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
model.fit(ref_train, sub_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(ref_test, sub_test)
print(f"Test Accuracy: {accuracy}")

# Calculate the EER
def calculate_eer(ref_test, sub_test, model):
    predictions = model.predict(ref_test)
    false_accepts = 0
    false_rejects = 0
    total_images = len(ref_test)

    for i in range(total_images):
        if predictions[i] < 0.5 and sub_test[i] == 1:
            false_rejects += 1
        elif predictions[i] >= 0.5 and sub_test[i] == 0:
            false_accepts += 1

    frr = false_rejects / total_images
    far = false_accepts / total_images

    eer = (frr + far) / 2
    return eer

eer = calculate_eer(ref_test, sub_test, model)
print(f"Equal Error Rate (EER): {eer}")

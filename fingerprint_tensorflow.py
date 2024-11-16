'''
This file uses tensorflow to define a siamese neural network for image similarity
References:
https://github.com/keras-team/keras-io/blob/master/examples/vision/siamese_contrastive.py
Author: Aidan LaFond 
'''

import random
import numpy as np
import keras
from keras import ops
from keras import utils
import matplotlib.pyplot as plt


epochs = 10
batch_sz = 16
margin = 1  # Margin for contrastive loss.


#load the data from the folders. Categorized into reference and subject
#also be sure to introduce the train/test split
train_ds = utils.image_dataset_from_directory(
    directory="NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/",
    labels='inferred',
    label_mode='categorical',
    seed=123,
    subset="training",
    validation_split=.25,
    batch_size=32,
    image_size=(512,512)    
)

test_ds = train_ds = utils.image_dataset_from_directory(
    directory="NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/",
    labels='inferred',
    label_mode='categorical',
    seed=123,
    subset="validation",
    validation_split=.25,
    batch_size=32,
    image_size=(512,512)    
)

#train the model
model = keras.applications.Xception(weights=None, input_shape=(512,512,3), classes=2)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(train_ds, epochs=10, validation_data=test_ds)

#evaluate the model
model.summary()
results = model.evaluate(train_ds, batch_size=32)
print(f"Accuracy: {results}")






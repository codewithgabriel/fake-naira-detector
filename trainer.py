# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 07:30:14 2025

@author: HP
"""

import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# load  the dataset using Keras 

from keras.api.utils import image_dataset_from_directory

IMG_SIZE = (64, 64)
trainset  = image_dataset_from_directory(
    "dataset/train",
    color_mode='rgb',
    image_size= IMG_SIZE,
    batch_size=16,
    label_mode='categorical',
    crop_to_aspect_ratio=True,
    shuffle=False,
    )




#%% 

from keras.api.models import Model 
from keras.api.layers import Input, Dense, Conv2D , MaxPool2D, Flatten, BatchNormalization

inputs = Input(shape=(64, 64, 3))

x = Conv2D(32, kernel_size=3, activation='relu')(inputs)
x = MaxPool2D(strides=2)(x)
x = Flatten()(x)



full_conn = Dense(128, activation='relu')(x)
outputs = Dense(2, activation='softmax')(full_conn)

model = Model(inputs=inputs , outputs=[outputs])
model.summary()

model.compile(optimizer='adam' , loss='categorical_crossentropy', metrics=['accuracy' , 'precision'])

_history = model.fit(trainset, epochs=9)

model.save('fake-naira-notes-detector.h5')
#%% 

history = _history.history
plt.plot(history['loss'] , color='red' , label='loss')
plt.title("Train Loss vs Validation Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show() 


history = _history.history
plt.plot(history['accuracy'] , color='blue' , label='accuracy')
plt.title("Train Loss vs Validation Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#%% 

from sklearn.metrics import classification_report , accuracy_score , precision_score 
from keras.api.models import load_model 
_model = load_model("fake-naira-notes-detector.h5")

y_class_names = trainset.class_names

y_true =  [label for _ , label in trainset.as_numpy_iterator()] 
y_true =  np.concatenate(y_true  , axis=0) 
y_true = np.argmax(y_true , axis=1)



# prediction
y_pred = _model.predict(trainset)
y_pred = np.argmax(y_pred, axis=1)

print( classification_report(y_true, y_pred, target_names=y_class_names) )
print("accuracy: " , accuracy_score(y_true, y_pred))

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 04:52:29 2025

@author: HP
"""
#%%
import json 
import numpy as np
import cv2 
import os

#%%
root_dir = 'project-9'
image_dir = f"{root_dir}/images"
annotation_file = f'{root_dir}/result.json'

#%%
with open(annotation_file, 'r') as annotation_data: 
    dataset  = json.load(annotation_data)

images = { img['id']: os.path.join(root_dir, img['file_name'] ) for img in dataset['images']}
annotations = dataset['annotations']
catagories = {cat['id']: cat['name'] for cat in dataset['categories'] }
NUM_CLASSES = len(catagories)
    
#%%
IMG_SIZE = (244,244)

X = []
Y = []


for annotation in annotations: 
    image_file = images[annotation['image_id']]
    bbox = annotation['bbox']
    category_id = annotation['category_id']
    
    # read image 
    img = cv2.imread(image_file , 0)
    # resize image and normalize it
    resized_img = cv2.resize(img, IMG_SIZE) / 255.0
    
    X.append(resized_img)
    Y.append([*bbox, category_id])


X = np.array(X)
Y = np.array(Y)


#%% 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
ct = ColumnTransformer([("class" , OneHotEncoder() , [4])] , remainder='passthrough')

ct.fit(Y)
Y = ct.transform(Y)

#%%
y_bbox = Y[: , NUM_CLASSES:]
y_class = Y[: , :NUM_CLASSES]

#%%
from sklearn.preprocessing import StandardScaler 
bbox_sc = StandardScaler() 
bbox_sc.fit(y_bbox)

#%%
Y = { 
     'bbox': bbox_sc.transform(y_bbox),
     'class': y_class
     }


#%% 
from sklearn.model_selection import train_test_split 

#X_train = X[: , 10:]
#y_train = Y[: , 10:]

#%% 
import tensorflow as tf
from keras.api.models import Model, Sequential
from keras.api.layers import LSTM, Conv1D , Dense, Input, Dropout, BatchNormalization
#%%


#%% 

inputs = Input(shape=(244, 244))
x = LSTM(100, activation='relu', return_sequences=True)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = LSTM(50, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

bbox_branch = Dense(511, activation='relu')(x)
bbox_branch = Dense(128, activation='relu')(bbox_branch)
bbox_branch = Dense(64, activation='relu')(bbox_branch)
box_branch = Dense(32, activation='relu')(bbox_branch)
bbox_output = Dense(4, activation='linear', name='bbox')(bbox_branch)

class_branch = Dense(511, activation='relu')(x)
class_branch = Dense(128, activation='relu')(class_branch)
class_branch = Dense(64, activation='relu')(class_branch)
class_branch = Dense(32, activation='relu')(class_branch)
class_output = Dense(NUM_CLASSES, activation='softmax', name='class')(class_branch)

#%%
model = Model(inputs=inputs, outputs=[bbox_output, class_output])
model.summary() 

#%%

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001e-2),
              loss={'bbox': 'mse', 'class': 'categorical_crossentropy'},
              metrics={'bbox': 'mae', 'class': 'accuracy'})

model.fit(X, Y, epochs=100, batch_size=32, callbacks=[tf.keras.callbacks.ReduceLROnPlateau()])
#%% 
y_pred = model.predict(X)
    
y_pred_class = np.argmax(y_pred[1] , axis=1)

y_true_class = np.argmax( Y['class'] , axis=1) 

from sklearn.metrics import accuracy_score 

print(accuracy_score(y_true_class , y_pred_class))
   

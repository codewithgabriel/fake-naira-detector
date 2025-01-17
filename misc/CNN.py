# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 06:29:11 2025

@author: HP
"""

import json
import os

image_dir = 'project-5/images'
annotation_result_file = 'project-5/result.json'


with open(annotation_result_file , 'r') as f: 
    data = json.load(f) 

annotations = data['annotations']
categories = {cat['id']:  cat['name'] for cat in data['categories']}
images = {img['id']: os.path.join("project-5" , img['file_name'] ) for img in data['images'] }

import cv2
import matplotlib.pyplot as plt
import numpy as np

i = 1

img_size = (224,224) 
X, Y = [] , []

for annotation in annotations: 
    image_file = images[annotation['image_id']]
    image_bbox =  annotation['bbox']
    category_id = annotation['category_id']
    img = cv2.imread(image_file)
    orginal_size = img.shape 
    resize_img = cv2.resize(img , img_size) / 255.0

    
    print("resized shape:" , resize_img.shape)
  
    x_scaled = img_size[1] / orginal_size[0]
    y_scaled = img_size[0]/ orginal_size[1]
 
    print(x_scaled , y_scaled)
    
    
    x , y , w , h = image_bbox
    print(x , y , w, h)
    x, y ,w , h = int(x*x_scaled) , int(y * y_scaled) , int(w*x_scaled) , int(h*y_scaled)
    
    print(x , y , w, h)
    cv2.rectangle(resize_img , (x ,y) ,  (x + w , y + h) , (0,255,0) , 2)
    
    plt.imshow(resize_img )
    plt.axis('off')
    #plt.show()
    
    
    
    X.append (resize_img)
    Y.append([*image_bbox , category_id])
    
    #print(image_file , image_bbox , category_id)
    i+=1
    

  
    
    
   
X = np.array(X)
Y = np.array(Y)
num_classes = len(categories)
# y to categorcial
from keras.api.utils import to_categorical 

y_categorical =  to_categorical(Y[: , 4].astype(int) , num_classes )
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
sc.fit(y_categorical)
y_categorical = sc.transform(y_categorical)

Y_train = np.concatenate( [Y[: , :4] , y_categorical] , axis=1)


# create simple cnn 
from keras.api.layers import Conv2D , MaxPool2D, Dense, Flatten , Input 
from keras.api.models import Sequential

model = Sequential() 
model.add(Input((224,224,3)))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(strides=(2,2)))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(4 + num_classes ,  activation='linear'))

model.summary()
model.compile(optimizer='adam' ,  loss='mse', metrics=['accuracy'])
model.fit(X, Y_train , epochs=50)



x1  = X[0]
y_pred = model.predict(X)

classes_pred = y_pred[: , 4:] 

classes_pred = np.argmax( sc.inverse_transform(y_pred[: , 4:])  , axis=1)
classes_true = np.argmax(Y_train[: , 4:] , axis=1)


from sklearn.metrics import *
accuracy_score(classes_true, classes_pred)


# test  
test_img = cv2.imread('project-5/images/7fbe4b0a-10-back-3.jpg') 
test_img = cv2.resize(test_img , img_size)  / 255.0 

test_pred = model.predict( np.array([test_img] ) )

bbox = test_pred[0][:4]
test_classes = sc.inverse_transform(test_pred[0][4:].reshape(1,-1)) 

test_classes = np.argmax(test_classes , axis=0)

y_scaled = img_size[1] / test_img.shape[0]
x_scaled = img_size[0] / test_img.shape[1]


x, y , w , h = bbox 

x, y , w , h  = int(x_scaled *x) , int(y_scaled *y) , int(x_scaled *w) , int(y_scaled *h)

cv2.rectangle(test_img , (x,y) , (x+w , y+h) , (0,255,0) , 3)
plt.imshow(test_img)












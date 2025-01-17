# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:48:24 2025
@author: HP
"""
#%%
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
import random
#%%
# Set global constants
images_dir = 'project-6/images'
annotation_path = 'project-6/result.json'
img_size = (244, 244)  # Image size for resizing
X, Y = [], []

# Import dataset
with open(annotation_path, 'r') as annotation_data:
    dataset = json.load(annotation_data)

images = {img['id']: os.path.join('project-6', img['file_name']) for img in dataset['images']}
annotations = dataset['annotations']
categories = {cat['id']: cat['name'] for cat in dataset['categories']}
#%%
# Functions for preprocessing and feature extraction
def read_img(file_path):
    """
    Reads an image from the given file path.
    """
    return cv2.imread(file_path, 1)


def resize_img(img):
    """
    Resizes the image to the target size.
    """
    resized_img = cv2.resize(img, img_size)
    return resized_img / 255.0  # Normalize to [0, 1]


def extract_features(img):
    """
    Extracts features using HOG and Edge Detection (Canny).
    Combines these with the original resized image.
    """
    resized_img = cv2.resize(img, img_size) / 255.0  # Resized image
    
    # HOG Features
    hog_features, _ = hog(
        resized_img, 
        orientations=8, 
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2), 
        visualize=True, 
        channel_axis=-1
    )
    
    # Edge Detection
    edge_map = cv2.Canny((resized_img * 255).astype(np.uint8), threshold1=100, threshold2=200)
    edge_map = edge_map / 255.0  # Normalize edge map to [0, 1]
    edge_map = edge_map.reshape(img_size[0], img_size[1], 1)

    # Concatenate HOG features, edge map, and original resized image
    combined_features = np.concatenate([resized_img, edge_map], axis=-1)
    
    return combined_features.astype(np.float32)


def scale_boundry(img, image_bbox):
    """
    Scales the bounding box coordinates to match the resized image.
    """
    original_size = img.shape
    x, y, w, h = image_bbox
    x_scaled = 244 / original_size[1]
    y_scaled = 244 / original_size[0]
    x, y, w, h = int(x * x_scaled), int(y * y_scaled), int(w * x_scaled), int(h * y_scaled)
    return x, y, w, h


def augment_image(img, bbox):
    """
    Apply random augmentations to the image and bounding box.
    """
    # Flip Image Horizontally
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        x, y, w, h = bbox
        bbox = [img.shape[1] - x - w, y, w, h]

    # Add Random Brightness
    if random.random() > 0.5:
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=random.randint(-30, 30))

    # Add Gaussian Noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
    
    return img, bbox


def append_img(img, image_bbox):
    """
    Appends the image and bounding box data to the dataset.
    """
    X.append(img)
    Y.append([*image_bbox, category_id])

#%%
# Processing the dataset
i = 1
for annotation in annotations:
    image_file = images[annotation['image_id']]
    image_bbox = annotation['bbox']
    category_id = annotation['category_id']

    # Read the image
    img = read_img(image_file)
    
    # Apply augmentation
    img, image_bbox = augment_image(img, image_bbox)
    
    # Scale bounding box
    x, y, w, h = scale_boundry(img, image_bbox)
    
    # Extract features
    img = extract_features(img)
    
    # Append the image and bounding box data
    append_img(img, image_bbox)

    # Show image (optional for debugging)
    plt.imshow(img[:, :, :3])  # Display RGB part of the combined features
    plt.axis('off')
    plt.show()

    i += 1
    if i == 10:  # Debug: Process only the first 3 images for testing
        break

#%%
# Convert data to arrays
X = np.array(X)
Y = np.array(Y)
num_classes = len(categories)
#%%
# Handle categorical labels
from keras.utils import to_categorical
y_categorical = to_categorical(Y[:, 4], num_classes)
Y = np.concatenate([Y[:, :4], y_categorical], axis=1)

y_bbox = Y[:, :4]
y_class = Y[:, 4:]
#%%
# Standardize bounding box coordinates
from sklearn.preprocessing import StandardScaler
y_sc = StandardScaler()
y_sc.fit(y_bbox)
y_bbox = y_sc.transform(y_bbox)

#%%
Y = {
    'bounding_box': y_bbox,
    'class': y_class
}

# Train-test split
X_train, X_test = X[10:], X[:10]
y_train = {
    'bounding_box': Y['bounding_box'][10:],
    'class': Y['class'][10:]
}
y_test = {
    'bounding_box': Y['bounding_box'][:10],
    'class': Y['class'][:10]
}
#%%
# Building the model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from keras.models import Model

input_shape = (244, 244, X.shape[3])

inputs = Input(shape=input_shape)
x = Conv2D(16, kernel_size=(3, 3), activation='relu')(inputs)
x = MaxPool2D(strides=(2, 2))(x)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPool2D(strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)

# Bounding box output
bbox_output = Dense(32, activation='relu')(x)
bbox_output = Dense(4, activation='linear', name='bounding_box')(bbox_output)

# Class output
class_output = Dense(num_classes, activation='softmax', name='class')(x)

#%%
model = Model(inputs=inputs, outputs=[bbox_output, class_output])
model.compile(optimizer='adam', 
              loss={'bounding_box': 'mse', 'class': 'categorical_crossentropy'}, 
              metrics={'bounding_box': 'mae', 'class': ['accuracy']})

# Training the model
history = model.fit(X_train, y_train, epochs=50)

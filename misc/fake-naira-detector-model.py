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
#%%
root_dir = 'project-9'
images_dir = f'{root_dir}/images'
annotation_path = f'{root_dir}/result.json'

# import dataset 
with open(annotation_path , 'r') as annotation_data:
    dataset  = json.load(annotation_data)
    
images =  { img['id']: os.path.join(root_dir , img['file_name']) for img in dataset['images'] }
annotations = dataset['annotations']
categories = { cat['id'] : cat['name'] for cat in dataset['categories'] }




#%%

X ,  Y = [] , []
img_size = (244, 244)
i = 0
#%%
def read_img(file_path): 
    img = cv2.imread(image_file, 1)
    #resize the image
    return img

def resize_img(img):
    resized_img = cv2.resize(img , img_size)
    return resized_img / 255.0
    

def extract_features(img):
    # feature extraction using HOG
    #feature, hog_image = hog(img, orientations=8, pixels_per_cell=(8,8), 
    #                       cells_per_block=(2, 2), visualize=True, channel_axis=None)
  
    # normalize the image 
    #hog_image = hog_image / 255.0 
    
    edge = cv2.Canny(img, threshold1=100, threshold2=100)
    return np.array(edge).reshape(244,244,1).astype(np.float32)
    
def scale_boundry(img , image_bbox):
    orginal_size = img.shape 
    
    x , y , w , h = image_bbox
  
    
    #  scale bounding box
    x_scaled = 244 / orginal_size[1]
    y_scaled = 244  / orginal_size[0]
 
   
    x, y ,w , h = int(x*x_scaled) , int(y * y_scaled) , int(w*x_scaled) , int(h*y_scaled)
    
    return ( x ,y , w, h)

def draw_boundry_box(img, x ,y , w, h):
    cv2.rectangle(img , (x ,y) ,  (x + w , y + h) , (0,255,0) , 2)
    
def show_img(img):
    #showing the image
    plt.imshow(img)
    plt.title(" Image ")
    plt.axis('off')
    plt.show()
    
def append_img(img , image_bbox):
    print(img)
    X.append (img)
    Y.append([*image_bbox , category_id])
       
    
#%%
i = 1
for annotation in annotations: 
    image_file = images[annotation['image_id']]
    image_bbox =  annotation['bbox']
    category_id = annotation['category_id']
    
    
    # read the image
    img = read_img(image_file)
    # scale boundry box 
    x ,y , w, h = scale_boundry(img , image_bbox)
    
    #image_bbox =  [x ,y , w, h]
    
    # apply feature extractions here
    
    
    #resize img
    img = resize_img(img)
    
    # draw boundry box 
    #draw_boundry_box(img, x, y, w, h)
    
    #img = extract_features(img)
    
    # append image
    append_img(img , image_bbox)
    #show image
    #show_img(img[: , : , ::-1])
    
    i+=1
    if i == 4:
        pass
    
    
    
    
    
#%%
    
X = np.array(X) 
Y = np.array(Y)
num_classes = len(categories)
#%%

# handling categorical 
from keras.api.utils import to_categorical 
y_categorical = to_categorical(Y[: , 4] , num_classes)
Y = np.concatenate([Y[: , :4].astype(int) , y_categorical], axis=1)


#%%
y_bbox = Y[: , :4]
y_class = Y[: , 4:]

# standard scale bbox
from sklearn.preprocessing import StandardScaler
y_sc = StandardScaler()
y_sc.fit(y_bbox)
y_bbox = y_sc.transform(y_bbox) 

#%%

Y = { 
    'bounding_box': y_bbox,
    'class': y_class
    }


X_train = X[10:]
X_test = X[:10]

y_train =  { 
    'bounding_box': Y['bounding_box'][10:],
    'class': Y['class'][10:]
    }

y_test =  { 
    'bounding_box': Y['bounding_box'][:10],
    'class': Y['class'][:10]
    }

#%%
from keras.api.layers import Conv2D, MaxPool2D, AveragePooling2D , Dense , Input , Flatten
from keras.api.metrics import MeanSquaredError, Accuracy, MeanAbsoluteError
from keras.api.models import Sequential
#%%

# building the model
input_shape  = (244,244, 3)

inputs = Input(shape=input_shape)


x = Conv2D(16,kernel_size=(3,3), activation='relu')(inputs)
x = MaxPool2D(strides=(2,2))(x)

x = Conv2D(32,kernel_size=(3,3), activation='relu')(x)
x = MaxPool2D(strides=(2,2))(x)


x = Flatten()(x)
x = Dense(511, activation='relu')(x)
x = Dense(128 , activation='relu') (x)


# for predicting bounding box 
bbox_output = Dense(32, activation='relu')(x)
bbox_output = Dense(4, activation='linear', name='bounding_box')(bbox_output)

# for class 
class_output = Dense(num_classes , activation='softmax' , name='class')(x)

#%%
from keras.api.models import  Model
# the model 
model = Model(inputs=inputs , outputs=[bbox_output, class_output])

# compiling the model
model.compile(
    optimizer='adam' , 
    loss= {
        'bounding_box': 'mse'  ,
        'class': 'categorical_crossentropy'        
        }, 
   
    metrics= { 
        'bounding_box': 'mae',
        'class': ['accuracy', 'precision']
        })

model.summary() 

#%%
# training the model
history = model.fit(X_train, y_train , epochs=34)


#plot the learning curve
plt.plot(history.history['loss']  )
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['class_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


#%%
y_pred = model.predict(X_test)
#%%


# for bounding box 
y_bbox_pred = y_sc.inverse_transform( y_pred[0] )
y_bbox_true = y_sc.inverse_transform( y_test['bounding_box'])

# for class
y_class_pred = np.argmax(y_pred[1] , axis=1)
y_class_true = np.argmax(y_test['class'] , axis=1)


y_class_pred_label = [categories[pred] for pred in y_class_pred]
y_class_true_label = [categories[pred] for pred in y_class_true]


from sklearn.metrics import accuracy_score , mean_absolute_error

print("accuracy: " , accuracy_score(y_class_true , y_class_pred))
print("MAE: " , mean_absolute_error(y_bbox_true , y_bbox_pred))


#%%
#predict single image

pred_image_file = 'project-8/images/3d95d6bf-200-back-4.jpg'

test_img = cv2.imread(pred_image_file)

resized_test_img = resize_img(test_img)

test_img_pred = model.predict(np.array([resized_test_img]))

test_img_pred_class = categories[np.argmax( test_img_pred[1])]

test_img_pred_bbox = y_sc.inverse_transform( test_img_pred[0][0].reshape(1,-1) )[0]

x ,y , w, h = scale_boundry(test_img ,test_img_pred_bbox )



draw_boundry_box(resized_test_img, x, y, w, h)
show_img(resized_test_img[: ,: , ::-1])

print(test_img_pred_class  , test_img_pred_bbox)
#%%

# save the model
model.save('1.h5')
#model.save_weights('w.weights.h5')


#%%


# testing the model 
from keras.api.models import load_model
model = load_model('1.h5',
        custom_objects={
        'MeanSquaredError': MeanSquaredError,
        'MeanAbsoluteError': MeanAbsoluteError,
        'Accuracy': Accuracy
    }
  )
y_pred = model.predict(X_test)

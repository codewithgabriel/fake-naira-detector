import numpy as np 

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



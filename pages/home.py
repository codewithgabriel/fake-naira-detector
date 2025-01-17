import streamlit as st 
from io import StringIO 
from keras.api.models import load_model
import numpy as np

model = load_model("model/fake-naira-notes-detector.h5")
import pandas as pd

st.title("Counterfeit Naira Note Detector")

# metrics row 

history_col , metrics_col = st.columns(2)


# show training report 
from history import history

chart_data = pd.DataFrame({ "accuracy": history['accuracy'] , "precision": history["precision"] })
history_col.subheader("Loss Vs Epoch")
history_col.line_chart(history['loss']  , x_label="Epoch" , y_label="Loss")

metrics_col.subheader("Accuracy Vs Epoch")
metrics_col.line_chart(chart_data , x_label="Epoch" , y_label="Progress per epoch" , color=['#ff0000' , '#0000ff'])
# show classification report
classification_report = pd.read_csv('result/result.csv')
total_report = pd.read_csv("result/result2.csv")



st.subheader("Performance Metrics Report")
st.table(classification_report)
st.table(total_report)

main_cols = st.columns(2)


couterfeit_notes  = [
    "dataset/train/counterfeit/100 naira/100-back-1-Color-Restored.jpg",
    "dataset/train/counterfeit/200 naira/200-back-4.jpg",
    "dataset/train/counterfeit/500 naira/500-back-1.jpg",
    "dataset/train/counterfeit/1000 naira/1000-back-10.jpg",
]

genuine_notes = [
    "dataset/train/genuine/100 naira/100-back-2.jpg",
    "dataset/train/genuine/200 naira/200-back-4.jpg",
    "dataset/train/genuine/500 naira/500-back-4.jpg",
    "dataset/train/genuine/1000 naira/1000-back-3.jpg"
]

st.subheader("Counterfeit Notes Sample")
counterfeit_row = st.columns(4 , gap='small') 
st.subheader("Genuine Notes Sample")
genuine_row = st.columns(4 , gap='small')



from keras.api.utils import load_img , img_to_array

@st.dialog("Classification Result")
def open_popup(class_pred, y_pred):
    st.progress(float(y_pred[0]) , text='Counterfeit')
    st.progress(float(y_pred[1]) , text='Genuine')
    
    st.text(f"predicted class: {class_pred}")
    
    
def predict_img(img_path):
    class_names = ["counterfeit" , "genuine"]
    img = load_img(img_path , target_size=(64,64))
    img = img_to_array(img)
    y_pred = model.predict( np.array([img]) )
  
    
    class_pred = class_names[np.argmax(y_pred)]
    y_pred = y_pred[0]
    open_popup(class_pred , y_pred)
    

def on_click(img_path):
    # Define what happens when the image is clicked
    predict_img(img_path)
   
    

for col, img_path in zip(counterfeit_row, couterfeit_notes):
    col.image(img_path)
    if col.button('test run', key=img_path):
        on_click(img_path)




    
for col, img_path in zip(genuine_row , genuine_notes): 
    col.image(img_path , width=150)
    if col.button('test run', key=img_path):
        on_click(img_path)
    
    
    
# main form row 
form_row = st.columns(1)
file_upload = form_row[0].file_uploader("Choose a Test Sample" , type=["png" , 'jpg'])

if form_row[0].button("Execute Detector"):
    if file_upload is not None: 
        predict_img(file_upload)
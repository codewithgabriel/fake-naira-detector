import streamlit as st 

with open('trainer.py' , 'r') as code:
    st.code(code.read(3024) , language='python')
    code.close()
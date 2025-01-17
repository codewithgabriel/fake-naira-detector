import streamlit as st

pages = [
        st.Page("pages/home.py", title="Counterfeit Naira Note Detection"),
       st.Page("pages/source_code.py", title="Project Source Code"),
]

pg = st.navigation(pages)
pg.run()
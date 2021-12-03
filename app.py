import streamlit as st
st.set_page_config(layout="wide")
from predict_page import show_predict_page
from explore_page import show_explore_page
page = st.sidebar.selectbox('Explore or Predict', ("Predict", "Explore"))
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()

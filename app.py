import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("Final Project Data Science")
st.write("**Bootcamp Data Science dan Data Analyst Offline Batch 1** - Dhia Irsyad")

st.title("Student Performance")
# TRACK ACTIVE TAB
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "About Dataset"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'About Dataset',
    'Dashboards',
    'Machine Learning Regresi',
    'Machine Learning Klasifikasi',
    'Prediction App',
    'Contact Me'
])
with tab1:
    st.session_state.active_tab = "About Dataset"
    import about
    about.about_dataset()

with tab2:
    st.session_state.active_tab = "Dashboards"
    import visualisasi
    visualisasi.chart()

with tab3:
    st.session_state.active_tab = "Machine Learning Regresi"
    import machine_learning
    machine_learning.ml_model()

with tab4:
    st.session_state.active_tab = "Machine Learning Klasifikasi"
    import machine_learning2
    machine_learning2.ml_model2()

with tab5:
    st.session_state.active_tab = "Prediction App"
    import prediction
    prediction.prediction_app()

with tab6:
    st.session_state.active_tab = "Contact Me"
    import contact
    contact.contact_me()

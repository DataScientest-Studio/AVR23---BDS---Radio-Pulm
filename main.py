import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

title = "Analyse de radiographies pulmonaires Covid-19"
sidebar_name = "Home"


def run():
    st.title(title)
    
    st.header("Projet")
    st.header("Objectifs")
    st.markdown(
        """
        Prédire si le patient est atteint de la COVID-19 ou non à partir d'une radiographie thoracique à l'aide via approche d'apprentissage supervisé
        """
        )
    
    
    st.header("Démarche")
    st.markdown(""" """)
    image = Image.open('/Users/hind/Desktop/illustrations/methodo.png')
    st.image(image, use_column_width=True)

    
    st.header("Sources")

    source = st.radio(" ",["KAGGLE", "GitHub", "Autres"])

    if source == "KAGGLE":
        st.write("KGL_RSNA_Pneumonia : https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview/timeline")
        st.write("KGL_Chest_Xray_Pneumonia : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")

    
    elif source == "GitHub":
        st.write("GitHub_covid_CXNet : https://github.com/armiro/COVID-CXNet")
        st.write("GitHub_covid_rep : https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png")
        st.write("GitHub_covid_chestray_ds : https://github.com/ieee8023/covid-chestxray-dataset")
    

    elif source == "Autres":
        st.write("Bimcv_covid19 : https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711")
        st.write("Eurorad : https://eurorad.org")
        st.write("SIRM_Covid : https://sirm.org/category/senza-categoria/covid-19/")



import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


title = "Analyse de radiographies pulmonaires Covid-19"
sidebar_name = "Accueil"

# Emplacement à changer en local
path = "C:/Users/Nina/Documents/GitHub/AVR23---BDS---Radio-Pulm/"

def run():
    st.title(title)
    st.divider()

    st.header("Objectif")
    st.markdown(
        """
        Prédire si le patient est atteint de la COVID-19 ou non à partir d'une radiographie thoracique via une 
        approche d'apprentissage supervisé.
        """
        )
    
    st.header("Démarche")
    st.markdown("""\n\n""")
    image = Image.open(path + 'streamlit_app/assets/methodo.png')
    st.image(image, use_column_width=True)

    
    st.header("Source")
    st.markdown("""\n\n""")
    st.markdown(
        """
        Le set de donnée contient des images radiographiques pulmonaires pour des cas positifs à la COVID-19 mais aussi 
        des images radiographiques de pneumonies normales et virales. 
        \n Lien vers le dataset : https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
        \n(Taille des données : 1.15 Gb)
        """
        )
    



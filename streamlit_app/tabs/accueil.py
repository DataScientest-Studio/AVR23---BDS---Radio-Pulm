import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


title = "Analyse de radiographies pulmonaires Covid-19"
sidebar_name = "Accueil"


def run():
    st.title(title)
    st.header("Objectifs")
    st.markdown(
        """
        Prédire si le patient est atteint de la COVID-19 ou non à partir d'une radiographie thoracique via une 
        approche d'apprentissage supervisé. [voir Contexte medical](#contexte_medical)
        """
        )
    
    st.header("Démarche")
    st.markdown("""\n\n""")
    image = Image.open('/Users/hind/Desktop/illustrations/methodo.png')
    st.image(image, use_column_width=True)

    
    st.header("Sources")
    st.markdown("""\n\n""")
    st.markdown(
        """
        Le set de donnée contient des images radiographiques pulmonaires pour des cas positifs à la COVID-19 mais aussi 
        des images radiographiques de pneumonies normales et virales. 
        \n Lien vers le dataset : https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
        \n (Taille des données : 1.15 Gb)
        """
        )
    


run()
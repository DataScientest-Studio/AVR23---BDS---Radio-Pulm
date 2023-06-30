import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


title = "Conclusion"
sidebar_name = "Conclusion"

path = "D://documents/GitHub/AVR23---BDS---Radio-Pulm/streamlit_app/assets/"

def run():
      
    displayFin = False
    
    st.markdown(
        """
        #
        """)
    if st.button("C'est la fin !", type = "primary") :
                displayFin = True
        
    if displayFin is False :
    
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:

            st.title(title)
            st.divider()

            tab1, tab2,tab3 = st.tabs(["Performances des Modèles", "Dificultés Rencontrées", "Ouvertures"])

            with tab1 : #Performances des Modèles

                st.markdown(
                    """
                    #### Analyse :
                    - Globalement les performances des modèles sont meilleurs pour les __images brutes__ (sauf pour LeNet).
                    - __EfficientNetB1 & ResNet101__ ont les meilleurs performances.
                    - Resnet152 présente une performance __non comparable__ aux autres modèles __(?)__""")

                colA, colB = st.columns([6,1])
                with colA :
                    image1 = Image.open(path + 'performance_modele1.png')
                    st.image(image1, width = 400)
                  
                with colB :
                    image2 = Image.open(path + 'performance_modele2.png')
                    st.image(image2, width = 400)

            with tab2 :
                st.markdown(
                    """
                    Dans notre démarche de modélisation, plusieurs défis ont ralenti notre progression :
                    - __L'étude anticipée des modules de Deep Learning__
                    - __La prise en main des outils__
                    - __Les notions non abordées dans DataScientest__ 
                        - _KerasTuner, ResNet & GradCAM_
                    - __L'accès au GPU__
                    """)

            with tab3 :
                st.markdown(
                    """
                    __Pour aller plus loin ...__
                    - Intégration du modèle binaire (_COVID vs Non-COVID_) qui a eu des meilleurs résultats.
                    - Analyse des différentes images du Jeu de Données pour bien comprendre les possibles biais (source, position, genre/âge du patient, écritures sur les radios).
                    - Compréhension des meilleurs performances avec les images brutes (_non attendu_).
                    """)    

        with col2: 
            st.markdown(
                    """
                    #
                    """)
            st.image(Image.open(path + 'x-ray.png'), use_column_width=True)
               
    
    if displayFin is True :
        st.title("Merci !")
        col1,col2,col3 = st.columns([0.1,0.8,0.1])
        with col2 :
            st.image(Image.open(path + 'skeleton.png'),use_column_width=True)
            st.markdown(
                """
                #
                """)

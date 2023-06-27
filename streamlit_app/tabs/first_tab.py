import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

title = "Introduction"
sidebar_name = "Introduction"


def run():
    st.title(title)

    tab1, tab2, tab3 = st.columns(3)

    with tab1:
        base_radio_button = st.button("COVID-19")

    with tab2:
        detection_button = st.button("Détection")

    with tab3:
        radiology_button = st.button("Radiologie")

    if base_radio_button:
        base_radio()
    elif detection_button:
        detection()
    elif radiology_button:
        radiology()

def base_radio():
    st.subheader("COVID-19")
    
    columns = st.columns([4,0.5])
    
    with columns[0]:
       st.markdown("""
        Syndrome respiratoire aigu ayant causé près de 7 millions de décès et infecté plus de 765 millions de personnes au 9 mai 2023.
        Principaux symptômes: 
        \n - fièvre
        \n - toux
        \n - dyspnée 
        \n - présence d’infiltrats bilatéraux à l’imagerie
        """
       )

       image0 = Image.open('/Users/hind/Desktop/illustrations/radio_scan_covid.png')
       st.image(image0, use_column_width=True, width=150)
       

       st.markdown("""
       \n\n\n Cas graves = syndrome de détresse respiratoire aiguë (SDRA) ou une insuffisance respiratoire complète, nécessitant une assistance ventilatoire mécanique et des soins intensifs en unité de soins intensifs. 
       \nLes individus immunodéprimés et âgés sont les plus susceptibles de développer des formes graves de la maladie : 
       \n- insuffisance cardiaque 
       \n- insuffisance rénale 
       \n- choc septique
        """)
        
    with columns[1]: 
        image1 = Image.open('/Users/hind/Desktop/illustrations/fievre.png')
        st.image(image1, use_column_width=True)
        image2 = Image.open('/Users/hind/Desktop/illustrations/toux.png')
        st.image(image2, use_column_width=True)
        image3 = Image.open('/Users/hind/Desktop/illustrations/dyspnee.png')
        st.image(image3, use_column_width=True)
       

def detection():
    st.subheader("Outils de détection")
    
    st.markdown("""
    La détection fiable de la COVID-19 est donc un enjeu crucial mais : 
    \n - diagnostic pas toujours évident car les 
    symptômes courants généralement indiscernables d’autres infections virales telles que la pneumonie 
    \n\n Outil de diagnostic principal = RT-PCR. 
    """)
    
    
    image = Image.open('/Users/hind/Desktop/illustrations/pcr.png')
    st.image(image, use_column_width=True, width=100)


    st.markdown("""
    \n Limites : 
    \n - faux négatifs en cas de prélèvement inapproprié
    \n - charge virale basse
    \n - mutations génétiques 
    \n - nécessité de laboratoires spécialisés
    \n - délais de traitement des échantillons longs 
    
    \n\n\n Imagerie thoracique préféré : sensibilité + élevée par rapport à la RT-PCR
    \n - scanners
    \n - radiographies -> plus simples, rapides et moins coûteuses /scan. 
    
    """ 
    )
    

def radiology():
    st.subheader("Notions de radiologie et signes")
    
    st.markdown("""
    \n - opacités bilatérales
    \n - multifocales (en plusieurs endroits)
    \n - distribution postérieure ou périphérique, principalement dans les lobes pulmonaires inférieurs
    (similitudes avec ceux d’une pneumonie virale et d’autres infections pulmonaires)
    \n
    \n - stade tardif => fibrose pulmonaire
    
    """)


    col = st.columns([2,4])

    with col[0]:
        selected = st.radio(
            "Select sample",
            options=["Normal", "COVID", "Pneumonie", "Autre PP"]
        )

    with col[1]:
        if selected == 'Normal':
            st.image('/Users/hind/Desktop/illustrations/radio_normal.png', width=500)

        elif selected == 'COVID':
            st.image('/Users/hind/Desktop/illustrations/radio_covid.png', width=500)
            
        elif selected == 'Pneumonie':
            st.image('/Users/hind/Desktop/illustrations/radio_pneumonie.png', width=500)

        elif selected == 'Autre PP':
            st.image('/Users/hind/Desktop/illustrations/radio_lung_opacity.png', width=500)

radiology()






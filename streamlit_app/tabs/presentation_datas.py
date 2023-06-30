import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from streamlit import cache


title = "Présentation des Données"
sidebar_name = "Présentation des Données"

# Emplacement
path = "D://documents/GitHub/AVR23---BDS---Radio-Pulm/"

# Constantes et dictionnaires
colors = ["red", "yellow", "green", "orange"]
category_order = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']


def run():
    
    st.markdown(
        """
        #
        """)
    df = load_data1(path + "data/df.csv")
    
    st.title(title)
    st.markdown("---")
       
    tab_main1, tab_main2 = st.tabs(["Dataset & Volumétrie", "Images Brutes / Masquées"])
    #options = ('Dataset & Volumétrie', 'Images Brutes / Masquées')
    #option = st.radio('', options)

    #st.divider()
    
    with tab_main1:
        tab1, tab2, tab3= st.tabs(["Par Catégorie", "Par Source", "Par Source & Catégorie"])
    
        with tab1 :
            st.markdown("#### Nombre d'images par catégorie")
            fig = go.Figure()
            for category, color in zip(category_order, colors):
                count = df['label'].value_counts()[category]
                fig.add_trace(go.Bar(x=[category], y=[count], name=category, marker=dict(color=color)))
            fig.update_layout(
                xaxis=dict(title="Catégorie"),
                yaxis=dict(title="Nombre d'images"),
                hovermode='x')
            st.plotly_chart(fig)
            
            with st.expander("**Définition des catégories**"):
                st.markdown(
                    """
                    Les 4 catégories d’images (_“label”_) disponibles sont :
                    - **COVID** : Individu atteint de la COVID-19.
                    - **Lung_Opacity** : Individu présentant des opacités sur sa radiographie, symptôme d’une pathologie pulmonaire (non explicitement indiquée).
                    - **Normal** : Individu sain, sans problème aux poumons.
                    - **Viral_Pneumonia** : Individu atteint de pneumonie virale.
                    #
                    _Remarque :_ Chaque image aura un masque associé.
                    """)
            
            st.markdown("""
                #
                #
                #
                """)

        with tab2 :
            st.markdown("#### Nombre d'images par source")
            
            source_order = ["KGL_RSNA_Pneumonia", "KGL_Chest_Xray_Pneumonia", "SIRM_Covid", "GitHub_covid_repo", "Eurorad", "GitHub_covid_CXNet", "GitHub_covid_chestray_ds", "Bimcv_covid19"]
            count = df['source'].value_counts()[source_order]
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=count, y=source_order, orientation='h'))
            fig1.update_layout(xaxis=dict(title="Nombre d'images"), yaxis=dict(title="Source"), hovermode='x')
            st.plotly_chart(fig1, use_container_width=True)

        with tab3 :
            st.markdown("#### Nombre d'images par source et par catégorie")

            group_data = df[df['label'].isin(['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'])] # Filtrer données par groupe
            count = group_data.groupby(['source', 'label']).size().reset_index(name='count') # Comptage par source + groupe
            fig2 = go.Figure()
            fig2 = px.bar(count, x='count', y='source', color='label', orientation='h', category_orders={'source': source_order})
            fig2.update_layout(xaxis=dict(title="Nombre d'images"), yaxis=dict(title="Source"), hovermode='x')
            st.plotly_chart(fig2, use_container_width=True)


    
    with tab_main2:           
        tabA, tabC= st.tabs(["Utilité des Masques", "Création Masque"])
        with tabA :       
            st.markdown(
                """
                #### Intérêt des masques pulmonaires
                Certaines études ont opté pour une **segmentation des poumons** comme étape initiale de leur système de détection. 
                Les auteurs de l'ensemble de notre dataset ont segmenté les images en **excluant** la partie des poumons située derrière le cœur et en suivant des repères anatomiques passant par les côtes, l'arc aortique, le péricarde et le diaphragme.
            
                Les **masques pulmonaires** informent les modèles qu'ils doivent **accorder plus d'attention aux régions pulmonaires** contenant les **manifestations cliniques** de la COVID-19. 
            
                Ces masques réduisent par conséquent l'**espace de recherche** des signes COVID-19, fournissant aux modèles des indications précises sur les zones spécifiques auxquelles il convient de prêter attention.
                """)
            
            st.markdown("#### Images Brutes / Masquées")

            # definition des paths de chaque catégorie d'images
            image_path_covid = path + "data/COVID/images/"
            mask_path_covid = path + "data/COVID/masks/"
            masked_image_path_covid = path + "data/COVID/masked_images/"

            image_path_lung_opacity = path + "data/Lung_Opacity/images/"
            mask_path_lung_opacity = path + "data/Lung_Opacity/masks/"
            masked_image_path_lung_opacity = path + "data/Lung_Opacity/masked_images/"

            image_path_normal = path + "data/Normal/images/"
            mask_path_normal = path + "data/Normal/masks/"
            masked_image_path_normal = path + "data/Normal/masked_images/"

            image_path_pneumonia = path + "data/Viral Pneumonia/images/"
            mask_path_pneumonia = path + "data/Viral Pneumonia/masks/"
            masked_image_path_pneumonia = path + "data/Viral Pneumonia/masked_images/"

            # chargement images
            images_covid = []
            images_lung_opacity = []
            images_normal = []
            images_pneumonia = []

            for i in range(100, 105):
                image_covid = Image.open(image_path_covid + "COVID-" + str(i) + ".png")
                mask_covid = Image.open(mask_path_covid + "COVID-" + str(i) + ".png")
                masked_image_covid = Image.open(masked_image_path_covid + "COVID-" + str(i) + ".png")
                images_covid.append([image_covid, mask_covid, masked_image_covid])

                image_lung_opacity = Image.open(image_path_lung_opacity + "Lung_Opacity-" + str(i) + ".png")
                mask_lung_opacity = Image.open(mask_path_lung_opacity + "Lung_Opacity-" + str(i) + ".png")
                masked_image_lung_opacity = Image.open(masked_image_path_lung_opacity + "Lung_Opacity-" + str(i) + ".png")
                images_lung_opacity.append([image_lung_opacity, mask_lung_opacity, masked_image_lung_opacity])

                image_normal = Image.open(image_path_normal + "Normal-" + str(i) + ".png")
                mask_normal = Image.open(mask_path_normal + "Normal-" + str(i) + ".png")
                masked_image_normal = Image.open(masked_image_path_normal + "Normal-" + str(i) + ".png")
                images_normal.append([image_normal, mask_normal, masked_image_normal])

                image_pneumonia = Image.open(image_path_pneumonia + "Viral Pneumonia-" + str(i) + ".png")
                mask_pneumonia = Image.open(mask_path_pneumonia + "Viral Pneumonia-" + str(i) + ".png")
                masked_image_pneumonia = Image.open(masked_image_path_pneumonia + "Viral Pneumonia-" + str(i) + ".png")
                images_pneumonia.append([image_pneumonia, mask_pneumonia, masked_image_pneumonia])

            # image selection
            option = st.selectbox("__Sélectionnez une catégorie :__", ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"))

            # selection images au hasard
            if option == "COVID":
                random_pair = random.choice(images_covid)
            elif option == "Lung_Opacity":
                random_pair = random.choice(images_lung_opacity)
            elif option == "Normal":
                random_pair = random.choice(images_normal)
            elif option == "Viral Pneumonia":
                random_pair = random.choice(images_pneumonia)

            # affiche les paires d'images
            st.image([random_pair[0], random_pair[1], random_pair[2]], caption=['Image', 'Masque', 'Image masquée'], width=200)


            with st.expander("__Pour en savoir plus__"):
                    st.markdown(
                        """
                        Le U-Net est une architecture de réseau de neurones utilisée pour la **segmentation sémantique** (= segmenter l'image en différentes régions et attribuer une étiquette à chacune de ces régions) des poumons à partir d'images médicales. 
                        
                        Il se compose d'un **encodeur** et d'un **décodeur**, qui capturent respectivement les **informations contextuelles et spatiales**.
                        
                        Ainsi, le U-Net génère des **masques** précis en identifiant les **caractéristiques distinctives des poumons** dans les images.
                        """)
                    image = Image.open(path + "streamlit_app/assets/u-net-architecture.png")
                    st.image(image, use_column_width=True)
                    st.write("""source : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/""")

            st.markdown("""
                #
                #
                #
                """)       
                    
        with tabC : 
            st.markdown("#### Percentile")
            PATH = path+"data/percentile/"
            colA, colB, colC= st.columns((0.5,0.1,0.4))
            with colA:
                number  = st.slider("__Choisissez un pourcentage__ :", 20, 70,  step=5)
                st.markdown(
                    """
                    La fonction _SelectPercentile_ est une méthode utilisée pour **sélectionner les paramètres les plus importants** d'un jeu de données.
                    
                    Ceci est accompli **en calculant un score de chaque paramètre** et en sélectionnant un pourcentage défini par le score le plus élevé.
                    
                    Le pourcentage est spécifié par l'utilisateur lors de la création de la fonction.
                    Ce qui nous permet ici d'estimer un pourcentage des pixels les plus importants sur les radiographies.
                    """)

            with colC:
                st.write("   ")
                st.write("   ")
                st.write("   ")

                st.write("Les ", number," % pixels les plus importants")
                image = Image.open(PATH + str(number)+'.png')
                st.image(image)


### FONCTIONS APPELEES

@st.cache_data
#Fonction de chargement dataframe à partir csv
def load_data1(url):
    df = pd.read_csv(url)  # 👈 Download the data
    return df

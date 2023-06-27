import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


title = "Présentation données"
sidebar_name = "Présentation données"


def run():
    st.title(title)

    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        dataset_button = st.button("Dataset")
    with col2:
        volume_button = st.button("Volumétrie")
    with col3:
        images_button = st.button("Images brutes/masquées")

    if dataset_button:
        dataset()
    if volume_button:
        volume()
    if images_button:
        images()
   

def dataset():
    
    ### JDD 4 CATEGORIES
    def run():
        st.subheader("Jeu de données")

        # Emplacement
        path = '/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/'

        #Import des fichiers METADATA
        df_covid = pd.read_excel(path + "COVID.metadata.xlsx")
        df_lung_opacity = pd.read_excel(path + "Lung_Opacity.metadata.xlsx")
        df_normal = pd.read_excel(path + "Normal.metadata.xlsx")
        df_pneumonia = pd.read_excel(path + "Viral Pneumonia.metadata.xlsx")
        
        # Création d'une colonne avec le label associé de chaque catégorie
        df_covid['label'] = 'COVID'
        df_lung_opacity['label'] = 'Lung_Opacity'
        df_normal['label'] = 'Normal'
        df_pneumonia['label'] = 'Viral Pneumonia'

        df = pd.concat([df_covid, df_lung_opacity, df_normal, df_pneumonia], axis = 0)
        #st.dataframe(df.head())
        colors = ["red", "yellow", "green", "orange"]
        category_order = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

        fig = go.Figure()

        for category, color in zip(category_order, colors):
            count = df['label'].value_counts()[category]
            fig.add_trace(go.Bar(x=[category], y=[count], name=category, marker=dict(color=color)))

        fig.update_layout(
            title="Nombre d'images par catégorie",
            xaxis=dict(title="Catégorie"),
            yaxis=dict(title="Nombre d'images"),
            hovermode='x'
        )

        st.plotly_chart(fig)
    run()



    ### EXEMPLES RADIO POUR CHAQUE GROUPE
    def run():
        st.subheader("Exemples d'images pour chaque groupe")
        path_radios = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/"
        col1, col2 = st.columns(2)
        
        with col1 :
            
            options_categories = ("COVID", "Normal", "Lung_Opacity", "Viral Pneumonia")
            option_categories = st.radio("Sélectionner une catégorie", options_categories)
            
            if option_categories == options_categories[0]:
            
                options_image = ("COVID-1", "COVID-2", "COVID-3", "COVID-4")
                option_image = st.selectbox("Sélectionner une radio", options_image)
                path_choosen_image = path_radios + option_categories + "/images/" + option_image + ".png"
                                
            elif option_categories == options_categories[1]:
            
                options_image = ("Normal-1", "Normal-2", "Normal-3", "Normal-4")
                option_image = st.selectbox("Sélectionner une radio", options_image)
                path_choosen_image = path_radios + option_categories + "/images/" + option_image + ".png"
                
            elif option_categories == options_categories[2]:
            
                options_image = ("Lung_Opacity-1", "Lung_Opacity-2", "Lung_Opacity-3", "Lung_Opacity-4")
                option_image = st.selectbox("Sélectionner une radio", options_image)
                path_choosen_image = path_radios + option_categories + "/images/" + option_image + ".png"
                
            elif option_categories == options_categories[3]:
            
                options_image = ("Viral Pneumonia-1", "Viral Pneumonia-2", "Viral Pneumonia-3", "Viral Pneumonia-4")
                option_image = st.selectbox("Sélectionner une radio", options_image)
                path_choosen_image = path_radios + option_categories + "/images/" + option_image + ".png"
                                        
            choosen_image = Image.open(path_choosen_image)                                        
        
        #Affichage de l'Image Selectionnée
        with col2 :
            st.image(choosen_image)
    run()


def volume():
    
    st.subheader("Volumetrie")
    summary = """ 
    Nous nous sommes également intéressés à l’étude de la distribution des moyennes, des écart- types et des maximums de luminosité des images en fonction des labels (Normal, COVID, Lung_Opacity, Viral Pneumonia). 
    """
    st.markdown('<div style="text-align: justify;">' + summary, unsafe_allow_html=True)

    # MENU
    tab1, tab2, tab3, tab4 = st.tabs(["Graphique 1", "Graphique 2", "Graphique 3 ", "Graphique 4"])


    ####Données relatives DF
    def run():

        # Emplacement
        path = '/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/'
        df_covid = pd.read_excel(path + "COVID.metadata.xlsx")
        df_lung_opacity = pd.read_excel(path + "Lung_Opacity.metadata.xlsx")
        df_normal = pd.read_excel(path + "Normal.metadata.xlsx")
        df_pneumonia = pd.read_excel(path + "Viral Pneumonia.metadata.xlsx")
        df_covid['label'] = 'COVID'
        df_lung_opacity['label'] = 'Lung_Opacity'
        df_normal['label'] = 'Normal'
        df_pneumonia['label'] = 'Viral Pneumonia'

        df = pd.concat([df_covid, df_lung_opacity, df_normal, df_pneumonia], axis=0)

        colors = ["red", "yellow", "green", "orange"]
        category = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        urls = list(df["URL"].unique())
        source_order = ["KGL_RSNA_Pneumonia",
                        "KGL_Chest_Xray_Pneumonia",
                        "SIRM_Covid",
                        "GitHub_covid_repo",
                        "Eurorad",
                        "GitHub_covid_CXNet",
                        "GitHub_covid_chestray_ds",
                        "Bimcv_covid19"]

        df["source"] = df["URL"].replace(urls, source_order)
        df["path"] = path + df["label"] + "/" + "images" + "/" + df["FILE NAME"] + "." + df["FORMAT"].str.lower()
        df = df.reset_index()
        df["image"] = df["path"].map(lambda x: np.asarray(Image.open(x).convert("L").resize((75, 75))))

        # Mesure d'intensités
        im_mean = []
        im_std = []
        im_max = []
        im_min = []

        for i in range(0, len(df)):
            im_mean.append(np.mean(df["image"][i]))
            im_std.append(np.std(df["image"][i]))
            im_max.append(np.max(df["image"][i]))
            im_min.append(np.min(df["image"][i]))

        df["im_mean"] = im_mean
        df["im_std"] = im_std
        df["im_max"] = im_max
        df["im_min"] = im_min

       # GRAPHE 01 : Graphique du nombre d'images par source
        with tab1:
            fig1 = go.Figure()
            count = df['source'].value_counts()[source_order]
            fig1.add_trace(go.Bar(x=count,
                                  y=source_order,  
                                  orientation='h'))
            fig1.update_layout(
                title="Nombre d'images par source",
                xaxis=dict(title="Nombre d'images"),
                yaxis=dict(title="Source"),
                hovermode='x'
            )
            st.plotly_chart(fig1, use_container_width=True)


        # GRAPHE 02 : Graphique du nombre d'images par source et par jdd
        with tab2:
            # Filtrer données par groupe
            group_data = df[df['label'].isin(['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'])]

            # Comptage par source + groupe
            count = group_data.groupby(['source', 'label']).size().reset_index(name='count')
            
            # Graphe
            fig2 = px.bar(count, 
                          x='count',
                          y='source', 
                          color='label', 
                          orientation='h', 
                          category_orders={'source': source_order})
            fig2.update_layout(
                title="Nombre d'images par source et par catégorie",
                xaxis=dict(title="Nombre d'images"),
                yaxis=dict(title="Source"),
                hovermode='x'
            )
            st.plotly_chart(fig2, use_container_width=True)
            

        # GRAPHE 03 : Moyenne des luminosités
        with tab3:
            
            fig3 = plt.figure(figsize = (20,8))
            plt.subplot(121)
            sns.kdeplot(x = "im_mean", data = df, hue = "label", palette = colors)
            plt.title("Distribution des moyennes de luminosité en fonction du jdd")
            plt.subplot(122)
            sns.kdeplot(x = "im_mean", data = df, hue = "source")
            plt.title("Distribution des moyennes de luminosité en fonction des sources")

            # Afficher le graphique
            st.pyplot(fig3)


         # GRAPHE 04 : Écarts-types des luminosités
        with tab4:
            
            fig4 = plt.figure(figsize = (20,8))
            plt.subplot(121)
            sns.kdeplot(x = "im_std", data = df, hue = "label", palette = colors)
            plt.title("Distribution des écarts-types de luminosité en fonction du jdd")
            plt.subplot(122)
            sns.kdeplot(x = "im_std", data = df, hue = "source")
            plt.title("Distribution des écarts-types de luminosité en fonction des sources")

            # Afficher le graphique
            st.pyplot(fig4)
          

    run()

  
def images():

    st.subheader("Intérêt des masques pulmonaires")
    st.markdown(""" 
    Certaines études ont opté pour une segmentation des poumons comme étape initiale de leur système de détection. 
    \n -> segmentaion des images en excluant la partie des poumons située derrière le cœur 
    \n -> en suivant des repères anatomiques passant par les côtes, l'arc aortique, le péricarde et le diaphragme (masques pulmonaires ou « masks »)
    """)
    
    #col1, col2 = st.columns(2)
    
    image1 = Image.open('/Users/hind/Desktop/illustrations/gen_mask1.png')
    image2 = Image.open('/Users/hind/Desktop/illustrations/gen_mask2.png')
    st.image(image1, use_column_width=True)
    st.image(image2, use_column_width=True)


    st.markdown("""
    \n\n
    \n Réduction des espaces de recherche des signes COVID-19 : 
    \n -> informe les modèles qu'ils doivent accorder plus d'attention aux régions pulmonaires contenant les manifestations cliniques de la COVID-19
    """)
    
    
    


    def run():
        st.subheader("Images brutes / masquées")

        # Definition des paths de chaque catégorie d'images
        image_path_covid = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/COVID/images/"
        mask_path_covid = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/COVID/masks/"
        masked_image_path_covid = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/COVID/masked_images/"

        image_path_lung_opacity = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Lung_Opacity/images/"
        mask_path_lung_opacity = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Lung_Opacity/masks/"
        masked_image_path_lung_opacity = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Lung_Opacity/masked_images/"

        image_path_normal = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Normal/images/"
        mask_path_normal = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Normal/masks/"
        masked_image_path_normal = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Normal/masked_images/"

        image_path_pneumonia = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Viral Pneumonia/images/"
        mask_path_pneumonia = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Viral Pneumonia/masks/"
        masked_image_path_pneumonia = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/Viral Pneumonia/masked_images/"

        # Load the images
        images_covid = []
        images_lung_opacity = []
        images_normal = []
        images_pneumonia = []

        for i in range(100, 110):
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

        # Create a dropdown menu for image selection
        option = st.selectbox("Sélectionnez une option", ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"))

        # Randomly select a pair of [image, mask, masked_image] based on the selected option
        if option == "COVID":
            random_pair = random.choice(images_covid)
        elif option == "Lung_Opacity":
            random_pair = random.choice(images_lung_opacity)
        elif option == "Normal":
            random_pair = random.choice(images_normal)
        elif option == "Viral Pneumonia":
            random_pair = random.choice(images_pneumonia)

        # Display the selected pair
        st.image([random_pair[0], random_pair[1], random_pair[2]], caption=['Image', 'Masque', 'Image masquée'], width=200)

    run()

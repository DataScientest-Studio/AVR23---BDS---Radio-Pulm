import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from streamlit import cache #datas en cache->pas besoin de recharger


title = "Présentation des données"
sidebar_name = "Présentation des données"

# Emplacement
path = '/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/'

def load_data(url):
    df = pd.read_csv(url)  # 👈 Download the data
    return df

df = load_data("/Users/hind/Documents/AVR23---BDS---Radio-Pulm/data/df.csv")
#st.dataframe(df)
st.button("Rerun")

@st.cache_data
def load_data():
    # Import des fichiers METADATA
    df_covid = pd.read_excel(path + "COVID.metadata.xlsx")
    df_lung_opacity = pd.read_excel(path + "Lung_Opacity.metadata.xlsx")
    df_normal = pd.read_excel(path + "Normal.metadata.xlsx")
    df_pneumonia = pd.read_excel(path + "Viral Pneumonia.metadata.xlsx")

    # Création d'une colonne avec le label associé de chaque catégorie
    df_covid['label'] = 'COVID'
    df_lung_opacity['label'] = 'Lung_Opacity'
    df_normal['label'] = 'Normal'
    df_pneumonia['label'] = 'Viral Pneumonia'

    df = pd.concat([df_covid, df_lung_opacity, df_normal, df_pneumonia], axis=0)

    urls = list(df["URL"].unique())
    source_order = ["KGL_RSNA_Pneumonia", "KGL_Chest_Xray_Pneumonia", "SIRM_Covid", "GitHub_covid_repo", "Eurorad", "GitHub_covid_CXNet", "GitHub_covid_chestray_ds", "Bimcv_covid19"]

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

    return df

colors = ["red", "yellow", "green", "orange"]
category_order = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

def run():
    df = load_data()

    st.title(title)
    st.markdown("---")
    
    options = ('Dataset & Volumétrie', 'Images brutes + masquées')
    option = st.radio('', options)

    st.divider()

    if option == options[0]:
        tab1, tab2, tab3, tab4 = st.tabs(["images/catégorie", "images/source", "images/source+catégorie", "luminosités"])
    
        with tab1 :
            st.markdown("""
            Notre ensemble de données contient : 
            - **3 616** images de radios **Covid**
            - **1 345** images de radios **Pneumonie**
            - **10 192** images de radios **Normal**
            - **6 012** images de radios **Lung_Opacity**
            \n 
            Au total, **21 165** images de radios thoraciques (autant de masques) composent notre jeu de données.
            """)

            fig = go.Figure()
            for category, color in zip(category_order, colors):
                count = df['label'].value_counts()[category]
                fig.add_trace(go.Bar(x=[category], y=[count], name=category, marker=dict(color=color)))
            fig.update_layout(
                xaxis=dict(title="Catégorie"),
                yaxis=dict(title="Nombre d'images"),
                hovermode='x')
            st.plotly_chart(fig)

            with st.expander("Détails"):
                st.write("""
                Les 4 catégories d’images (variable “label”) disponibles sont :
                \n - **:red[COVID]** : Individu atteint de la COVID-19.
                \n - **:yellow[Lung_Opacity]** : Individu présentant des opacités sur sa radiographie, symptôme d’une pathologie pulmonaire (non explicitement indiquée).
                \n - **:green[Normal]** : Individu sain, sans problème aux poumons.
                \n - **:orange[Viral_Pneumonia]** : Individu atteint de pneumonie virale.
                
                \n Chaque image de n’importe quelle catégorie a un **masque** pulmonaire associé ([cf. LIEN](#option1)).
                """)

        with tab2 :
            st.subheader("Nombre d'images par source")
            #st.markdown("""\n\n\n""")

            #fig2, ax2 = plt.subplots()
            #sns.countplot(y = "source", data = df, order = df["source"].value_counts().index)
            #ax2.set_xlabel("Nombre d'images")
            #ax2.set_ylabel("Source")
            #st.pyplot(fig2)
            
            source_order = ["KGL_RSNA_Pneumonia", "KGL_Chest_Xray_Pneumonia", "SIRM_Covid", "GitHub_covid_repo", "Eurorad", "GitHub_covid_CXNet", "GitHub_covid_chestray_ds", "Bimcv_covid19"]
            count = df['source'].value_counts()[source_order]
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=count, y=source_order, orientation='h'))
            fig1.update_layout(xaxis=dict(title="Nombre d'images"), yaxis=dict(title="Source"), hovermode='x')
            st.plotly_chart(fig1, use_container_width=True)

        with tab3 :
            st.subheader("Nombre d'images par source et par catégorie")
            #st.markdown("""\n\n\n""")

            #fig3, ax3 = plt.subplots()
            #sns.countplot(y="source", data=df, hue="label", palette=colors, order=df["source"].value_counts().index, ax=ax3)
            #ax3.set_xlabel("Nombre d'images")
            #ax3.set_ylabel("Source")
            #ax3.legend(title="Catégorie", loc="center right")
            #st.pyplot(fig3)

            group_data = df[df['label'].isin(['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'])] # Filtrer données par groupe
            count = group_data.groupby(['source', 'label']).size().reset_index(name='count') # Comptage par source + groupe
            fig2 = px.bar(count, x='count', y='source', color='label', orientation='h', category_orders={'source': source_order})
            fig2.update_layout(xaxis=dict(title="Nombre d'images"), yaxis=dict(title="Source"), hovermode='x')
            st.plotly_chart(fig2, use_container_width=True)


        with tab4 :
            st.subheader("Distribution des moyennes de luminosité en fonction du jeu de données (jdd) et des sources")
            st.markdown("""\n\n\n""")

            fig4, axes4 = plt.subplots(1, 2, figsize=(20, 8))
            sns.kdeplot(x="im_mean", 
                        data=df, 
                        hue="label", 
                        palette=colors, 
                        ax=axes4[0])
            axes4[0].set_title("Distribution des moyennes de luminosité en fonction du jdd")
            
            sns.kdeplot(x="im_mean", 
                        data=df, 
                        hue="source", 
                        ax=axes4[1])
            axes4[1].set_title("Distribution des moyennes de luminosité en fonction des sources")
            
            st.pyplot(fig4)
        
    
    if option == options[1]:           

        st.subheader("Intérêt des masques pulmonaires")
       
        st.markdown(""" 
        Certaines études ont opté pour une **segmentation des poumons** comme étape initiale de leur système de détection. 
        Les auteurs de l'ensemble de notre dataset ont segmenté les images en **excluant** la partie des poumons située derrière le cœur et en suivant des repères anatomiques passant par les côtes, l'arc aortique, le péricarde et le diaphragme.
        \n
        Les **masques pulmonaires** informent les modèles qu'ils doivent **accorder plus d'attention aux régions pulmonaires** contenant les **manifestations cliniques** de la COVID-19. 
        \n
        Ces masques réduisent par conséquent l'**espace de recherche** des signes COVID-19, fournissant aux modèles des indications précises sur les zones spécifiques auxquelles il convient de prêter attention.
                    """)
        
        st.markdown("""\n\n\n""")
        st.markdown("""\n\n\n""")

        st.subheader("Images brutes / masquées")

        # definition des paths de chaque catégorie d'images
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
        option = st.selectbox("Sélectionnez une option", ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"))

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


        with st.expander("Pour en savoir plus"):
                st.write("""
                Le U-Net est une architecture de réseau de neurones utilisée pour la **segmentation sémantique** (= segmenter l'image en différentes régions et 
                attribuer une étiquette à chacune de ces régions) des poumons à partir d'images médicales. 
                Il se compose d'un **encodeur** et d'un **décodeur**, qui capturent respectivement les **informations contextuelles et spatiales**.
                Ainsi, le U-Net génère des **masques** précis en identifiant les **caractéristiques distinctives des poumons** dans les images.
                \n
                """)
                image = Image.open('/Users/hind/Desktop/illustrations/u-net-architecture.png')
                st.image(image, use_column_width=True)
                st.write("""source : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/""")



run()
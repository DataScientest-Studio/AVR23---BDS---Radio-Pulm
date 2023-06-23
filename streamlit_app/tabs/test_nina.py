import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.vgg16 import VGG16


title = "Test des modèles"
sidebar_name = "Test des modèles"
path = "C:/Users/Nina/Documents/GitHub/AVR23---BDS---Radio-Pulm/"
ID_DIR = 7
labels = ["Normal", "COVID", "Lung_Opacity", "Viral Pneumonia"]
clean_label = { "Normal" : "Normal", "COVID" : "Covid", "Lung_Opacity" : "Lung Opacity", "Viral Pneumonia" : "Viral Pneumonia"}
id_to_clean_label = { 0 : "Normal", 1 : "Covid", 2 : "Lung Opacity", 3 : "Viral Pneumonia"}


def run():
    image_path = path + "streamlit_app/assets/filling_img.jpg"
    image_name = None
    displayPrediction = False
    #Initialisation compteur de scores
    if 'Lenet' not in st.session_state :
        st.session_state['Lenet'] = 0
    if 'VGG16' not in st.session_state :
        st.session_state['VGG16'] = 0
    if 'EfficientNetB1' not in st.session_state :
        st.session_state['EfficientNetB1'] = 0
    if 'Lenet_Total' not in st.session_state :
        st.session_state['Lenet_Total'] = 0
    if 'VGG16_Total' not in st.session_state :
        st.session_state['VGG16_Total'] = 0
    if 'EfficientNetB1_Total' not in st.session_state :
        st.session_state['EfficientNetB1_Total'] = 0

    # Titre
    st.title(title)

    # Paramétrage du test
    st.subheader("Choix image et modèles")
    col1, col2 = st.columns([0.3, 0.7])
         
    with col2:
        # Onglets
        tab1, tab2 = st.tabs(["Image existante", "Import d'une image"])
        
        # Choix image existante
        with tab1:

            # Choix de catégorie
            chosen_category = st.selectbox("Catégorie à reconnaître", ("Aucune","Peu importe !", "Normal", "COVID", "Lung_Opacity", "Viral Pneumonia"))

            # Choix image (selon catégorie)
            if chosen_category == "Peu importe !" :
                category = np.random.choice(labels)
                image_name = category + "-" + str(np.random.randint(1, 1345)) +".png"
            if chosen_category == "Normal" :
                image_name_n = st.selectbox("Choix image",("Normal-1.png", "Normal-2.png", "Normal-3.png", "Normal-4.png"))
                image_name = image_name_n
                category = "Normal"
            if chosen_category == "COVID" :
                image_name_c = st.selectbox("Choix image",("COVID-1.png", "COVID-2.png", "COVID-3.png", "COVID-4.png"))
                image_name = image_name_c
                category = "COVID"
            if chosen_category == "Lung_Opacity" :
                image_name_lo = st.selectbox("Choix image",("Lung_Opacity-1.png", "Lung_Opacity-2.png", "Lung_Opacity-3.png", "Lung_Opacity-4.png"))
                image_name = image_name_lo
                category = "Lung_Opacity"
            if chosen_category == "Viral Pneumonia" :
                image_name_vp = st.selectbox("Choix image",("Viral Pneumonia-1.png", "Viral Pneumonia-2.png", "Viral Pneumonia-3.png", "Viral Pneumonia-4.png"))
                image_name = image_name_vp
                category = "Viral Pneumonia"
    
        # Chargement image importée
        with tab2:
            image_name_up = st.file_uploader("Importez une image", type=['png', 'jpeg', 'jpg'])

        #Choix des modèles
        models = st.multiselect('Modèles à comparer',['Lenet', 'VGG16', 'EfficientNetB1'],['Lenet', 'VGG16', 'EfficientNetB1'])
        
        
    # Colonne d'affichage de l'image
    with col1:
        # Récupération image importée s'il y en a une
        if (image_name_up is not None) :
            image_path = image_name_up
        # Sinon récupération image choisie s'il y en a une
        else :
            if (image_name is not None) :
                image_path = path + "data/"+ category + "/images/" + image_name

        # Affichage image à prédire
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.image(Image.open(image_path), output_format = "png")
        if (image_path == image_name_up) :
            st.write(image_name_up.name)
        elif (image_name is not None) :
            st.write(image_name)
        elif (image_name is None) :
            st.write("")

    # Bouton de lancement des prédictions
    if st.button("C'est grave docteur ?", type = "primary", disabled = (image_path == (path + "streamlit_app/assets/filling_img.jpg"))) :
        displayPrediction = True
    else :
        displayPrediction = False


    st.divider()  

    # Prédictions des modèles
    st.subheader("Prédictions")
    if displayPrediction :
        
        # Récupération de la catégorie réelle
        if (image_name_up is None) :
            label = clean_label[image_path.split('/')[ID_DIR]]
        else :
            label = "Inconnu (image importée)" #Si image importée, catégorie réelle inconnue

        # Calcul des prédictions :
        with st.spinner('Hmm, regardons cette radio...'):
            # Initialisation des modèles
            lenet, vgg16, effnet = init_models()

            # Calcul des prédictions uniquement pour les modèles sélectionnées
            modelName_to_model = { "Lenet" : lenet, "VGG16" : vgg16, "EfficientNetB1" : effnet}
            preds = {}
            for model in models :
                preds[model] = np.argmax(modelName_to_model[model].predict(image_processing(image_path, model)))

        # Affichage des prédictions
        colA1, colB1, colC1 = st.columns([0.3, 0.3, 0.4])
        with colA1:
            st.markdown("**Modèle**")
        with colB1:
            st.markdown("**Prédiction**")
        with colC1:
            st.markdown("Réalité :  **:blue[" + label + "]**")
                
        for pred in preds :
            with st.container():
                colA, colB, colC, colD = st.columns([0.3, 0.3, 0.2, 0.2])
                with colA:
                    st.markdown(pred)
                with colB:
                    st.markdown(id_to_clean_label[preds[pred]])
                with colC:
                    if (label == id_to_clean_label[preds[pred]]) :
                        st.image(Image.open(path + "streamlit_app/assets/success.png"), output_format = "png")
                        st.session_state[pred] += 1
                        st.session_state[pred + "_Total"] += 1
                    elif (label == "Inconnu (image importée)") :
                        st.image(Image.open(path + "streamlit_app/assets/unknown.png"), output_format = "png")
                    elif (label != id_to_clean_label[preds[pred]]) :
                        st.image(Image.open(path + "streamlit_app/assets/fail.png"), output_format = "png")
                        st.session_state[pred + "_Total"] += 1
                with colD :
                    st.write("**" + str(st.session_state[pred]) + "**/" + str(st.session_state[pred + "_Total"]))





# FONCTIONS APPELEES
###############################################################################################

# Fonction de processing des images pour prédiction
def image_processing(image_path, modelName):
    if modelName == "Lenet" :
        size = 256
        color_mode = "grayscale"
    elif modelName == "VGG16" :
        size = 224
        color_mode = "rgb"
    elif modelName == "EfficientNetB1" :
        size = 240
        color_mode = "rgb"
    im = tf.keras.utils.load_img(image_path, target_size = (size, size), color_mode= color_mode )
    #im = tf.keras.utils.img_to_array(im)/size
    im = np.expand_dims(im, axis = 0)
    return im


# Fonction d'initialisiation de tous les modèles
@st.cache_resource
def init_models() :
    lenet = init_lenet((256,256,1))
    vgg16 = init_vgg16()
    effnet = init_effnet((240,240,3))
    return lenet, vgg16, effnet


# Fonction d'initialisation Le_net
def init_lenet(size) :

    # Instanciation modèle séquentiel
    model = Sequential()

    # Ajout des différentes couches
    model.add(Conv2D(filters = 30 , kernel_size = (5,5), input_shape =size, activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dropout(rate = 0.2))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dense(units = 4, activation = "softmax"))

    # Compilation
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
    model.load_weights(path + "/data/models/model_lenet_2000im_30ep.h5")
    return model

# Fonction d'initialisation VGG16
def init_vgg16() :
    base_model = VGG16(weights="imagenet", include_top = False)
    for layer in base_model.layers :
        layer.trainable = False

    # Instanciation modèle séquentiel
    model = Sequential()

    # Ajout des différentes couches
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units = 1024, activation = "relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units = 512, activation = "relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units = 4, activation = "softmax"))

    # Compilation
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
    model.load_weights(path + "/data/models/model_vgg16_2000im_30ep_imnt.h5")
    return model

# Fonction d'initialisation EfficientNetB1
def init_effnet(size) :

    #Chargement et freeze modèle de base Eficient Net
    base_model = EfficientNetB1(weights = 'imagenet', include_top=False, input_shape=size)
    for layer in base_model.layers :
        layer.trainable = False

    # Création des différentes couches
    global_average = GlobalAveragePooling2D()
    dense1 = Dense(units = 1024, activation = "relu")
    dropout1 = Dropout(rate=0.2)
    dense2 = Dense(units = 512, activation = "relu")
    dropout2 = Dropout(rate=0.2)
    dense3 = Dense(units = 4, activation = "softmax")

    #Application des opérations
    x = base_model.output
    x = global_average(x)
    x = dense1(x)
    x = dropout1(x)
    x = dense2(x)
    x = dropout2(x)
    output_model = dense3(x)

    #Création du modèle
    model = Model(inputs = base_model.input, outputs = output_model)

    # Compilation
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
    model.load_weights(path + "/data/models/model_efnet1_func2_2000im_30ep.h5")
    return model

# Fonction de comptage des scores
@st.cache_resource
def init_counts() :
    return 0
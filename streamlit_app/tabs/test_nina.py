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
clean_label = { "Normal" : "Normal", "COVID" : "Covid", "Lung_Opacity" : "Lung Opacity", "Viral Pneumonia" : "Viral Pneumonia"}
id_to_clean_label = { 0 : "Normal", 1 : "Covid", 2 : "Lung Opacity", 3 : "Viral Pneumonia"}

def run():
    image_path = path + "streamlit_app/assets/filling_img.jpg"
    no_image = True
    displayPrediction = False
    

    st.title(title)

    # Paramétrage du test
    st.subheader("Choix image et modèles")
    #st.markdown(
    #"    """
    #    Testez les différents modèles de reconnaissance sur des images exemples ou sur vos propres images.
    #    """
    #)
    col1, col2 = st.columns([0.3, 0.7])
         
    with col2:
        #Choix de l'image
        tab1, tab2 = st.tabs(["Image existante", "Import d'une image"])
        
        # Image existante
        with tab1:
            chosen_image = st.selectbox(
            "Choisissez une image à reconnaître",
            ("Aucune", "image au hasard",
             'Image 1 brute - Normal', 'Image 1 masquée - Normal',
             'Image 2 brute - Covid', 'Image 2 masquée - Covid',
             'Image 3 brute - Lung_Opacity', 'Image 3 masquée - Lung_Opacity',
            'Image 4 brute - Viral Pneumonia', 'Image 4 masquée - Viral Pneumonia'))
    
        # Image importée
        with tab2:
            uploaded_image = st.file_uploader("Importez une image", type=['png', 'jpeg', 'jpg'])

        
        #Récupération de l'image correspondant au choix
        if (chosen_image == "image au hasard") :
            image_path = path + "data/COVID/images/COVID-2.png"

        models = st.multiselect('Modèles à comparer',['Lenet', 'VGG16', 'EfficientNetB1'],['Lenet', 'VGG16', 'EfficientNetB1'])

    with col1:
        st.image(Image.open(image_path), output_format = "png")
      
    if (image_path != path + "streamlit_app/assets/filling_img.jpg" ) :
        no_image = False

    if st.button("C'est grave docteur ?", type = "primary", disabled = no_image) :
        displayPrediction = True
    else :
        displayPrediction = False


    st.divider()  

    # Affichage des prédictions des modèles
    st.subheader("Prédictions")
    if displayPrediction :
        
        label = image_path.split('/')[ID_DIR]

        lenet = init_lenet((256,256,1))
        effnet = init_effnet((240,240,3))

        y_pred_lenet = np.argmax(lenet.predict(image_processing_lenet(image_path)))
        y_pred_effnet = np.argmax(effnet.predict(image_processing_effnet(image_path)))

        st.write("Réel : ", clean_label[label])
        st.write("Prédit Lenet : ", id_to_clean_label[y_pred_lenet])
        st.write("Prédit EffNetB1 : ", id_to_clean_label[y_pred_effnet])



def image_processing_lenet(image_path):
    im = tf.keras.utils.load_img(image_path, target_size = (256, 256), color_mode= "grayscale" )
    im = tf.keras.utils.img_to_array(im)/255
    im = np.expand_dims(im, axis = 0)
    return im

def image_processing_effnet(image_path):
    im = tf.keras.utils.load_img(image_path, target_size = (240, 240), color_mode= "rgb" )
    im = tf.keras.utils.img_to_array(im)/255
    im = np.expand_dims(im, axis = 0)
    return im
    

# Construction et compilation et chargement poids Le_net
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

def init_effnet(size) :
    #Définition entrée du modèle
    input_model = Input(shape = (240,240,3))

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
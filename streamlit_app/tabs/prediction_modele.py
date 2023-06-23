#https://docs.streamlit.io/

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from PIL import Image
import random
import cv2

# Classes Tensorflow
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.vgg16 import VGG16

title = ':blue[Prédictions des Modèles]'
sidebar_name = 'Prédictions'

#Différents Paths utilisés 
path = "C://Users/utilisateur/COVID19 - Projet/"
path_radios = "D://documents/GitHub/AVR23---BDS---Radio-Pulm/data/"
path_images = "C://Users/utilisateur/COVID19 - Projet/streamlit_app/images/"
path_weights = "D://documents/GitHub/AVR23---BDS---Radio-Pulm/data/models/"

#Dictionnaire pour rélier id à nom label
id_to_label = {0 : "Normal", 1 : "COVID", 2 : "Lung_Opacity", 3 : "Viral Pneumonia"}

def run():
    st.title(title)
    st.markdown("---")
    
    st.subheader(":blue[Choisir une radio pour prédiction]")
        
    col1, col2 = st.columns(2)
       
    #Selectionner une Image ou Charger une Image
    with col1 :
        
        options_categories = ("COVID", "Normal", "Lung_Opacity", "Viral Pneumonia", "Random")
        option_categories = st.radio("Sélectionner une catégorie", options_categories)
        
        if option_categories == options_categories[0]:
        
            options_image = ("COVID-1", "COVID-2", "COVID-3", "COVID-4")
            option_image = st.selectbox("Sélectionner une radio", options_image)
            path_choosen_image = path_radios + option_categories + "/images/" + option_image + ".png"
                               
        if option_categories == options_categories[1]:
        
            options_image = ("Normal-1", "Normal-2", "Normal-3", "Normal-4")
            option_image = st.selectbox("Sélectionner une radio", options_image)
            path_choosen_image = path_radios + option_categories + "/images/" + option_image + ".png"
               
        if option_categories == options_categories[2]:
        
            options_image = ("Lung_Opacity-1", "Lung_Opacity-2", "Lung_Opacity-3", "Lung_Opacity-4")
            option_image = st.selectbox("Sélectionner une radio", options_image)
            path_choosen_image = path_radios + option_categories + "/images/" + option_image + ".png"
            
        if option_categories == options_categories[3]:
        
            options_image = ("Viral Pneumonia-1", "Viral Pneumonia-2", "Viral Pneumonia-3", "Viral Pneumonia-4")
            option_image = st.selectbox("Sélectionner une radio", options_image)
            path_choosen_image = path_radios + option_categories + "/images/" + option_image + ".png"
        
        if option_categories == options_categories[4]: #Random Choice
            
            random_categorie = random.randrange(0,3)
            random_image = str(random.randrange(1,500))
            option_categories = id_to_label[random_categorie]
            
            path_choosen_image = path_radios + option_categories + "/images/" + option_categories + "-" + random_image + ".png"
            st.write(path_choosen_image)
            
                                                         
        choosen_image = Image.open(path_choosen_image)                                        
        
        agree = st.checkbox("Charger une Radio PNG")
        if agree :
            choosen_image = st.file_uploader("Charger une image PNG", accept_multiple_files = False)
            st.write(choosen_image)
            st.write("filename:", choosen_image.name)
    
    #Affichage de l'Image Selectionnée
    with col2 :
        st.subheader(":blue[Radio choisie pour prédiction]")
        st.image(choosen_image)

    
    st.divider()
    
    #Choisir les modèles à comparer 
    st.subheader(":blue[Comparaison entre modèles]")
    options_modeles = ("LeNet", "VGG16","ResNet152", "EfficientNetB1")
    option_modeles = st.multiselect("Modèles à prédire", options_modeles)
        
    st.divider()
    
    if st.button("C'est grave Docteur?") :
        displayPrediction = True
    else :
        displayPrediction = False
         
    if displayPrediction :    
        st.write("Label réel :", option_categories)
        if "LeNet" in option_modeles :
            model_lenet = lenet()
            size_lenet = 256
            y_pred_lenet = np.argmax(model_lenet.predict(image_processing(path_choosen_image, size_lenet,1)))
            st.write("Label prédit LeNet :", id_to_label[y_pred_lenet])
        
        if "VGG16" in option_modeles :
            model_vgg16 = vgg16_seq()
            size_vgg16 = 224
            y_pred_vgg16 = np.argmax(model_vgg16.predict(image_processing(path_choosen_image, size_vgg16)))
            st.write("Label prédit VGG16 :", id_to_label[y_pred_vgg16])
        
        if "ResNet152" in option_modeles :
            #model_resnet = resnet152()
            size_resnet = 256
            #y_pred_resnet = np.argmax(model_resnet.predict(image_processing(path_choosen_image, model_resnet)))
            st.write(0)
            
        if "EfficientNetB1" in option_modeles :
            model_efficientnet = efficientnetb1()
            size_efficientnetb1 = 240
            y_pred_efficientnetb1 = np.argmax(model_efficientnet.predict(image_processing(path_choosen_image, size_efficientnetb1)))
            st.write("Label prédit efficientNetB1 :", id_to_label[y_pred_efficientnetb1])
            
            #GradCAM
            VizGradCAM(model_efficientnet, img_to_array(image_gradcam(path_choosen_image, size_efficientnetb1)), plot_results=True)
        
    st.divider()
    
def image_processing(image_path, size, canaux = 3):
    
    if canaux == 1 :
        im = tf.keras.utils.load_img(image_path, target_size = (size, size), color_mode= "grayscale" )
    if canaux == 3 :
        im = tf.keras.utils.load_img(image_path, target_size = (size, size), color_mode= "rgb" )
    
    im = tf.keras.utils.img_to_array(im)/size
    im = np.expand_dims(im, axis = 0)
    return im

def image_gradcam(image_path, size, canaux = 3):
    im = tf.keras.utils.load_img(image_path, target_size = (size, size))
    return im

def lenet() :
        model = Sequential()

        model.add(Conv2D(filters = 30 , kernel_size = (5,5), input_shape =[256,256,1], activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2,2)))

        model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2,2)))

        model.add(Flatten())
        model.add(Dropout(rate = 0.2))

        model.add(Dense(units = 128, activation = "relu"))
        model.add(Dense(units = 4, activation = "softmax"))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        #hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        hp_learning_rate = 0.001

        model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
        model.load_weights(path_weights + "model_lenet_2000im_30ep.h5")

        return model
        


def vgg16() :
        #Définition entrée du modèle
        input_model = Input(shape = (224,224,3))

        #Chargement et freeze modèle de base Eficient Net
        base_model = VGG16(weights = 'imagenet', include_top=False)
        for layer in base_model.layers :
            layer.trainable = False

        # Création des différentes couches
        global_average = GlobalAveragePooling2D()
        dense1 = Dense(units = 1024, activation = "relu")
        dropout1 = Dropout(rate = 0.2)
        dense2 = Dense(units = 512, activation = "relu")
        dropout2 = Dropout(rate = 0.2)
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
        model.load_weights(path_weights + "model_vgg16_2000im_30ep_imnt.h5")
        
        return model

def vgg16_seq() :
        vgg16_base = VGG16(weights = 'imagenet', include_top = False)

        for layer in vgg16_base.layers :
            layer.trainable = False    
        model = Sequential()

        model.add(vgg16_base)

        model.add(GlobalAveragePooling2D())
        model.add(Dense(units = 1024, activation = 'relu'))
        model.add(Dropout(rate = 0.2))
        model.add(Dense(units = 512, activation = 'relu'))
        model.add(Dropout(rate = 0.2))
        model.add(Dense(units = 4, activation = 'softmax'))

        # Compilation
        model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
        model.load_weights(path_weights + "model_vgg16_2000im_30ep_imnt.h5")

        return model
    
def resnet152() :
        #Définition entrée du modèle
        input_model = Input(shape = (256,256,3))

        #Chargement et freeze modèle de base Eficient Net
        base_model = ResNet152V2(weights = None, include_top=False, input_shape=(256,256,3))
        for layer in base_model.layers :
            layer.trainable = False

        # Création des différentes couches
        global_average = GlobalAveragePooling2D()
        dense1 = Dense(units = 1024, activation = "relu")
        dropout1 = Dropout(rate = 0.2)
        dense2 = Dense(units = 512, activation = "relu")
        dropout2 = Dropout(rate = 0.2)
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
        model.load_weights(path_weights + "model_efnet1_func2_2000im_30ep.h5")
        
        return model

def efficientnetb1() :
        #Définition entrée du modèle
        input_model = Input(shape = (240,240,3))

        #Chargement et freeze modèle de base Eficient Net
        base_model = EfficientNetB1(weights = None, include_top=False, input_shape=(240,240,3))
        for layer in base_model.layers :
            layer.trainable = False

        # Création des différentes couches
        global_average = GlobalAveragePooling2D()
        dense1 = Dense(units = 1024, activation = "relu")
        dropout1 = Dropout(rate = 0.2)
        dense2 = Dense(units = 512, activation = "relu")
        dropout2 = Dropout(rate = 0.2)
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
        model.load_weights(path_weights + "model_efnet1_func2_2000im_30ep.h5")
        
        return model
    
def VizGradCAM(model, image, interpolant = 0.5, plot_results = True):

    """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
    using the gradients from the last convolutional layer. This function
    should work with all Keras Application listed here:
    https://keras.io/api/applications/
    Parameters:
    model (keras.model): Compiled Model with Weights Loaded
    image: Image to Perform Inference On
    plot_results (boolean): True - Function Plots using PLT
                            False - Returns Heatmap Array
    Returns:
    Heatmap Array?
    """
    #sanity check
    assert (interpolant > 0 and interpolant < 1), "Heatmap Interpolation Must Be Between 0 - 1"

    #STEP 1: Preprocesss image and make prediction using our model
    #input image
    original_img = np.asarray(image, dtype = np.float32)
    #expamd dimension and get batch size
    img = np.expand_dims(original_img, axis = 0)
    #predict
    prediction = model.predict(img)
    #prediction index
    prediction_idx = np.argmax(prediction)

    #STEP 2: Create new model
    #specify last convolutional layer
    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, keras.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    #compute gradient of top predicted class
    with tf.GradientTape() as tape:
        #create a model with original model inputs and the last conv_layer as the output
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        #pass the image through the base model and get the feature map  
        conv2d_out, prediction = gradient_model(img)
        #prediction loss
        loss = prediction[:, prediction_idx]

    #gradient() computes the gradient using operations recorded in context of this tape
    gradients = tape.gradient(loss, conv2d_out)

    #obtain the output from shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
    output = conv2d_out[0]

    #obtain depthwise mean
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))


    #create a 7x7 map for aggregation
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    #multiply weight for every layer
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    #resize to image size
    activation_map = cv2.resize(activation_map.numpy(), 
                                (original_img.shape[1], 
                                 original_img.shape[0]))
    #ensure no negative number
    activation_map = np.maximum(activation_map, 0)
    #convert class activation map to 0 - 255
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    #rescale and convert the type to int
    activation_map = np.uint8(255 * activation_map)


    #convert to heatmap
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    #superimpose heatmap onto image
    original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cvt_heatmap = img_to_array(cvt_heatmap)

    #enlarge plot
    plt.rcParams["figure.dpi"] = 100

    if plot_results == True:
        
        plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
        plt.axis('off')
        
    else:
        return cvt_heatmap    
        

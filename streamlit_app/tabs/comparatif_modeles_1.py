import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.vgg16 import VGG16


title = "Comparatif entre Modèles"
sidebar_name = "Comparatif entre Modèles"
path = "D://documents/GitHub/AVR23---BDS---Radio-Pulm/"

ID_DIR = 6
labels = ["Normal", "COVID", "Lung_Opacity", "Viral Pneumonia"]
clean_label = { "Normal" : "Normal", "COVID" : "Covid", "Lung_Opacity" : "Lung Opacity", "Viral Pneumonia" : "Viral Pneumonia"}
id_to_clean_label = { 0 : "Normal", 1 : "Covid", 2 : "Lung Opacity", 3 : "Viral Pneumonia"}


def run():
    st.markdown(
        """
        #
        """)
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
    if 'Resnet101' not in st.session_state :
        st.session_state['Resnet101'] = 0
    if 'Resnet101_Total' not in st.session_state :
        st.session_state['Resnet101_Total'] = 0
        
        
    # Titre
    st.title(title)
    st.divider()

    # Paramétrage du test
    st.subheader("Choix image et modèles")
    col1, col2 = st.columns([0.3, 0.7])
         
    with col2:
        # Onglets
        tab1, tab2 = st.tabs(["Image existante", "Import d'une image"])
        
        # Choix image existante
        with tab1:

            # Choix de catégorie
            chosen_category = st.selectbox("__Catégorie à reconnaître :__", ("Aucune","Peu importe !", "Normal", "COVID", "Lung_Opacity", "Viral Pneumonia"))

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
        models = st.multiselect('__Modèles à comparer :__',['Lenet', 'VGG16', 'EfficientNetB1','Resnet101'],['Lenet', 'VGG16', 'EfficientNetB1','Resnet101'])
        
        
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
            lenet, vgg16, effnet, resnet101 = init_models()

            # Calcul des prédictions uniquement pour les modèles sélectionnées
            modelName_to_model = { "Lenet" : lenet, "VGG16" : vgg16, "EfficientNetB1" : effnet, "Resnet101" : resnet101}
            probas = {}
            preds = {}
            for model in models :
                print(model)
                if (model == "Resnet101") :
                    proba = eval(image_path)
                else:
                    proba = modelName_to_model[model].predict(image_processing(image_path, model))
                probas[model] = np.max(proba)
                preds[model] = np.argmax(proba)

        # Affichage des prédictions

        colA1, colB1, colC1 = st.columns([0.25, 0.4, 0.35])
        with colA1:
            st.markdown("**Modèle**")
        with colB1:
            st.markdown("**Prédiction**")
        with colC1:
            st.markdown("Réalité :  **:blue[" + label + "]**")
                
        for pred, proba in zip(preds, probas) :
            colA, colB, colC, colD = st.columns([0.25, 0.4, 0.15, 0.2])
            with colA:
                st.markdown(pred)
            with colB:
                st.markdown(id_to_clean_label[preds[pred]] + " - _" + str(round(probas[proba]*100, 2)) + "%_")
                st.markdown("#") 
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
    resnet101 = init_resnet101()
    return lenet, vgg16, effnet, resnet101


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
#############################################################################################################################################################    

#PyTorch
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
torch.manual_seed(1)

import torchvision
from torchvision import transforms, models

# data augmentation library
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

invNorm = transforms.Normalize(( -0.509/0.229 ),( 1/0.229))
def displayTensorNorm(t):
    trans = transforms.ToPILImage()
    return trans(invNorm(t))

normalize = transforms.Normalize(
    mean=[0.509],
    std=[0.229]
)

dataTransforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    normalize])

def init_resnet101() :
    resnet101 = torchvision.models.resnext101_32x8d(pretrained=False)
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    resnet101.conv1 = new_conv
    in_features = resnet101.fc.in_features
    features = list(resnet101.fc.children())[:-1]
    features.extend([nn.Linear(in_features,4),nn.LogSoftmax(dim=1)])
    resnet101.fc = nn.Sequential(*features)
    
    resnet101.load_state_dict(torch.load(path+'/data/models/resnet_model.pth', map_location=torch.device('cpu')))
    return resnet101
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dff, transform):
        super(Dataset, self).__init__()
        # Store the filenames and labels
        self.samples = np.array([],dtype=int)
        self.labels = np.array([],dtype=int)
        self.samples = np.append(self.samples,dff)
        self.labels = np.append(self.labels,4)
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,i):
        img = Image.open(self.samples[i])
        # Return the image with the sample label or ID
        return self.transform(img), self.labels[i]
    
def eval(dff):
    testData = Dataset(dff,dataTransforms)
    eval_dl = torch.utils.data.DataLoader(testData,batch_size=1,num_workers=0)
    resnet101 = init_resnet101()
    resnet101.eval()
    probas=[]
    for i, batch in enumerate(eval_dl):
        with torch.no_grad():
            x = batch[0]
            y = resnet101(x)
            print(y)            
            prob = F.softmax(y, dim=1)
            # Récupérer la probabilité pour chaque classe
            proba=[]
            proba.append( prob[0][0].item())
            proba.append( prob[0][1].item())
            proba.append( prob[0][2].item())
            proba.append( prob[0][3].item())
            probas.append(proba)
    return probas

# Fonction de comptage des scores
@st.cache_resource
def init_counts() :
    return 0
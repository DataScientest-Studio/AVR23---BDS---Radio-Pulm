import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.vgg16 import VGG16


title = "Exemples de tirages"
sidebar_name = "Exemples de tirages"

# Constantes
labels = ["Normal", "COVID", "Lung_Opacity", "Viral Pneumonia"]
image_type_repo = { "Images brutes" : "images", "Images masquées (poumons uniquement)" : "masked_images"}
path = "C:/Users/Nina/Documents/GitHub/AVR23---BDS---Radio-Pulm/"
id_to_label = { 0 : "Normal", 1 : "COVID", 2 : "Lung_Opacity", 3 : "Viral Pneumonia"}

def run():
    # Instanciation variables
    displayPrediction = False

    st.title(title)
    st.divider()
    colA, colB, colC= st.columns((0.4,0.1,0.4))
    with colA:
        model_name = st.selectbox("Sélection du modèle", ['Lenet', 'VGG16', 'EfficientNetB1'], index = 2 )
        if (model_name == "EfficientNetB1") :
            grad_cam_On = st.checkbox("Afficher GradCam", value = True)
    with colC:
        image_type =st.radio("Type d'images", ("Images brutes", "Images masquées (poumons uniquement)"))
        number = st.slider("Nombre d'images à tirer", min_value = 1, max_value = 15, value = 6, step = 1, format = "%g")
        
    
    if st.button("Lancer un tirage", type = "primary") :
        displayPrediction = True
    else :
        displayPrediction = False

    st.divider()

    if displayPrediction :
        
        # Tirage de 10 images au hasard du type choisi
        image_paths = []
        label_names = []
        for i in range(number) :
            label_name = np.random.choice(labels)
            image_name = label_name + "-" + str(np.random.randint(1000, 1275)) +".png"
            image_path = path + "data/"+ label_name + "/" + image_type_repo[image_type] + "/" + image_name
            image_paths.append(image_path)
            label_names.append(label_name)
        df = pd.DataFrame(list(zip(image_paths, label_names)), columns = ['filepath', 'nameLabel'])
        #st.write(df)

        # Chargement des poids du modèle (si pas fait avant)
        model = init_model(model_name, image_type_repo[image_type])

        # Calcul prédictions
        df["predicted"]= df["filepath"].apply(lambda x : id_to_label[np.argmax(model.predict(image_processing(x, model_name)))])

        # Affichage image, catégorie et prédiction
        nbCol = 3
        grid = st.columns(nbCol)
        col = 0
        for i in range(len(df)):
            with grid[col]:
                col1, col2 = st.columns((0.1,0.9))

                # Affichage icone vrai/faux
                with col1 :
                    if (df["nameLabel"][i] == df["predicted"][i]) :
                        st.image(Image.open(path + "streamlit_app/assets/success.png"), output_format = "png")
                    else :
                        st.image(Image.open(path + "streamlit_app/assets/fail.png"), output_format = "png")

                # Affichage label prédit et réel
                with col2 :
                    st.write("Prédiction :", df["nameLabel"][i])
                    st.write("Réalité :", df["predicted"][i])

                # Affichage GradCam si effNet et sélectionné
                if ((model_name == "EfficientNetB1") & grad_cam_On) :
                    st.image(VizGradCAM(model, img_to_array(Image.open(df["filepath"][i]).resize((240,240))), plot_results=True))
                # Affichage image sinon
                else :
                    st.image(Image.open(df["filepath"][i]), output_format = "png")
                st.write("\n  \n \n")

            col = (col + 1) % nbCol


# FONCTIONS APPELEES
###############################################################################################

# Fonction d'initialisiation d'un modèle
@st.cache_resource
def init_model(model, image_type) :
    if (model == 'Lenet') :
        return init_lenet((256,256,1), image_type)
    if (model == 'VGG16') :
        return init_vgg16(image_type)
    if (model == 'EfficientNetB1') :
        return init_effnet((240,240,3), image_type)
    
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

# Fonction d'initialisation Le_net
def init_lenet(size, image_type) :

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
    if (image_type == "images") :
        model.load_weights(path + "/data/models/model_lenet_2000im_30ep.h5")
    else :
        model.load_weights(path + "/data/models/model_lenet_2000imk_30ep.h5")
    return model

# Fonction d'initialisation VGG16
def init_vgg16(image_type) :
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
    if (image_type == "images") :
        model.load_weights(path + "/data/models/model_vgg16_2000im_30ep_imnt.h5")
    else :
        model.load_weights(path + "/data/models/model_vgg_2000imk_30ep.h5")
    return model

# Fonction d'initialisation EfficientNetB1
def init_effnet(size, image_type) :

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
    if (image_type == "images") :
        model.load_weights(path + "/data/models/model_efnet1_func2_2000im_30ep.h5")
    else :
        model.load_weights(path + "/data/models/model_efnet1_func2_2000imk_30ep.h5")
    return model

# Fonction de GradCam

def VizGradCAM(model, image, interpolant=0.5, plot_results=True):
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
    img = np.expand_dims(original_img, axis=0)
    #predict
    prediction = model.predict(img)
    #prediction index
    prediction_idx = np.argmax(prediction)

    #STEP 2: Create new model
    #specify last convolutional layer
    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
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
    cvt_heatmap = tf.keras.utils.img_to_array(cvt_heatmap)

    #enlarge plot
    plt.rcParams["figure.dpi"] = 100

    if plot_results == True:
        return(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
    else:
        return cvt_heatmap
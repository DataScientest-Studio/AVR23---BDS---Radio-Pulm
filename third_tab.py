import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from PIL import Image
import random
import cv2


title = 'Partie 3'
sidebar_name = 'Modélisation'

def run():
    st.title(title)
    st.markdown("""DONNEES""")
        
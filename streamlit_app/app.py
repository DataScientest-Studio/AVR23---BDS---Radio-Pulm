from collections import OrderedDict

import streamlit as st
from PIL import Image
import config
import torch
from tabs import accueil, comparatif_modeles, test_par_modele, modelisation, contexte_medical, presentation_datas


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

# A CHANGER EN LOCAL !!
path = "C:/Users/Nina/Documents/GitHub/AVR23---BDS---Radio-Pulm/"


with open(path + "streamlit_app/style.css", "r") as f:
    style = f.read()



st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


TABS = OrderedDict(
    [
        (accueil.sidebar_name, accueil),
        (contexte_medical.sidebar_name, contexte_medical),
        (presentation_datas.sidebar_name, presentation_datas),
        (modelisation.sidebar_name, modelisation),
        (test_par_modele.sidebar_name, test_par_modele),
        (comparatif_modeles.sidebar_name, comparatif_modeles),
    ]
)


def run():
    #st.sidebar.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png", width=200,)
    st.sidebar.image(Image.open(path + "streamlit_app/assets/ds_studio.png"), width=200)
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()

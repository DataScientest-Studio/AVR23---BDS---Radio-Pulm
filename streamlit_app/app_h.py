from collections import OrderedDict
import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import accueil, contexte_medical, presentation_datas

# A CHANGER EN LOCAL !!
path = "/Users/hind/Documents/AVR23---BDS---Radio-Pulm/"


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open(path + "streamlit_app/style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by passing the name in the sidebar as key and 
# the imported tab as value as follow :
TABS = OrderedDict(
    [
        (accueil.sidebar_name, accueil),
        (contexte_medical.sidebar_name, contexte_medical),
        (presentation_datas.sidebar_name, presentation_datas)
    ]
)

def run():
    st.sidebar.image("/Users/hind/Documents/AVR23---BDS---Radio-Pulm/streamlit_app/assets/logo_dst.png", width=200)

    st.sidebar.markdown("### Navigation")
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    

    st.sidebar.markdown(f"## {config.PROMOTION}")
    st.sidebar.markdown("### Team members")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]
    tab.run()

if __name__ == "__main__":
    run()
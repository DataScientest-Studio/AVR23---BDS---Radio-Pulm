import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import datetime
import altair as alt

title = "Contexte médical"
sidebar_name = "Contexte médical"


def run():

    st.title(title)
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(['COVID-19', 'Détection', 'Radiologie'])

    st.divider()

    with tab1 : 
            
        with st.expander("INFOGRAPHIE COVID-19 - Données OMS de janvier 2020 à juin 2023"):
                def run():
                    data = pd.read_csv('/Users/hind/Desktop/data_covid/world_data.csv')
                    data = data.rename(columns={'Cumulative_cases' : 'Cas cumulés', 
                                                'Cumulative_deaths' : 'Décès cumulés'})
                    data['Date'] = pd.to_datetime(data['Date'])
                    Countries = ['All_countries'] + list(data['Country'].unique())
                    cases = pd.read_csv('/Users/hind/Desktop/data_covid/fr_new_cases.csv')
                    cases['Date'] = pd.to_datetime(cases['Date'])

                    # 1re LIGNE DE DATA
                    world_panel = st.container()
                    with world_panel:
                        columns = st.columns([2, 0.2, 2.1, 0.2, 1.2, 0.2, 1.2])
                        with columns[0]:
                            start_time = st.slider("Selectionnez la date", 
                                                data['Date'].min().date(), 
                                                data['Date'].max().date(),
                                                data['Date'].max().date())
                            current_data = data.loc[data['Date'] == pd.to_datetime(start_time)]

                            #st.dataframe(current_data)

                            country = columns[2].selectbox(
                                'Pays',
                                (Countries),
                                key = "country")

                            if country == 'All_countries':
                                country_data = data.groupby('Date').sum().reset_index()
                            else:
                                country_data = data.loc[data['Country'] == country]

                            #st.dataframe(country_data)

                            timespan = columns[4].radio(
                                'Time',
                                ('All Time', '30 derniers jours'))

                            max_date = data['Date'].max()
                            past_30_days = max_date - datetime.timedelta(days=30)

                            if timespan == 'All Time':
                                filter_data = country_data
                            else:
                                filter_data = country_data.loc[country_data['Date'] > past_30_days]
                            
                            #st.dataframe(filter_data)

                            daily_type = columns[6].radio(
                                'Type',
                                ('Cas cumulés', 'Décès cumulés'))
                        

                        chart1, chart2 = st.columns([3,3])
                        
                        with chart1:
                            url = "https://raw.githubusercontent.com/deldersveld/topojson/master/world-countries.json"
                            pays = alt.topo_feature(url, "countries1")

                            # Code adapté de https://altair-viz.github.io/gallery/choropleth.html
                            world_map = alt.Chart(pays).mark_geoshape(stroke='lightgray').encode(
                                color=alt.Color('Cas cumulés:Q'),
                                tooltip=['countries1:N', 'Cas cumulés:Q']
                            ).transform_lookup(
                                lookup='properties.name',
                                from_=alt.LookupData(current_data, 'Country', list(current_data.columns))
                            ).project(
                                type='naturalEarth1'
                            )

                            st.altair_chart(world_map, use_container_width=True)

                        with chart2:
                            line = alt.Chart(filter_data).mark_line().encode(
                                x = alt.X('Date:T'),
                                y = alt.Y(daily_type),
                                tooltip = ['Date', daily_type]
                            ).properties(
                                title = daily_type + ' in ' + country
                            ).interactive(
                                bind_y = False #zoom
                            )
                            st.altair_chart(line, use_container_width=True)


                    #2e LIGNE DE DATA
                    france_panel = st.container()
                    with france_panel:
                        columns = st.columns([4, 1])

                    with columns[0]:
                        bar_chart = alt.Chart(cases).mark_bar().encode(
                            x = alt.X('Date:T'),
                            y = alt.Y('New_cases:Q'),
                            tooltip = ['Date', 'New_cases']
                        ).properties(
                            title = 'Nouveaux cas en France'
                        ).interactive(
                            bind_y=False
                        )

                        line = alt.Chart(cases).mark_rule(color='red').encode(
                            y = 'mean(New_cases):Q',
                            size = alt.SizeValue(2),
                            tooltip = ['mean(New_cases)']   
                        )

                        chart = bar_chart + line
                        
                        st.altair_chart(chart, use_container_width=True)
                run()

        col1, col2 = st.columns([5, 0.5])

        with col1 : 
            st.subheader("COVID-19")
            st.markdown("""
            La COVID-19, déclarée par l'OMS en tant que pandémie mondiale, est un **syndrome respiratoire aigu** apparu en décembre 2019. 
            Elle a déjà entraîné près de **7 millions** de décès et infecté plus de **770 millions** de personnes jusqu'au mois de juin 2023. 
                        """)
            st.markdown("""
            Les symptômes courants comprennent : 
            \n - fièvre
            \n - toux
            \n - dyspnée (difficulté respiratoire)
            \n - anomalies pulmonaires à l'imagerie
                        """)
            
            st.markdown('\n\n\n')
            st.markdown('\n\n\n')

            st.markdown("""
            Les **cas graves** peuvent provoquer un **syndrome de détresse respiratoire aiguë (SDRA)** ou une **insuffisance respiratoire**, nécessitant une ventilation mécanique et des soins intensifs. 
            Les personnes **immunodéprimées** et **âgées** sont les plus à risque de développer des formes graves, pouvant aller jusqu'à une **défaillance cardiaque, rénale** ou un **choc septique**.
                        """)
            image0 = Image.open('/Users/hind/Desktop/illustrations/radio_scan_covid.png')
            st.image(image0, use_column_width=True, width=150)
                
        
        with col2 : 
            image0 = Image.open('/Users/hind/Desktop/illustrations/blank.png')
            st.image(image0, use_column_width=True)
            image0 = Image.open('/Users/hind/Desktop/illustrations/blank.png')
            st.image(image0, use_column_width=True)
            image1 = Image.open('/Users/hind/Desktop/illustrations/fievre.png')
            st.image(image1, use_column_width=True)
            image2 = Image.open('/Users/hind/Desktop/illustrations/toux.png')
            st.image(image2, use_column_width=True)
            image3 = Image.open('/Users/hind/Desktop/illustrations/dyspnee.png')
            st.image(image3, use_column_width=True)
       

    with tab2 :
        
        col1, col2 = st.columns([4, 2.5])
        
        with col1 :
            st.subheader("Outils de détection")
            
            st.markdown("""
            La détection de la COVID-19 est un enjeu crucial, cependant, cela peut s'avérer difficile en raison de la **symptomatologie similaire à d'autres infections virales**. 
            La méthode de diagnostic actuellement privilégiée est la **RT-PCR**, mais elle présente des **limites** : 
            \n - faux négatifs en cas de prélèvement inapproprié
            \n - charge virale basse
            \n - mutations génétiques 
            \n - nécessité de laboratoires spécialisés
            \n - délais de traitement des échantillons longs 
                        """)
    
        with col2 :
            st.markdown("\n\n\n")
            image = Image.open('/Users/hind/Desktop/illustrations/rt_pcr.png')
            st.image(image, use_column_width=True, width=100)
        
        st.markdown('\n\n\n')
        st.markdown('\n\n\n')
        
        st.markdown("""
        Malgré leurs performances supérieures, les **scanners** présentent certains inconvénients : leur **sensibilité est limitée aux premiers stades de la COVID-19**, nécessitant plus de **temps** et **exposant** à une dose plus élevée de rayonnements.
        \n Les **radiographies** sont par conséquent plus souvent utilisées comme **premier outil de dépistage** : plus **simples**, plus **rapides** et **moins coûteuses**. 
                    """)    
    
    with tab3 :
        st.subheader("Notions de radiologie et signes")
        
        st.markdown("""
        Pour rappel, en radiologie, tout ce qui est opaque aux rayons X apparait **blanc** (tissu, liquide : on parle d’**opacité**) et le reste apparait **noir** (air : on parle de **clarté**). 
        On peut distinguer plusieurs niveaux radiologiques selon la distribution des lésions : 
        \n - **Alvéolaire** (opacité systématisée (pneumonie) ou floues et confluentes (OAP))
        \n - **Nodulaire** (opacités arrondies)
        \n - **Interstitiel** (opacités diffuses ou localisée, miliaires (tuberculose))
        \n - **Bronchique** (opacité systématisée et homogène)
        \n - **Pleural** (épanchement liquidien)
                    """)
        
        st.markdown('\n\n\n')
        
        image = Image.open('/Users/hind/Desktop/illustrations/patho_pulmo.png')
        st.image(image, use_column_width=True, width=100)
        
        st.markdown('\n\n\n')
        st.markdown('\n\n\n')
       
        col = st.columns([2,4])

        with col[0]:
            selected = st.radio(
                "Selectionnez une radio",
                options=["Normal", "COVID", "Pneumonie", "Autre PP"]
            )

        with col[1]:
            if selected == 'Normal':
                st.image('/Users/hind/Desktop/illustrations/radio_normal.png', width=500)

            elif selected == 'COVID':
                st.image('/Users/hind/Desktop/illustrations/radio_covid.png', width=500)
                
            elif selected == 'Pneumonie':
                st.image('/Users/hind/Desktop/illustrations/radio_pneumonie.png', width=500)

            elif selected == 'Autre PP':
                st.image('/Users/hind/Desktop/illustrations/radio_lung_opacity.png', width=500)



run()





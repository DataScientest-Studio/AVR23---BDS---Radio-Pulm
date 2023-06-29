from cv2 import mean
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime

# load dataset
data = pd.read_csv('/Users/hind/Desktop/data_covid/world_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
Countries = ['All_countries'] + list(data['Country'].unique())

cases = pd.read_csv('/Users/hind/Desktop/data_covid/fr_new_cases.csv')
cases['Date'] = pd.to_datetime(cases['Date'])



# 1re LIGNE DE DATA
world_panel = st.container()
with world_panel:
    columns = st.columns([2, 0.2, 2.1, 0.2, 1.2, 0.2, 1.2])
    with columns[0]:
        start_time = st.slider("select the date", 
                               data['Date'].min().date(), 
                               data['Date'].max().date(),
                               data['Date'].max().date())
        current_data = data.loc[data['Date'] == pd.to_datetime(start_time)]

        #st.dataframe(current_data)

        country = columns[2].selectbox(
            'Country',
            (Countries),
            key = "country"

        )


        if country == 'All_countries':
            country_data = data.groupby('Date').sum().reset_index()
        else:
            country_data = data.loc[data['Country'] == country]

        #st.dataframe(country_data)


        timespan = columns[4].radio(
            'Time',
            ('All Time', 'Last 30 days')
        )


        max_date = data['Date'].max()
        past_30_days = max_date - datetime.timedelta(days=30)

        if timespan == 'All Time':
            filter_data = country_data
        else:
            filter_data = country_data.loc[country_data['Date'] > past_30_days]
        
        #st.dataframe(filter_data)


        daily_type = columns[6].radio(
            'Type',
            ('Cumulative_cases', 'Cumulative_deaths')
        )
    

    chart1, chart2 = st.columns([1,2])
    
    with chart1:
        url = "https://raw.githubusercontent.com/deldersveld/topojson/master/world-countries.json"

        pays = alt.topo_feature(url, "countries1")

        # Adapt code from https://altair-viz.github.io/gallery/choropleth.html
        world_map = alt.Chart(pays).mark_geoshape(stroke='lightgray').encode(
            color=alt.Color('Cumulative_cases:Q'),
            tooltip=['countries1:N', 'Cumulative_cases:Q']
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
        title = 'New cases in France'
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
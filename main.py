import pandas as pd
import numpy as np
import streamlit as st
from streamlit import caching
import requests
import json
from Functions import *
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pandas.io.json import json_normalize



# st.set_page_config(layout="wide")
st.set_page_config(layout="wide", page_icon="📈", page_title="Adam's app")
st.title("📈 Stocks Analyzing app")
st.markdown('This Application is for Instant access for **real-time and historical** data of stocks using API. :chart:')


st.sidebar.header('Filter panel')
# measure = st.sidebar.selectbox('Choose Measure',('close','open','high', 'low', 'volume') ,index=0)
# interval = st.sidebar.selectbox('Choose Interval',('1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '1day', '1week', '1month') ,index=0)
# history = st.sidebar.number_input('Choose History' ,value=30)
# predict = st.sidebar.button('Predict')

country = st.sidebar.selectbox(
    'Please Select Country',
    ('United States', 'Israel',  'Germany', 'China', 'South Korea', 'Hong Kong', 'Malaysia',
     'Taiwan', 'India', 'United Kingdom', 'Italy', 'Mexico',
     'Saudi Arabia', 'Japan', 'Australia', 'Singapore', 'France',
     'Venezuela', 'Norway', 'Sweden', 'Thailand', 'Netherlands',
     'Denmark', 'Spain', 'Hungary', 'South Africa', 'Brazil', 'Ireland',
     'Canada', 'Greece', 'Romania', 'Indonesia',
     'Finland', 'Switzerland', 'Kuwait', 'United Arab Emirates',
     'Chile', 'Argentina', 'New Zealand', 'Belgium', 'Qatar',
     'Russia', 'Botswana', 'Turkey', 'Portugal', 'Lithuania',
     'Czech Republic', 'Estonia', 'Iceland', 'Latvia', 'Austria',
     'Egypt'))

data = get_country_stocks(country)

with st.sidebar.form("my_form"):

    stocks = st.multiselect('Select Stock', data.name.unique())

    
    measure = st.multiselect('Select Measure', ('close', 'open', 'high', 'low', 'volume','close_rolling_mean'), default=["close"])
    interval = st.selectbox('Select Interval', (
    '1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '1day', '1week', '1month'), index=0)
    history = st.number_input('Select History', value=30)
    # predict = st.button('Predict')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
if submitted:

    filtered_df = data.query(f'name=={stocks}')

    # st.table(filtered_df)

    symbol_data = get_symbl_data(filtered_df.symbol.unique(), interval, history)

    st.subheader('Real-Time Stocks Price:')
# Getting last date close measure value and compare it real time price
    counter = 0
    collist = st.columns(len(filtered_df.symbol.unique()))

    for s in list(filtered_df.symbol.unique()):

        try:
            f_df = symbol_data[symbol_data['symbol'] == s].sort_values(by='datetime', ascending=False)
            last_close = f_df.iloc[0]['close']

            rt_close = get_real_time_price(s)

            diff = (float(rt_close) - float(last_close))

            collist[counter].metric(label=f'Real-Time Price of {s}', value=rt_close, delta=diff,delta_color='normal')
            counter = counter + 1
        except Exception as s:

            st.write(f'Currently no real-time data available for {s}')


    with st.expander("Inspect row Data"):
        try:
            fdf = symbol_data.copy()
            fdf.set_index(['datetime', 'symbol'], inplace=True)
            d = fdf.groupby(['datetime', 'symbol']).agg({'close': min, 'open': min})
            st.table(d.unstack(1))
        except:
            st.write(f"**symbol** {stocks} is not available")


    with st.expander("Prediction of  Stocks closing Price"):
        st.markdown('This is a simple Linear regression ML model in order to predict Stocks closing price.')
        st.markdown('The model take stocks last 100 minutes and return prediction for next Closing price ')
        counter_1 = 0
        collist_1 = st.columns(len(filtered_df.symbol.unique()))


        for stc in list(filtered_df.symbol.unique()):
            preddf, score, prediction ,modelscore= get_symbol_prediction(stc)
            # st.write('model score: ', modelscore)

            todayclose = preddf.iloc[0]['close']

            diff = str((float(prediction[0]) - float(todayclose)))


            collist_1[counter_1].metric(label=f"Model Score for {stc} is:  {modelscore:.2f}", value=prediction[0], delta=diff,delta_color='normal')
            counter_1 = counter_1 + 1


    with st.container():

        try:
            for meas in measure:
                st.header(meas)
                fig = px.line(symbol_data, x="datetime", y=meas, color="symbol" ,
                              template='seaborn', width=1000, height=600)#,text=meas

                fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                #
                # st.plotly_chart(fig)

        except Exception as err:
            # st.exception(err)
            st.write(f"**symbol** {stocks} is not available ! Please Choose Something else")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('starting')



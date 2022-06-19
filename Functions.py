import pandas as pd
import streamlit as st
import requests
import json
from datetime import datetime
from pandas.io.json import json_normalize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score


pd.set_option('display.precision', 2)


# Functions
@st.cache(suppress_st_warning=True)
def get_country_stocks(country='Israel'):
    try:
        st.write(country)
        url = "https://twelve-data1.p.rapidapi.com/stocks"

        querystring = {"country": {country}, "format": "json"}

        headers = {
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com",
            "X-RapidAPI-Key": "f4d9dcda63msh765702e167b413dp18cc9ajsn7ba4641a6019"
        }

        response = requests.request("GET", url, headers=headers, params=querystring)

        dict = json.loads(response.text)
        df = pd.json_normalize(dict['data'], max_level=5)
        pd.set_option('display.precision', 2)
        return df
    except:
        st.write(f'NOTE: something went wrong! Please try again')


@st.cache(suppress_st_warning=True)
def get_symbl_data(symbols, interval='1min', history='5'):

    df = pd.DataFrame()
    for sym in symbols:

        url = "https://twelve-data1.p.rapidapi.com/time_series"

        querystring = {"symbol": sym, "interval": {interval}, "outputsize": {history}, "format": "json"}

        headers = {
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com",
            "X-RapidAPI-Key": "f4d9dcda63msh765702e167b413dp18cc9ajsn7ba4641a6019"
        }

        response = requests.request("GET", url, headers=headers, params=querystring)


        if response.json()['status'] == 'ok':

            dict = json.loads(response.text)
            df1 = pd.json_normalize(dict['values'], max_level=5)
            df1['symbol'] = sym
            df = df.append(df1)
        else:
            st.write(f"**symbol** {sym} is not available")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['close_rolling_mean'] = df.close.rolling(2, min_periods=1).mean()


    pd.set_option('display.precision', 2)

    return df


@st.cache(suppress_st_warning=True)
def get_real_time_price(symbol):
    url = "https://twelve-data1.p.rapidapi.com/price"

    querystring = {"symbol": {symbol}, "format": "json", "outputsize": "30"}

    headers = {
        "X-RapidAPI-Key": "f4d9dcda63msh765702e167b413dp18cc9ajsn7ba4641a6019",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    if len(response.json()) == 1:
        return float(response.json()['price'])
    else:
        st.write(f'NO Real-Time price for {symbol}')
        return None

def get_symbol_prediction(symbolname):
    df = pd.DataFrame()
    url = "https://twelve-data1.p.rapidapi.com/time_series"

    querystring = {"symbol": symbolname, "interval": "1min", "outputsize": "100", "format": "json"}

    headers = {
        "X-RapidAPI-Key": "f4d9dcda63msh765702e167b413dp18cc9ajsn7ba4641a6019",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    # st.write('API Status is: ', response.json()['status'])
    dict = json.loads(response.text)
    df1 = pd.json_normalize(dict['values'], max_level=5)
    # df1['symbol'] = s

    df = df.append(df1)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['diff'] = df['close'] - df['open']
    df['tommorow_close'] = df.close.shift(periods=1)
    df['weekday'] = df.datetime.dt.weekday

    df.sort_values(by=['datetime'], ascending=False, inplace=True)

    df_topredict = df[df['datetime'] == max(df.datetime)]

    df.drop(index=0, axis=0,inplace=True)

    df_test = df.iloc[:3]
    df_train = df.iloc[4:]



    X_train = df_train.drop(['datetime','tommorow_close'], axis=1)
    y_train = df_train['tommorow_close']

    X_test = df_test.drop(['datetime', 'tommorow_close'], axis=1)
    y_test = df_test['tommorow_close']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    scaler = StandardScaler()
    x_train_scale = scaler.fit_transform(X_train)
    x_test_scale = scaler.fit_transform(X_test)

    model = LinearRegression()
    # model = SGDRegressor(max_iter=1000, shuffle=False)

    model.fit(x_train_scale,y_train)
    y_pred = model.predict(x_test_scale)

    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    model_score = model.score(x_train_scale, y_train)

    # X_final = df.drop(['datetime', 'tommorow_close'], axis=1)
    # y_final = df['tommorow_close']
    #
    # X_final_sc = scaler.fit_transform(X_final)
    #
    # model.fit(X_final_sc,y_final)
    #
    df_topredict.drop(['datetime', 'tommorow_close'], axis=1, inplace=True)
    # st.write('Line to Predict',df_topredict)
    scal = scaler.fit_transform(df_topredict)

    prediction = model.predict(scal)


    return  df_topredict, r2_score(y_test, y_pred), prediction, model_score





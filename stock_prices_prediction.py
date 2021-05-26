import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf
import investpy

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def format_date(date):
    return datetime.datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')

@st.cache(persist=True)
def get_stocks_data(): 
    stocks_df = investpy.get_stocks(country='brazil')

    return stocks_df

def get_stock_data(stock):
    stock_data = yf.Ticker(f'{stock}.SA')

    df = stock_data.history(period="max", interval="1d")

    df = df.reset_index()

    return df

def generate_predictions(stock):
    df = get_stock_data(stock)

    df['MM_3'] = df['Close'].rolling(window=3).mean()
    df['MM_9'] = df['Close'].rolling(window=9).mean()
    df['MM_17'] = df['Close'].rolling(window=17).mean()

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    df['Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    total_length = len(df)
    train_length = int(total_length * 0.7)
    test_length = train_length + int(total_length * 0.25) - 1

    features = df.drop(['Date', 'Close', 'Open', 'MM_3', 'MM_9'], 1)
    labels = df['Close']

    scaler = MinMaxScaler().fit(features)
    features_scaled = scaler.transform(features)

    X_train = features_scaled[:train_length]
    X_test = features_scaled[train_length:test_length]

    y_train = labels[:train_length]
    y_test = labels[train_length:test_length]

    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_train, y_train)
    predictions = lr_model.predict(X_test)

    X_validation = features_scaled[test_length:total_length]

    date = df['Date']
    date = date[test_length:total_length]

    close = df['Close']
    close = close[test_length:total_length]

    prediction = lr_model.predict(X_validation)

    df_validation=pd.DataFrame({ 'Date': date, 'Close': close, 'Prediction': prediction })
    df_validation['Close'] = df_validation['Close'].shift(+1)

    last_date = df_validation.iloc[-1]['Date']
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    df_validation.set_index('Date', inplace=True)

    def is_business_day(date):
        return bool(len(pd.bdate_range(date, date)))

    for i in prediction:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        next_date = datetime.datetime.strftime(next_date, '%Y-%m-%d')
        
        if (is_business_day(next_date)):
            df_validation.loc[next_date] = [np.nan for _ in range(len(df_validation.columns)-1)]+[i]
        
    df_validation.reset_index(inplace=True)

    return df_validation

st.sidebar.subheader("Tabela")
table = st.sidebar.empty()

st.sidebar.subheader("Pesquise pela ação:")

form = st.sidebar.form(key='stock-form')

stocks_df = get_stocks_data()

stocks_list = sorted(stocks_df['symbol'].map(lambda stock: stock).unique())
stock_selected = form.selectbox(
    label="Ação",
    options=stocks_list,
)
submit = form.form_submit_button('Gerar predições')

st.title("Stock prices prediction")
filter_info = st.empty()
description = st.empty()

if stock_selected:
    df = get_stock_data(stock_selected)

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

    layout = go.Layout(
        title=f'{stock_selected} Candlestick chart',
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )

    fig.update_layout(layout)

    st.plotly_chart(fig)

if submit:
    df_validation = generate_predictions(stock_selected)

    x_date = df_validation['Date']
    y_close = df_validation['Close']
    y_prediction = df_validation['Prediction']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_date,
        y=y_close,
        name = 'Close'
    ))
    fig.add_trace(go.Scatter(
        x=x_date,
        y=y_prediction,
        name = 'Prediction'
    ))

    fig.update_layout(title=f'{stock_selected} future predictions')

    st.plotly_chart(fig)

if table.checkbox("Mostrar tabela com os dados") and stock_selected:
    df = get_stock_data(stock_selected)
    st.write(df)
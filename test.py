import warnings
warnings.simplefilter('ignore')

import pandas as pd
from prophet import Prophet

import streamlit as st

df = pd.read_csv('data.csv')

df['ds'] = pd.to_datetime(df['ds'])

m = Prophet(interval_width=0.95,daily_seasonality=True)
model = m.fit(df)

future = m.make_future_dataframe(periods=200,freq='D')
forecast = m.predict(future)
print('--------------')
print(forecast[['ds','yhat_lower','yhat_upper','yhat']])
st.write(forecast)
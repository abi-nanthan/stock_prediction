import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Correct date format for yfinance
start = '2015-01-01'
today = date.today().strftime('%Y-%m-%d')

st.title('STOCK PREDICTION APP')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select stock', stocks)

n_years = st.slider('Years of prediction', 1, 4)
period = n_years * 365


st.sidebar.title('Welcome!')
st.sidebar.write("""
This is a simple stock prediction app using streamlit of python.
And I have used a api package that is 'yfinance' for companies
real time stock data.
For forecasting I have used 'prophet' package of python.
And for plotting I have used 'plotly' package.
""")

if selected_stock:

    @st.cache_data
    def load_data(stock):
        data = yf.download(stock, start, today)
        data.reset_index(inplace=True)
        return data

    data = load_data(selected_stock)

    st.subheader('Stock Data')
    st.write(data.tail())

    # Forecasting with Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet()
    model.fit(df_train)

    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.subheader('Forecast Plot')
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.subheader('Forecast Components')
    fig2 = model.plot_components(forecast)
    st.write(fig2)

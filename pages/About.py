import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="https://cryptologos.cc/logos/xrp-xrp-logo.png?v=032"
)

key_features = """
- Historical Data Analysis: We analyze Crypto's past price trends to understand the underlying patterns and volatility.
- Next 30 Days Forecast: Our model predicts Crypto's closing prices for the upcoming 30 days, offering you a glimpse into potential future movements.
- Interactive Graph: Dive into our dynamic chart that compares the last 15 days of actual prices with the predicted prices for the next 30 days. This allows for easy comparison and better insight into price trends.
- User-Friendly Interface: Our web page is designed to be intuitive, making it simple for you to navigate through the data and predictions.
"""
st.markdown("### Key Features:")
st.markdown(key_features)

how_it_works = """
- Data Collection: We collect and preprocess Crypto price data from reliable sources.
- Model Training: Using LSTM (Long Short-Term Memory) neural networks, our model is trained on historical data to learn complex patterns.
- Prediction: The trained model generates forecasts for the next 30 days, which are then scaled back to actual price values for accuracy.
- Visualization: The predictions are displayed alongside recent actual prices in an easy-to-understand line chart.
"""
st.markdown("### How It Works:")
st.markdown(how_it_works)

st.markdown('### Disclaimer')
st.markdown("Please note that while our predictions are based on advanced models and historical data, they are not guaranteed and should not be considered as financial advice. Always conduct your own research and consult with a professional before making any investment decisions.")

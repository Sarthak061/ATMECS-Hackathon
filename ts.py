import streamlit as st
import pandas as pd
import sqlite3 as sq
import datetime
import yfinance as yf
from preprocess import preprocessing
import time
import google.generativeai as genai
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

st.set_page_config(page_title="Market Mentor", layout="wide", page_icon="📈")
# Custom styled navigation bar
st.markdown("""
    <nav style="padding: 10px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);">
        <h1 style="color: white; text-align: center; 
                    text-shadow: 2px 2px 4px rgba(255, 0, 0, 0.7);">
            Market Mentor
        </h1>
    </nav>
""", unsafe_allow_html=True)

# Introductory Markdown for the sidebar
st.sidebar.markdown("""
    <style>
        .sidebar-title {
            color: #FF5733;  /* Bright color for title */
            font-size: 20px; /* Title font size */
            font-weight: bold; /* Title weight */
        }
        .sidebar-text {
            color: #FFFFFF; /* White color for text */
            font-size: 16px; /* Slightly larger text font size */
            line-height: 1.5; /* Line height for readability */
        }
        .sidebar-container {
            background-color: #1D1E22; /* Dark background for sidebar */
            padding: 10px; /* Padding for container */
            border-radius: 8px; /* Rounded corners */
        }
    </style>

    <div class="sidebar-container">
        <div class="sidebar-title">Welcome to Market Mentor!</div>
        <div class="sidebar-text">
            🚀 Your hub for stock market insights.<br>
            Select options on the left to get started!
        </div>
    </div>
                    <br>
""", unsafe_allow_html=True)



# Custom CSS styling to hide a specific element
st.markdown(
    """
    <style>
    .st-emotion-cache-h4xjwg {
        position: fixed;
        top: 0px;
        left: 0px;
        right: 0px;
        height: 3.75rem;
        background: rgb(14, 17, 23);
        outline: none;
        z-index: 999990;
        display: none; /* Hides the element */
    }
    @media (min-width: 576px) {
        .st-emotion-cache-1jicfl2 {
            padding-left: 5rem;
            padding-right: 5rem;
            top: unset;
            padding-top: inherit;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

tab2, tab1 = st.tabs(["Market Mentor","RAG Bot"])

with tab1:
    model = genai.GenerativeModel('gemini-1.5-pro-002')
    st.subheader("Market Mentor Chatbot Interaction")

    # Chatbot Free Input
    user_query = st.text_input("Ask me anything related to the stock market or business:", "")

    # Some default prompts for user convenience
    st.markdown("**Try these default prompts by clicking on them:**")
    default_prompts = [
        "What are the key drivers affecting the stock market today?",
        "What are the historical trends for Apple (AAPL) stock?",
        "How does the S&P 500 look for the next 6 months?",
        "Is now a good time to invest in Tesla (TSLA)?",
        "How do interest rates affect stock prices?"
    ]

    # Display default prompts as clickable buttons
    for prompting in default_prompts:
        if st.button(prompting):
            user_query = prompting

    if user_query:
        # Generate response from the chatbot (this will use the Generative AI API)
        final_input_tab1 = f"You are a financial advisor for this conversation - henceforth do not respond anything like 'i am not a financial advisor', 'i can not give financial advices' or any similar sentence in your responses please refrain from talking about yourself and focus on context of the query. the query is: {user_query}"

        # Call the generative AI API
        response = model.generate_content(final_input_tab1)

        # Display the response from the chatbot
        st.write(response.text)


with tab2:

    "This is an app to predict the High and Low of the any Stock, generate market leads and trends & answer any business related queries"
    db = sq.connect('stocks.db')

    # Query to get distinct countries
    qry = "SELECT DISTINCT(Country) FROM tkrinfo;"
    count = pd.read_sql_query(qry, db)

    # Country selection with tooltip
    country_select = st.sidebar.selectbox(
        "Select country", 
        count,
        help="🔧 **Country Selection**: Choose a country from which you want to see stock information."
    )

    # Query to get distinct exchanges based on the selected country
    qry = "SELECT DISTINCT(Exchange) FROM tkrinfo WHERE Country = '" + country_select + "'"
    exchange = pd.read_sql_query(qry, db)

    # Exchange selection with tooltip
    exchange_select = st.sidebar.selectbox(
        "Select exchange", 
        exchange, 
        index=1,
        help="🔧 **Exchange Selection**: Select the stock exchange available in the chosen country."
    )

    # Query to get distinct stock names based on the selected exchange
    qry = "SELECT DISTINCT(Name) FROM tkrinfo WHERE Exchange = '" + exchange_select + "'"
    name = pd.read_sql_query(qry, db)

    # Stock name selection with tooltip
    choice_name = st.sidebar.selectbox(
        "Select the Stock", 
        name,
        help="🔧 **Stock Selection**: Choose the stock for which you want to retrieve data."
    )


    # get stock tickr
    qry = "SELECT DISTINCT(Ticker) FROM tkrinfo WHERE Exchange = '" + exchange_select + "'" + "and Name = '" + choice_name + "'"
    tckr_name = pd.read_sql_query(qry, db)
    tckr_name = tckr_name.loc[0][0]

    # st.write("This is a nice country  ", country_select)
    # st.write("It has exchange:,",exchange_select)
    # st.write(choice_name)

    # get start date
    #start_date = st.sidebar.date_input("Start Date", value=datetime.date.today() - datetime.timedelta(days=30))
    #st.write(start_date)

    # get end date
    #end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
    #st.write(end_date)
    #st.write(str(tckr_name))

    # Get interval with a q-tip
    intvl = st.sidebar.selectbox(
        "Select Interval", 
        ['1d', '1wk', '1mo', '3mo'],
        help="⏳ Choose the interval based on your trading strategy:\n"
            "- **1d** (Daily): Ideal for day traders or short-term analysis.\n"
            "- **1wk/1mo** (Weekly/Monthly): Good for identifying medium-term trends.\n"
            "- **3mo** (Quarterly): Useful for observing quarterly trends or cycles."
    )

    # Get period with a q-tip
    prd = st.sidebar.selectbox(
        "Select Period", 
        ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'], 
        index=2,
        help="🗓️ Choose the time period for analysis:\n"
            "- **1mo - 6mo**: For short-term trend analysis.\n"
            "- **1y - 5y**: Great for analyzing medium-term market behavior.\n"
            "- **10y - Max**: For long-term investments or understanding historical performance."
    )


    # get stock data
    stock = yf.Ticker(str(tckr_name))
    #data = stock.history(interval=intvl, start=start_date, end=end_date)
    data = stock.history(interval=intvl, period=prd)

    if len(data) == 0:
        st.write("Unable to retrieve data. This ticker may no longer be in use. Try some other stock.")
    else:
        # Preprocessing
        data = preprocessing(data, intvl)

        # Forecast horizon based on period and interval with q-tips
        if prd == '1mo' or prd == '3mo':
            set_horizon = st.sidebar.slider(
                "Forecast horizon", 1, 15, 5, 
                help="📅 Select the number of time units (days/weeks) to forecast ahead. For shorter periods (1-3 months), forecasting a smaller horizon can offer more accurate results."
            )
        else:
            if intvl == '1d' or intvl == '1wk':
                set_horizon = st.sidebar.slider(
                    "Forecast horizon", 1, 30, 5, 
                    help="📅 Select how far into the future to forecast. For daily or weekly data, you may want a longer forecast horizon (up to 30 time units)."
                )
            else:
                set_horizon = st.sidebar.slider(
                    "Forecast horizon", 1, 15, 5, 
                    help="📅 Choose the forecast horizon. For longer intervals (monthly or quarterly), a shorter horizon can help maintain accuracy."
                )

        # Model selection with q-tips
        model = st.selectbox(
            'Model Selection', 
            [
                'Simple Exponential Smoothing', 
                'Holt Model', 
                'Holt-Winter Model', 
                'Auto Regressive Model',
                'Moving Average Model', 
                'ARMA Model', 
                'ARIMA Model', 
                'AutoARIMA',
                'Linear Regression', 
                'Random Forest', 
                'Gradient Boosting', 
                'Support Vector Machines'
            ],
            help=(
                "🧠 Choose a forecasting model:\n\n"
                "- **Simple Exponential Smoothing**: Suitable for time series data without trend or seasonality. It gives more weight to recent observations, making it useful for short-term forecasts.\n\n"
                "- **Holt Model**: Extends simple exponential smoothing to capture linear trends in data. This model is appropriate when there is a trend present but no seasonality.\n\n"
                "- **Holt-Winter Model**: This model accounts for both trend and seasonality. It is best for data that exhibits seasonal patterns along with trends, providing more accurate forecasts in such cases.\n\n"
                "- **Auto Regressive Model (AR)**: This model predicts future behavior based on past behavior. It assumes that the current value is a linear combination of previous values, making it ideal for stationary time series data.\n\n"
                "- **Moving Average Model (MA)**: This model uses past forecast errors to predict future values. It is often used in conjunction with other models to refine forecasts and capture noise in the data.\n\n"
                "- **ARMA Model**: Combines both AR and MA models. It's best suited for stationary time series data where both trends and seasonality are absent, providing a comprehensive approach for forecasting.\n\n"
                "- **ARIMA Model**: The AutoRegressive Integrated Moving Average model is a generalization of ARMA for non-stationary time series data. It includes differencing to make the data stationary before applying ARMA techniques.\n\n"
                "- **AutoARIMA**: Automatically identifies the best ARIMA model parameters (p, d, q) for your data, simplifying the modeling process for users unfamiliar with ARIMA complexities.\n\n"
                "- **Linear Regression**: A basic predictive modeling technique that estimates relationships between variables. It can be used for forecasting by modeling the relationship between a dependent variable and one or more independent variables.\n\n"
                "- **Random Forest**: An ensemble learning method that uses multiple decision trees to improve prediction accuracy. It's effective for both regression and classification tasks and can handle large datasets with higher dimensionality.\n\n"
                "- **Gradient Boosting**: Another ensemble technique that builds models sequentially. Each new model attempts to correct errors made by previous models, leading to better performance. It's widely used in competitive machine learning.\n\n"
                "- **Support Vector Machines (SVM)**: Primarily a classification algorithm but can also be used for regression. It works well for high-dimensional spaces and is effective in cases where the number of dimensions exceeds the number of samples."
            )
        )



        if model == 'Simple Exponential Smoothing':
            col1, col2 = st.columns(2)
            
            with col1:
                alpha_high = st.slider(
                    "Alpha_high", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Alpha_high** controls the smoothing factor for high prices. A higher value gives more weight to recent observations, which can lead to more responsive forecasts but may also increase volatility in predictions."
                )
                
            with col2:
                alpha_low = st.slider(
                    "Alpha_low", 
                    0.0, 
                    1.0, 
                    0.25,
                    help="🔧 **Alpha_low** is the smoothing factor for low prices. Similar to Alpha_high, this parameter adjusts how much influence recent low prices have on the forecast. Choosing a suitable value is essential for balancing responsiveness and stability in forecasts."
                )
            
            from SES import SES_model
            data_final, smap_low, smap_high, optim_alpha_high, optim_alpha_low = SES_model(data, set_horizon, alpha_high, alpha_low)

            # Display the resulting data
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMAPE for High: {}".format(smap_high))
                st.write("Optimal Alpha for High: {} ".format(optim_alpha_high))
                
            with col2:
                st.write("SMAPE for Low: {}".format(smap_low))
                st.write("Optimal Alpha for Low: {} ".format(optim_alpha_low))


        elif model == 'Holt Model':
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                level_high = st.slider(
                    "Level High", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Level High** controls the baseline level for high prices. This parameter adjusts the forecast to capture the current level of high prices accurately."
                )
            
            with col2:
                trend_high = st.slider(
                    "Trend High", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Trend High** influences the rate of increase or decrease in the high prices. A higher trend value allows the model to adapt to upward or downward trends in high prices more rapidly."
                )
            
            with col3:
                level_low = st.slider(
                    "Level Low", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Level Low** establishes the baseline level for low prices. It helps the model accurately reflect the current state of low prices in the forecasts."
                )
            
            with col4:
                trend_low = st.slider(
                    "Trend Low", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Trend Low** defines the change in low prices over time. Adjusting this parameter helps the model track increasing or decreasing trends in low prices effectively."
                )
            
            from SES import Holt_model
            data_final, smap_low, smap_high, optim_level_high, optim_level_low, optim_trend_high, optim_trend_low = Holt_model(
                data, 
                set_horizon, 
                level_high, 
                level_low, 
                trend_high, 
                trend_low
            )
            
            # Display the resulting data
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMAPE for High: {}".format(smap_high))
                st.write("Optimal Level for High: {}".format(optim_level_high))
                st.write("Optimal Trend for High: {}".format(optim_trend_high))
                
            with col2:
                st.write("SMAPE for Low: {}".format(smap_low))
                st.write("Optimal Level for Low: {}".format(optim_level_low))
                st.write("Optimal Trend for Low: {}".format(optim_trend_low))



        elif model == 'Holt-Winter Model':
            col1, col2 = st.columns(2)
            
            with col1:
                level_high = st.slider(
                    "Level High", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Level High** sets the baseline level for high prices. Adjusting this affects the forecast's response to the current high price level."
                )
                trend_high = st.slider(
                    "Trend High", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Trend High** determines the rate of change in high prices. A higher value allows the model to react quickly to upward or downward trends."
                )
                season_high = st.slider(
                    "Seasonal High", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Seasonal High** captures the seasonal effects on high prices. Adjust this to improve the model's ability to forecast based on seasonal patterns."
                )
            
            with col2:
                level_low = st.slider(
                    "Level Low", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Level Low** sets the baseline level for low prices, similar to Level High. It helps adjust forecasts according to the current state of low prices."
                )
                trend_low = st.slider(
                    "Trend Low", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Trend Low** influences how low prices change over time. Higher values make the model more responsive to changes in low price trends."
                )
                season_low = st.slider(
                    "Seasonal Low", 
                    0.0, 
                    1.0, 
                    0.20,
                    help="🔧 **Seasonal Low** accounts for seasonal fluctuations in low prices. Adjusting this helps improve forecasts based on seasonal trends."
                )
            
            from SES import Holt_Winter_Model
            data_final, smap_low, smap_high, optim_level_high, optim_level_low, optim_trend_high, optim_trend_low, optim_season_high, optim_season_low = Holt_Winter_Model(
                data, 
                set_horizon, 
                level_high, 
                level_low, 
                trend_high, 
                trend_low, 
                season_high, 
                season_low
            )

            # Display the resulting data
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMAPE for High: {}".format(smap_high))
                st.write("Optimal Level for High: {}".format(optim_level_high))
                st.write("Optimal Trend for High: {}".format(optim_trend_high))
                st.write("Optimal Seasonal smoothing for High: {}".format(optim_season_high))
            
            with col2:
                st.write("SMAPE for Low: {}".format(smap_low))
                st.write("Optimal Level for Low: {}".format(optim_level_low))
                st.write("Optimal Trend for Low: {}".format(optim_trend_low))
                st.write("Optimal Seasonal smoothing for Low: {}".format(optim_season_low))


        elif model == 'Auto Regressive Model':
            col1, col2 = st.columns(2)
            
            with col1:
                p_high = st.slider(
                    "Order of High", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of High (p)** specifies the number of lag observations included in the model for high prices. Increasing this value allows the model to consider more past high prices, which can improve the forecast but may also lead to overfitting."
                )
            
            with col2:
                p_low = st.slider(
                    "Order of Low", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of Low (p)** indicates the number of lag observations for low prices. Similar to Order of High, adjusting this parameter helps the model learn from more past low price data, potentially enhancing accuracy."
                )
            
            from SES import AR_model

            data_final, smap_high, smap_low = AR_model(data, set_horizon, p_high, p_low)
            
            # Display the resulting data
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMAPE of High: {}".format(smap_high))
            
            with col2:
                st.write("SMAPE of Low: {}".format(smap_low))


        elif model == 'Moving Average Model':
            col1, col2 = st.columns(2)
            
            with col1:
                q_high = st.slider(
                    "Order of High", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of High (q)** specifies the number of lagged forecast errors to include in the model for high prices. Increasing this value allows the model to better capture the underlying patterns by considering more past errors, but may also lead to overfitting."
                )
            
            with col2:
                q_low = st.slider(
                    "Order of Low", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of Low (q)** indicates the number of lagged forecast errors for low prices. Adjusting this parameter helps the model learn from past errors in low price forecasts, potentially improving its predictive accuracy."
                )
            
            from SES import AR_model
            data_final, smap_high, smap_low = AR_model(data, set_horizon, q_high, q_low)
            
            # Display the resulting data
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMAPE of High: {}".format(smap_high))
            
            with col2:
                st.write("SMAPE of Low: {}".format(smap_low))


        elif model == 'ARMA Model':
            col1, col2 = st.columns(2)
            
            with col1:
                p_high = st.slider(
                    "Order of AR High", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of AR High (p)** is the number of lag observations included in the autoregressive model for high prices. A higher value allows the model to consider more past values of high prices, which can lead to better forecasts but may also increase the risk of overfitting."
                )
                q_high = st.slider(
                    "Order of MA High", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of MA High (q)** specifies the number of lagged forecast errors in the moving average model for high prices. Adjusting this parameter can help capture patterns in the residuals of the forecast, improving accuracy."
                )
            
            with col2:
                p_low = st.slider(
                    "Order of AR Low", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of AR Low (p)** determines the number of lag observations for the autoregressive model applied to low prices. Increasing this order allows the model to incorporate more past low price data, potentially enhancing forecasts."
                )
                q_low = st.slider(
                    "Order of MA Low", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of MA Low (q)** indicates the number of lagged forecast errors for low prices in the moving average model. This parameter helps the model learn from past forecast errors to improve its predictions."
                )
            
            from SES import ARMA_model
            data_final, smap_high, smap_low = ARMA_model(data, set_horizon, p_high, p_low, q_high, q_low)
            
            # Display the resulting data
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMAPE of High: {}".format(smap_high))
            
            with col2:
                st.write("SMAPE of Low: {}".format(smap_low))


        elif model == 'ARIMA Model':
            col1, col2 = st.columns(2)
            
            with col1:
                p_high = st.slider(
                    "Order of AR High", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of AR High (p)** is the number of lag observations included in the autoregressive model for high prices. A higher order can capture more past behavior but may lead to overfitting."
                )
                q_high = st.slider(
                    "Order of MA High", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of MA High (q)** specifies the number of lagged forecast errors in the moving average model for high prices. This helps improve the forecast by capturing patterns in the residuals."
                )
                i_high = st.slider(
                    "Order of Differencing High", 
                    0, 
                    10, 
                    0,
                    help="🔧 **Order of Differencing High (d)** is used to make the time series stationary by subtracting the previous observation from the current observation. A value of 0 indicates no differencing."
                )
            
            with col2:
                p_low = st.slider(
                    "Order of AR Low", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of AR Low (p)** indicates the number of lag observations for the autoregressive model applied to low prices. Increasing this order allows for more past data to be included."
                )
                q_low = st.slider(
                    "Order of MA Low", 
                    1, 
                    30, 
                    1,
                    help="🔧 **Order of MA Low (q)** indicates the number of lagged forecast errors for low prices. This parameter helps refine the model based on past forecast errors."
                )
                i_low = st.slider(
                    "Order of Differencing Low", 
                    0, 
                    10, 
                    0,
                    help="🔧 **Order of Differencing Low (d)** is used to make the low price time series stationary. Similar to the high price differencing, a value of 0 means no differencing."
                )
            
            from SES import ARIMA_model
            data_final, smap_high, smap_low = ARIMA_model(data, set_horizon, p_high, p_low, q_high, q_low, i_high, i_low)
            
            # Display the resulting data
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("SMAPE of High: {}".format(smap_high))
            
            with col2:
                st.write("SMAPE of Low: {}".format(smap_low))

        elif model == 'AutoARIMA':
            from SES import Auto_Arima
            st.write("Note: This model may take some time to fit")
            data_final = Auto_Arima(data,set_horizon)
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])

        else:
            from ML_models import forecast
            #data_final = forecast(data,set_horizon,model)
            data_final, smape_high, smape_low = forecast(data,set_horizon,model)
            st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])

            col1, col2 = st.columns(2)
            with col1:
                st.write("SMAPE of High: {}".format(smape_high))
            with col2:
                st.write("SMAPE of Low : {}".format(smape_low))

    db.close()


    import streamlit as st
    import google.generativeai as genai
    from PIL import Image
    import io
    import base64
    with st.container(border = True):
        st.subheader(f"Have any queries related to {choice_name}? Ask here!")

        # api_key = st.secrets["google_api_key"]
        api_key = "AIzaSyAEI2JWO9CMGWF5ZEI8EeAZzd6lRdTa9TE"

        if api_key:
            genai.configure(api_key=api_key)

            prompt = st.text_input(
                "What's your query?", 
                "How should I invest in the stock market?",
                help="🔧 **Your Query**: Enter your investment-related question here. Be as specific as possible for better advice, e.g., 'What are the best stocks to buy this year?'"
            )


            if st.button("Generate Response"):
                try:
                    model = genai.GenerativeModel('gemini-1.5-pro-002')
                    prompt_initial = f"You are a financial advisor for this conversation - henceforth do not respond anything like 'i am not a financial advisor', 'i can not give financial advices' or any disclaimer that you are an AI - please refrain from talking about yourself and any similar sentence in your responses, focus on context of the query, the company that i am asking this question about is: {choice_name}, the company is of {country_select} and is listed under {exchange_select} the question is:" + prompt
                    stock_data_summary = f"Stock: {choice_name}\nHistorical Data: \n{data.to_string()}\n"

                    # Concatenate user query with stock data context
                    final_input = f"User Query: {prompt_initial}\n consider this in Context as well, these are the historical stock data values:\n{stock_data_summary}"
                    response = model.generate_content(final_input)

                    if hasattr(response, 'text') and response.text:
                        st.subheader("Generated Response")
                        st.write(response.text)

                    if hasattr(response, 'images') and response.images:
                        st.subheader("Generated Images")
                        
                        for image_data in response.images:
                            if 'data' in image_data: 
                                img_bytes = base64.b64decode(image_data['data'])
                            elif 'raw_data' in image_data:
                                img_bytes = image_data['raw_data']
                            else:
                                continue 

                            image = Image.open(io.BytesIO(img_bytes))

                            st.image(image, caption="Generated Image", use_column_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("API key is missing. Please add your API key to secrets.")

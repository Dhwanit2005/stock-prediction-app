Stock Prediction & Trading Guidance App
=========================================

Description:
-------------
This Streamlit app downloads historical stock data, computes a comprehensive set of technical indicators, and trains a machine learning model to predict whether a stock’s closing price will be above its opening price on a given day. The app provides interactive visualizations, performs backtesting of a simple trading strategy, and gives dynamic trading guidance based on the model’s predictions.

Features:
----------
- Stock Selection:
  Supports stocks from the U.S. (NASDAQ, NYSE), Australia (ASX with .AX suffix), and India (NSE with .NS suffix).

- Technical Indicators:
  Computes a wide range of indicators including:
  - Base Indicators: RSI, MACD, Bollinger Bands, Stochastic Oscillator, On-Balance Volume (OBV), Money Flow Index (MFI), 10-day and 50-day moving averages, and volatility.
  - Additional Indicators: ATR (Average True Range), ADX (Average Directional Index), CCI (Commodity Channel Index), Williams %R, and Chaikin Money Flow (CMF).
  - Lagged Features: 1-day and 2-day lagged versions for several key indicators to capture short-term momentum shifts.

- Model Training:
  Uses TimeSeriesSplit with RandomizedSearchCV to tune a Random Forest classifier on historical data (excluding the last 365 days, which serve as a holdout set). Optionally, an XGBoost model can be trained and ensembled with the Random Forest for improved performance.

- Interactive Visualizations:
  Interactive graphs (using Plotly) display:
  - The last 365 days’ close prices with buy signals.
  - The cumulative return of a simple backtested strategy.

- Backtesting:
  A simple strategy is simulated on the holdout period where, on days with a "buy" signal, the strategy assumes buying at market open and selling at market close. The app calculates daily and cumulative returns.

- Dynamic Trading Guidance:
  Based on the model’s predicted probability for tomorrow’s price movement:
  - Buy Signal: If the probability is above 50%, the app shows a "Recommended BUY" signal with guidance to buy at market open and sell at market close.
  - Sell Signal: If below 50%, it advises caution and suggests waiting for clearer signals.

- Disclaimer:
  This tool is for educational purposes only and does not constitute financial advice. Always perform your own research and consult a professional before making trading decisions.

How to Run the App:
--------------------
1. Clone the Repository:
   Open your terminal and run:
     git clone https://github.com/Dhwanit2005/stock-prediction-app.git
     cd stock-prediction-app

2. Install Dependencies:
   Install the required packages by running:
     pip install -r requirements.txt
   (If you don't have a requirements.txt, you can generate one with:
     pip freeze > requirements.txt)

3. Run the App:
   Start the app with:
     streamlit run app.py
   Your browser will open the app interface.

Requirements:
--------------
- Python 3.7 or higher
- streamlit
- yfinance
- ta
- scikit-learn
- xgboost
- matplotlib
- pandas
- numpy
- pytz
- plotly

License:
---------
This project is licensed under the MIT License. See the LICENSE file for details.

Disclaimer:
-----------
This application is for educational purposes only. The predictions and trading guidance provided do not constitute financial advice. Use this tool at your own risk.

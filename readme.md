# Stock Prediction & Trading Guidance App

This Streamlit app downloads historical stock data, computes a comprehensive set of technical indicators, and trains a machine learning model to predict whether a stock’s closing price will be above its opening price on a given day. The app provides interactive visualizations, performs backtesting of a simple trading strategy, and gives dynamic trading guidance based on the model’s predictions.

## Features

- **Stock Selection:** Supports stocks from the U.S. (NASDAQ, NYSE), Australia (ASX with .AX suffix), and India (NSE with .NS suffix).
- **Technical Indicators:** Computes a wide range of indicators including RSI, MACD, Bollinger Bands, Stochastic Oscillator, On-Balance Volume (OBV), Money Flow Index (MFI), moving averages, volatility, and additional indicators like ATR, ADX, CCI, Williams %R, and Chaikin Money Flow (CMF). Also includes 1-day and 2-day lagged features for key indicators.
- **Model Training:** Uses TimeSeriesSplit with RandomizedSearchCV to tune a Random Forest classifier on historical data (excluding the last 365 days, which serve as a holdout set). Optionally, an XGBoost model can be trained and ensembled with the Random Forest for improved performance.
- **Interactive Visualizations:** Displays interactive graphs (using Plotly) such as:
  - The last 365 days’ close prices with buy signals.
  - The cumulative return of a simple backtested strategy.
- **Backtesting:** Simulates a simple trading strategy on the holdout period where, on days with a "buy" signal, it assumes buying at market open and selling at market close.
- **Dynamic Trading Guidance:** Based on the model’s predicted probability for tomorrow’s price movement:
  - **Buy Signal:** If the probability is above 50%, the app shows a "Recommended BUY" signal.
  - **Sell Signal:** If below 50%, it advises caution.
- **Disclaimer:** This tool is for educational purposes only and does not constitute financial advice. Always perform your own research and consult a professional before making trading decisions.

## How to Run the App

1. **Clone the Repository:**

   Open your terminal and run the following commands:
   ```bash
   git clone https://github.com/Dhwanit2005/stock-prediction-app.git
   cd stock-prediction-app
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.7 or higher installed. Then, install the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```
   *Tip:* If the `requirements.txt` file is missing or outdated, you can generate one with:
   ```bash
   pip freeze > requirements.txt
   ```

3. **Run the App:**

   Launch the Streamlit app by executing:
   ```bash
   streamlit run app.py
   ```
   Your default web browser will automatically open the app interface.

## Requirements

- Python 3.7 or higher
- [Streamlit](https://www.streamlit.io/)
- [yfinance](https://pypi.org/project/yfinance/)
- [ta](https://github.com/bukosabino/ta)
- [scikit-learn](https://scikit-learn.org/)
- [xgboost](https://xgboost.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [pytz](https://pypi.org/project/pytz/)
- [Plotly](https://plotly.com/python/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for educational purposes only. The predictions and trading guidance provided do not constitute financial advice. Use this tool at your own risk.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import pytz

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb

###############################################################################
# 1) BIG STOCK LIST (US, Australia, and Indian Stocks)
###############################################################################
def get_stock_list():
    """
    Returns a large selection of popular stocks from:
      - U.S. (NASDAQ, NYSE)
      - Australia (ASX, with .AX suffix)
      - India (NSE, with .NS suffix)
    """
    return [
        # US Tech & Growth
        {"symbol": "AAPL",   "name": "Apple Inc.",                     "market": "NASDAQ"},
        {"symbol": "MSFT",   "name": "Microsoft Corporation",          "market": "NASDAQ"},
        {"symbol": "AMZN",   "name": "Amazon.com Inc.",                "market": "NASDAQ"},
        {"symbol": "GOOG",   "name": "Alphabet Inc. (Class C)",        "market": "NASDAQ"},
        {"symbol": "GOOGL",  "name": "Alphabet Inc. (Class A)",        "market": "NASDAQ"},
        {"symbol": "TSLA",   "name": "Tesla Inc.",                     "market": "NASDAQ"},
        {"symbol": "NVDA",   "name": "NVIDIA Corporation",             "market": "NASDAQ"},
        {"symbol": "META",   "name": "Meta Platforms Inc.",            "market": "NASDAQ"},
        {"symbol": "NFLX",   "name": "Netflix Inc.",                   "market": "NASDAQ"},
        {"symbol": "INTC",   "name": "Intel Corporation",              "market": "NASDAQ"},
        {"symbol": "CSCO",   "name": "Cisco Systems Inc.",             "market": "NASDAQ"},
        {"symbol": "PEP",    "name": "PepsiCo Inc.",                   "market": "NASDAQ"},
        # US Large-Cap (NYSE)
        {"symbol": "JPM",    "name": "JPMorgan Chase & Co.",           "market": "NYSE"},
        {"symbol": "BRK-B",  "name": "Berkshire Hathaway Inc. Class B","market": "NYSE"},
        {"symbol": "JNJ",    "name": "Johnson & Johnson",              "market": "NYSE"},
        {"symbol": "PG",     "name": "Procter & Gamble Co.",           "market": "NYSE"},
        {"symbol": "XOM",    "name": "Exxon Mobil Corporation",        "market": "NYSE"},
        {"symbol": "V",      "name": "Visa Inc.",                      "market": "NYSE"},
        {"symbol": "MA",     "name": "Mastercard Inc.",                "market": "NYSE"},
        {"symbol": "HD",     "name": "Home Depot Inc.",                "market": "NYSE"},
        {"symbol": "DIS",    "name": "Walt Disney Company",            "market": "NYSE"},
        {"symbol": "WMT",    "name": "Walmart Inc.",                   "market": "NYSE"},
        {"symbol": "PFE",    "name": "Pfizer Inc.",                    "market": "NYSE"},
        {"symbol": "KO",     "name": "Coca-Cola Company",              "market": "NYSE"},
        {"symbol": "BA",     "name": "The Boeing Company",             "market": "NYSE"},
        # Australian (ASX) - .AX suffix
        {"symbol": "BHP.AX", "name": "BHP Group Ltd",                   "market": "ASX"},
        {"symbol": "CBA.AX", "name": "Commonwealth Bank of Australia",  "market": "ASX"},
        {"symbol": "WBC.AX", "name": "Westpac Banking Corp",           "market": "ASX"},
        {"symbol": "NAB.AX", "name": "National Australia Bank Ltd",    "market": "ASX"},
        {"symbol": "ANZ.AX", "name": "Australia & New Zealand Banking Group", "market": "ASX"},
        {"symbol": "WES.AX", "name": "Wesfarmers Ltd",                 "market": "ASX"},
        {"symbol": "WOW.AX", "name": "Woolworths Group Ltd",           "market": "ASX"},
        {"symbol": "TLS.AX", "name": "Telstra Corp Ltd",               "market": "ASX"},
        {"symbol": "RIO.AX", "name": "Rio Tinto Ltd",                  "market": "ASX"},
        # Indian (NSE) - .NS suffix
        {"symbol": "TCS.NS",       "name": "Tata Consultancy Services",    "market": "NSE"},
        {"symbol": "RELIANCE.NS",  "name": "Reliance Industries Ltd",      "market": "NSE"},
        {"symbol": "INFY.NS",      "name": "Infosys Ltd",                  "market": "NSE"},
        {"symbol": "HDFCBANK.NS",  "name": "HDFC Bank Ltd",                "market": "NSE"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd",               "market": "NSE"},
        {"symbol": "TATAMOTORS.NS","name": "Tata Motors Ltd",              "market": "NSE"},
        {"symbol": "TATASTEEL.NS", "name": "Tata Steel Ltd",               "market": "NSE"},
        {"symbol": "BHARTIARTL.NS","name": "Bharti Airtel Ltd",            "market": "NSE"},
        {"symbol": "MARUTI.NS",    "name": "Maruti Suzuki India Ltd",      "market": "NSE"},
        {"symbol": "WIPRO.NS",     "name": "Wipro Ltd",                    "market": "NSE"},
    ]

###############################################################################
# 2) DOWNLOAD ALL HISTORICAL DATA
###############################################################################
def download_data(ticker):
    """
    Download all available historical data from Yahoo Finance for the given ticker.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    """
    df = yf.download(ticker, start="1900-01-01", end=date.today(), progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).rstrip('_') for col in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    rename_map = {}
    for c in df.columns:
        if "_" in c:
            rename_map[c] = c.split("_")[0]
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()
    df = df[needed].copy()
    df.sort_index(inplace=True)
    return df

###############################################################################
# 3) FEATURE ENGINEERING WITH ADDITIONAL INDICATORS
###############################################################################
def add_technical_indicators(df):
    """
    Add a comprehensive set of technical indicators:
      - Base: RSI, MACD, Bollinger Bands, Stochastic, OBV, MFI, moving averages, and volatility.
      - Additional: ATR, ADX, CCI, Williams %R, and Chaikin Money Flow.
    """
    # Base indicators
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df["MFI"] = ta.volume.MFIIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14).money_flow_index()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Close"].rolling(10).std()
    # Additional indicators
    df["ATR"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    df["ADX"] = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()
    df["CCI"] = ta.trend.CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).cci()
    df["Williams_R"] = ta.momentum.WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], lbp=14).williams_r()
    df["CMF"] = ta.volume.ChaikinMoneyFlowIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=20).chaikin_money_flow()
    df.dropna(inplace=True)
    return df

def add_lagged_features(df, cols_to_lag, lag=2):
    """
    Add lagged versions (1 and 2 days) for selected indicator columns.
    """
    for col in cols_to_lag:
        for i in range(1, lag + 1):
            df[f"{col}_lag{i}"] = df[col].shift(i)
    df.dropna(inplace=True)
    return df

def build_dataset(df):
    """
    Build the dataset for ML:
      - Target: 1 if Close > Open, else 0.
      - Adds lagged features for selected technical indicators.
    """
    df["Target"] = (df["Close"] > df["Open"]).astype(int)
    # Define indicators to add lags for
    lag_cols = ["RSI", "MACD", "MACD_Signal", "Stoch_K", "Stoch_D", "MFI", "ATR", "ADX", "CCI", "Williams_R", "CMF"]
    df = add_lagged_features(df, lag_cols, lag=2)
    # Base feature list
    feats = [
        "RSI", "MACD", "MACD_Signal",
        "BB_High", "BB_Low", "BB_Mid", "BB_Width",
        "Stoch_K", "Stoch_D",
        "OBV", "MFI",
        "MA_10", "MA_50", "Volatility",
        "ATR", "ADX", "CCI", "Williams_R", "CMF"
    ]
    # Include original and lagged versions
    all_feats = []
    for base in feats:
        if base in df.columns:
            all_feats.append(base)
        for i in range(1, 3):
            lagged = f"{base}_lag{i}"
            if lagged in df.columns:
                all_feats.append(lagged)
    df.dropna(inplace=True)
    X = df[all_feats].values
    y = df["Target"].values
    return df, X, y, all_feats

###############################################################################
# 4) TIME-SERIES CROSS-VALIDATION + RANDOM SEARCH FOR RF
###############################################################################
def time_series_rf_tuning(X, y, n_splits=5, n_iter=10):
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"]
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)
    st.write(f"**RandomizedSearchCV** best params: {search.best_params_}")
    st.write(f"**CV best score**: {search.best_score_:.3f}")
    return search.best_estimator_

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model

###############################################################################
# 5) PREDICTIONS: LAST 365 DAYS AND TOMORROW
###############################################################################
def predict_on_recent(df, model, feature_cols):
    if len(df) < 365:
        recent_df = df.copy()
    else:
        recent_df = df.iloc[-365:].copy()
    X_recent = recent_df[feature_cols].values
    recent_df["Prediction"] = model.predict(X_recent)
    recent_df["Prob_Up"] = model.predict_proba(X_recent)[:, 1]
    return recent_df

def predict_tomorrow(latest_row, model, feature_cols):
    tomorrow_df = pd.DataFrame(index=[latest_row.name])
    tomorrow_df["Open"] = latest_row["Close"]  # set tomorrow's open as today's close
    for c in feature_cols:
        tomorrow_df[c] = latest_row[c]
    tomorrow_date = latest_row.name + pd.Timedelta(days=1)
    tomorrow_df["Prediction_Date"] = tomorrow_date.strftime("%Y-%m-%d")
    X_tom = tomorrow_df[feature_cols].values
    pred = model.predict(X_tom)[0]
    prob_up = model.predict_proba(X_tom)[:, 1][0]
    return pred, prob_up, tomorrow_df

###############################################################################
# 5A) BACKTESTING THE STRATEGY ON THE HOLDOUT PERIOD
###############################################################################
def backtest_strategy(df):
    """
    A simple backtest function:
    - For each day in the holdout period (df), if the model's Prediction == 1 (buy signal),
      we assume buying at the Open and selling at the Close.
    - Daily return = (Close - Open) / Open if a buy is made; 0 otherwise.
    - Cumulative return is computed as the cumulative product of (1 + Daily Return).
    """
    df = df.copy()
    df["Daily_Return"] = np.where(df["Prediction"] == 1,
                                  (df["Close"] - df["Open"]) / df["Open"],
                                  0)
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod()
    return df

###############################################################################
# 6) STREAMLIT APP: MAIN FUNCTION
###############################################################################
def main():
    st.set_page_config(page_title="Dynamic Buy/Sell Recommendation", layout="wide")
    st.title("Stock Prediction + Automatic 'Buy' Recommendation")
    st.markdown("""
    **How It Works & User Instructions:**
    
    1. **Stock Selection**: Enter a stock symbol or name (supports US, Australian (.AX), and Indian (.NS) stocks) and select one.
    
    2. **Data & Feature Engineering**: The app downloads all available historical data and computes a comprehensive set of technical indicators – including base indicators (RSI, MACD, Bollinger Bands, Stochastic, OBV, MFI, moving averages, volatility) and additional ones (ATR, ADX, CCI, Williams %R, Chaikin Money Flow) – along with 1‑ and 2‑day lagged features.
    
    3. **Model Training**: It uses TimeSeriesSplit with RandomizedSearchCV to tune a Random Forest model, training on all data except the last 365 days (which serve as a holdout set). Optionally, an XGBoost model is also trained and ensembled.
    
    4. **Predictions & Backtesting**: The model predicts the directional signal for the holdout period and creates a synthetic "tomorrow" prediction. A simple backtest is performed on the holdout period, assuming a trade is made (buy at open, sell at close) only when the model signals "buy".
    
    5. **Buy Recommendation & Trading Guidance**:  
       - **Buy**: If tomorrow's predicted probability is above 50%, the app shows a "Recommended BUY" signal.  
         **Trading Guidance**: When a BUY signal is shown, it is recommended to buy at market open and sell at market close to capture the predicted intraday move. Stop-loss orders may help manage risk.
       - **Sell**: If the signal is not present, avoid buying for a short-term gain.
    
    **Disclaimer**: This tool is for educational purposes only and does not constitute financial advice.
    """)

    # --- 1) Stock Selection ---
    stock_list = get_stock_list()
    search_query = st.text_input("Search by symbol or name:")
    if search_query:
        filtered = [s for s in stock_list if search_query.lower() in s["symbol"].lower() or search_query.lower() in s["name"].lower()]
    else:
        filtered = stock_list
    if not filtered:
        st.warning("No matches found. Try another search.")
        st.stop()
    label_options = [f"{s['symbol']} - {s['name']} ({s['market']})" for s in filtered]
    selected_label = st.selectbox("Select a stock:", label_options)
    selected_stock = next(s for s in filtered if f"{s['symbol']} - {s['name']} ({s['market']})" == selected_label)
    ticker = selected_stock["symbol"]

    # --- 2) Option to Use Ensemble ---
    use_ensemble = st.checkbox("Use XGBoost Ensemble?", value=False)

    # --- 3) Train Model ---
    if st.button("Train Model"):
        with st.spinner(f"Downloading data for {ticker}..."):
            df = download_data(ticker)
        if df.empty:
            st.error("No data returned. Try another ticker.")
            st.stop()
        with st.spinner("Adding technical indicators..."):
            df = add_technical_indicators(df)
        if df.empty:
            st.error("No data after adding indicators.")
            st.stop()
        with st.spinner("Building dataset with lagged features..."):
            dataset_df, X_all, y_all, feats = build_dataset(df)
        if len(dataset_df) < 400:
            st.warning("Not enough data for meaningful training. Try a different ticker or there is insufficient historical data.")
            st.stop()
        st.info("Running TimeSeriesSplit & RandomizedSearchCV... This may take a while.")
        best_rf = time_series_rf_tuning(X_all, y_all, n_splits=5, n_iter=10)
        # Use all data except the last 365 rows as training, and the final 365 as holdout.
        if len(dataset_df) < 365:
            train_df = dataset_df
            test_df = dataset_df
        else:
            train_df = dataset_df.iloc[:-365]
            test_df = dataset_df.iloc[-365:]
        X_train = train_df[feats].values
        y_train = train_df["Target"].values
        X_test = test_df[feats].values
        y_test = test_df["Target"].values
        st.write("Re-training best RF on (all minus last 365 days)...")
        best_rf.fit(X_train, y_train)
        y_pred_test = best_rf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        st.success(f"Final holdout accuracy (last 365 days): {test_acc:.3f}")
        xgb_model = None
        if use_ensemble:
            st.write("Training XGBoost for ensemble...")
            xgb_model = train_xgboost(X_train, y_train)
            rf_proba_test = best_rf.predict_proba(X_test)[:,1]
            xgb_proba_test = xgb_model.predict_proba(X_test)[:,1]
            combined_test = (rf_proba_test + xgb_proba_test) / 2
            ensemble_pred = (combined_test > 0.5).astype(int)
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            st.success(f"Ensemble accuracy (last 365 days): {ensemble_acc:.3f}")
        st.session_state["best_rf"] = best_rf
        st.session_state["rf_acc"] = test_acc
        st.session_state["data_df"] = dataset_df
        st.session_state["feats"] = feats
        st.session_state["ticker"] = ticker
        st.session_state["use_ensemble"] = use_ensemble
        st.session_state["xgb_model"] = xgb_model
    else:
        if "best_rf" not in st.session_state:
            st.warning("Click 'Train Model' to proceed.")
            st.stop()

    # --- 4) Predictions: Last 365 Days and Tomorrow ---
    best_rf = st.session_state["best_rf"]
    feats = st.session_state["feats"]
    data_df = st.session_state["data_df"]
    use_ensemble = st.session_state["use_ensemble"]
    ticker = st.session_state["ticker"]
    xgb_model = st.session_state["xgb_model"]

    st.markdown("---")
    st.subheader(f"Predictions on Last 365 Days for {ticker}")
    recent_pred_df = predict_on_recent(data_df, best_rf, feats)
    if use_ensemble and xgb_model is not None:
        X_recent = recent_pred_df[feats].values
        rf_proba_recent = recent_pred_df["Prob_Up"].values
        xgb_proba_recent = xgb_model.predict_proba(X_recent)[:,1]
        combined_recent = (rf_proba_recent + xgb_proba_recent) / 2
        combined_pred_recent = (combined_recent > 0.5).astype(int)
        recent_pred_df["Ensemble_Prob_Up"] = combined_recent
        recent_pred_df["Ensemble_Pred"] = combined_pred_recent
    st.write("Last 5 predictions:")
    display_cols = ["Open", "Close", "RSI", "MACD", "Prediction", "Prob_Up"]
    if use_ensemble and xgb_model is not None:
        display_cols += ["Ensemble_Pred", "Ensemble_Prob_Up"]
    st.dataframe(recent_pred_df[display_cols].tail(5))
    
    # Interactive Plot: Last 365 Days Close Price with Buy Signals
    fig = px.line(recent_pred_df, x=recent_pred_df.index, y="Close", title=f"{ticker} - Last 365 Days Close Price")
    buy_signals = recent_pred_df[recent_pred_df["Prediction"] == 1]
    fig.add_scatter(x=buy_signals.index, y=buy_signals["Close"],
                    mode="markers", marker_symbol="triangle-up", marker_color="green",
                    name="RF Pred Up")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Predict Tomorrow")
    latest_idx = recent_pred_df.index[-1]
    latest_row = recent_pred_df.loc[latest_idx]
    pred_tom, prob_up, tom_df = predict_tomorrow(latest_row, best_rf, feats)
    tomorrow_str = tom_df["Prediction_Date"].iloc[0]
    final_prob = prob_up
    final_pred = pred_tom
    if use_ensemble and xgb_model is not None:
        xgb_p = xgb_model.predict_proba(tom_df[feats].values)[:,1][0]
        ensemble_p = (prob_up + xgb_p) / 2
        final_prob = ensemble_p
        final_pred = 1 if ensemble_p > 0.5 else 0
    if final_pred == 1:
        st.markdown(
            f"<h4 style='color:green;'>Final Model: Tomorrow ({tomorrow_str}) => CLOSE > OPEN, Prob Up={final_prob:.2f}</h4>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h4 style='color:red;'>Final Model: Tomorrow ({tomorrow_str}) => CLOSE <= OPEN, Prob Up={final_prob:.2f}</h4>",
            unsafe_allow_html=True
        )
    with st.expander("Synthetic 'Tomorrow' Row"):
        st.dataframe(tom_df)
    
    # --- 4A) BACKTESTING THE STRATEGY ---
    st.markdown("---")
    st.subheader("Backtest Strategy on Last 365 Days")
    backtest_df = backtest_strategy(recent_pred_df)
    st.dataframe(backtest_df[["Open", "Close", "Prediction", "Daily_Return", "Cumulative_Return"]].tail(10))
    
    # Interactive Plot: Cumulative Return
    fig2 = px.line(backtest_df, x=backtest_df.index, y="Cumulative_Return", title="Cumulative Return of Strategy on Holdout Period")
    st.plotly_chart(fig2, use_container_width=True)
    
    # --- 5) DYNAMIC BUY RECOMMENDATION & TRADING GUIDANCE ---
    st.markdown("---")
    st.subheader("Buy Recommendation Summary & Trading Guidance")
    recommendation = ""
    if final_prob > 0.5:
        recommendation = "**Recommended BUY** (model predicts >50% chance that tomorrow's close exceeds its open)."
        trading_guidance = ("**Trading Guidance:**\n"
                            "- **Buy** at market open on the day with the BUY signal.\n"
                            "- **Sell** at market close on the same day to capture the predicted intraday move.\n"
                            "- Consider using stop-loss orders to manage risk.")
    else:
        recommendation = "**Not recommended** to buy for a short-term gain based on the model's outlook (<50% chance)."
        trading_guidance = ("**Trading Guidance:**\n"
                            "- When no BUY signal is present, it may be better to wait for clearer signals or additional confirmation.\n"
                            "- Avoid entering a trade solely based on this model if the probability is below 50%.")
    st.markdown(f"**Model-Based Recommendation**: {recommendation}")
    st.markdown(trading_guidance)
    st.info("""
    **Disclaimer**: This is not financial advice. A 'Recommended BUY' and the accompanying trading guidance
    are based solely on the model's prediction for a short-term up move. Real trading involves fees, slippage,
    and risks not captured by this model. Always perform your own research.
    """)

if __name__ == "__main__":
    main()

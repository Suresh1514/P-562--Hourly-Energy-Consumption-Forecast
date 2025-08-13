import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="PJMW 30-Day Power Forecast", layout="wide")

# Title and description
st.title("PJMW Power Supply Forecasting")
st.markdown("""
This application forecasts the next **30 days** of power supply (in MW) for the PJMW region.
""")

# Load and prepare data
@st.cache_data
def load_data():
    # Load the CSV data
    df = pd.read_csv("PJMW_hourly.csv", parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index()  # Ensure chronological order
    return df

# LSTM functions
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def build_lstm_model(look_back, lstm_units=50, dense_units=1):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(1, look_back)))
    model.add(Dense(dense_units))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

try:
    # Load data
    df = load_data()
    
    # Convert to daily average power
    daily_power = df.resample('D').mean()
    daily_power.columns = ['Daily_Power_MW']
    
    # Display data summary
    with st.expander("View Raw Data"):
        st.dataframe(daily_power)
    
    # Model configuration
    st.sidebar.header("Forecast Settings")
    
    # Model selection
    model_type = st.sidebar.radio(
        "Select Forecasting Model",
        ["ARIMA", "SARIMA", "LSTM"],
        index=0
    )
    
    forecast_steps = 30
    
    if model_type in ["ARIMA", "SARIMA"]:
        # Model parameters for ARIMA/SARIMA
        if model_type == "ARIMA":
            st.sidebar.subheader("ARIMA Parameters")
            col1, col2, col3 = st.sidebar.columns(3)
            with col1:
                p = st.selectbox("Autoregressive (p)", [0, 1, 2, 3], index=1)
            with col2:
                d = st.selectbox("Differencing (d)", [0, 1, 2], index=1)
            with col3:
                q = st.selectbox("Moving Average (q)", [0, 1, 2, 3], index=1)
            
            model = ARIMA(daily_power, order=(p, d, q))
            
        else:  # SARIMA
            st.sidebar.subheader("SARIMA Parameters")
            col1, col2, col3 = st.sidebar.columns(3)
            with col1:
                p = st.selectbox("Autoregressive (p)", [0, 1, 2, 3], index=1)
                P = st.selectbox("Seasonal AR (P)", [0, 1, 2], index=1)
            with col2:
                d = st.selectbox("Differencing (d)", [0, 1, 2], index=1)
                D = st.selectbox("Seasonal Diff (D)", [0, 1], index=1)
            with col3:
                q = st.selectbox("Moving Average (q)", [0, 1, 2, 3], index=1)
                Q = st.selectbox("Seasonal MA (Q)", [0, 1, 2], index=1)
            
            s = st.sidebar.selectbox("Seasonality Period", [7, 30], index=0,
                                    help="7 for weekly, 30 for monthly seasonality")
            
            model = SARIMAX(daily_power, order=(p, d, q), seasonal_order=(P, D, Q, s))
        
        # Fit model
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=daily_power.index[-1] + pd.Timedelta(days=1), 
                                      periods=forecast_steps, freq='D')
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecasted_Power_MW': forecast.predicted_mean,
            'Lower_Bound': forecast.conf_int().iloc[:, 0],
            'Upper_Bound': forecast.conf_int().iloc[:, 1]
        }).set_index('Date')
        
        # Model summary
        model_details = model_fit.summary()
        
    else:  # LSTM
        st.sidebar.subheader("LSTM Parameters")
        look_back = st.sidebar.slider("Look Back Period (days)", 1, 90, 30)
        lstm_units = st.sidebar.slider("LSTM Units", 10, 200, 50)
        epochs = st.sidebar.slider("Epochs", 10, 200, 50)
        batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)
        
        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(daily_power.values)
        
        # Create training dataset
        train_size = int(len(dataset) * 0.8)
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        
        # Reshape into X=t and Y=t+1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        
        # Reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        # Build and train LSTM model
        model = build_lstm_model(look_back, lstm_units)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(trainX, trainY, 
                          validation_data=(testX, testY),
                          epochs=epochs, 
                          batch_size=batch_size, 
                          callbacks=[early_stop],
                          verbose=0)
        
        # Generate forecast
        forecast_values = []
        current_batch = dataset[-look_back:].reshape(1, 1, look_back)
        
        for i in range(forecast_steps):
            current_pred = model.predict(current_batch)[0]
            forecast_values.append(current_pred[0])
            current_batch = np.append(current_batch[:, :, 1:], [[current_pred]], axis=2)
        
        # Inverse transform the forecast
        forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))
        
        forecast_index = pd.date_range(start=daily_power.index[-1] + pd.Timedelta(days=1), 
                                      periods=forecast_steps, freq='D')
        
        # Create forecast dataframe (LSTM doesn't provide confidence intervals)
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecasted_Power_MW': forecast_values.flatten(),
            'Lower_Bound': None,
            'Upper_Bound': None
        }).set_index('Date')
        
        # Model summary for LSTM
        model_details = f"""
        LSTM Model Summary:
        - Look Back Period: {look_back} days
        - LSTM Units: {lstm_units}
        - Training Epochs: {epochs}
        - Batch Size: {batch_size}
        - Final Training Loss: {history.history['loss'][-1]:.4f}
        - Final Validation Loss: {history.history['val_loss'][-1]:.4f}
        """
        
        # Plot training history
        st.subheader("LSTM Training History")
        fig_history, ax_history = plt.subplots(figsize=(10, 4))
        ax_history.plot(history.history['loss'], label='Training Loss')
        ax_history.plot(history.history['val_loss'], label='Validation Loss')
        ax_history.set_title('Model Loss During Training')
        ax_history.set_ylabel('Loss (MSE)')
        ax_history.set_xlabel('Epoch')
        ax_history.legend()
        st.pyplot(fig_history)
    
    # Visualization
    st.subheader("30-Day Power Supply Forecast")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data (last 90 days for context)
    historical = daily_power.last('90D')
    ax.plot(historical.index, historical['Daily_Power_MW'], 
            label='Historical Power', color='blue', linewidth=2)
    
    # Plot forecast
    ax.plot(forecast_df.index, forecast_df['Forecasted_Power_MW'], 
            label='Forecast', color='red', linestyle='--', linewidth=2)
    
    # Confidence interval (only for ARIMA/SARIMA)
    if model_type in ["ARIMA", "SARIMA"]:
        ax.fill_between(forecast_df.index, 
                       forecast_df['Lower_Bound'], 
                       forecast_df['Upper_Bound'], 
                       color='orange', alpha=0.2, label='Confidence Interval')
    
    # Formatting
    ax.set_title(f'PJMW 30-Day Power Supply Forecast ({model_type})', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Power (MW)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Display forecast table
    st.subheader("Detailed Forecast Data")
    forecast_display = forecast_df.copy()
    forecast_display.index = forecast_display.index.strftime('%Y-%m-%d')
    st.dataframe(forecast_display.style.format("{:.2f}"))
    
    # Download button
    csv = forecast_df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download 30-Day Forecast (CSV)",
        data=csv,
        file_name='pjwm_30day_power_forecast.csv',
        mime='text/csv',
        help="Download the forecast data as a CSV file"
    )
    
    # Model summary
    with st.expander("Model Details"):
        st.text(model_details)
    
except Exception as e:
    st.error(f"Error loading or processing data: {str(e)}")
    st.info("Please ensure the data file 'PJMW_hourly.csv' exists with columns 'Datetime' and 'PJMW_MW'")

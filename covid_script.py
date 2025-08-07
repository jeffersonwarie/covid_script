import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="COVID-19 Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)

# Title and introduction
st.title("🦠 COVID-19 Time Series Forecasting with Deep Learning")
st.markdown("""
This app demonstrates a sophisticated time series forecasting pipeline that combines:
- **Autoencoder-based denoising** to remove outliers and noise
- **Triple exponential smoothing** to capture trends and seasonality  
- **LSTM neural networks** for sequence-to-sequence prediction

Upload your COVID-19 data or use the sample dataset to see how each component contributes to the final forecast.
""")

# Sidebar for parameters
st.sidebar.header("🔧 Model Parameters")
predict_ahead = st.sidebar.slider("Prediction horizon (days)", 1, 14, 7)
train_split = st.sidebar.slider("Training data percentage", 0.6, 0.9, 0.75)
lstm_epochs = st.sidebar.slider("LSTM training epochs", 50, 300, 150)
hidden_units = st.sidebar.slider("LSTM hidden units", 50, 200, 100)

# Data upload section
st.header("📊 Data Input")
uploaded_file = st.file_uploader("Upload COVID-19 CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")
else:
    # Create sample data for demonstration
    st.info("No file uploaded. Using sample COVID-19 data for demonstration.")
    np.random.seed(42)
    dates = pd.date_range('2020-03-01', periods=500, freq='D')
    
    # Create realistic COVID-19 case pattern with trends, seasonality, and noise
    trend = np.exp(np.linspace(0, 2, 500)) * 100
    seasonal = 50 * np.sin(2 * np.pi * np.arange(500) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 50, 500)
    outliers = np.zeros(500)
    outliers[np.random.choice(500, 20)] = np.random.normal(0, 300, 20)  # Add some outliers
    
    cases = trend + seasonal + noise + outliers
    cases = np.maximum(cases, 0)  # Ensure no negative cases
    
    data = pd.DataFrame({
        'date': dates,
        'cases': cases.astype(int)
    })

# Display basic data info
st.subheader("Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", len(data))
    st.metric("Date Range", f"{len(data)} days")
with col2:
    st.metric("Average Daily Cases", f"{data['cases'].mean():.0f}")
    st.metric("Peak Cases", f"{data['cases'].max():,}")

# Show raw data plot
fig_raw = px.line(data, x='date', y='cases', title='Raw COVID-19 Cases Over Time')
fig_raw.update_layout(showlegend=False)
st.plotly_chart(fig_raw, use_container_width=True)

# Start the processing pipeline
st.header("🔄 Data Processing Pipeline")

# Step 1: Data Preprocessing
st.subheader("Step 1: Data Standardization")
st.markdown("""
**Why standardize?** Neural networks work better when input features have similar scales. 
Z-score normalization transforms data to have mean=0 and std=1.
""")

raw_cases = data['cases'].values
scaler = StandardScaler()
standardized_data = scaler.fit_transform(raw_cases.reshape(-1, 1)).flatten()

col1, col2 = st.columns(2)
with col1:
    fig_before = px.histogram(raw_cases, title="Before Standardization", 
                              labels={'value': 'Cases', 'count': 'Frequency'})
    st.plotly_chart(fig_before, use_container_width=True)
with col2:
    fig_after = px.histogram(standardized_data, title="After Standardization",
                             labels={'value': 'Standardized Cases', 'count': 'Frequency'})
    st.plotly_chart(fig_after, use_container_width=True)

# Step 2: Autoencoder Denoising
st.subheader("Step 2: Autoencoder-Based Denoising")
st.markdown("""
**Purpose:** Remove outliers and noise by training an autoencoder to reconstruct 'normal' patterns.
Points with high reconstruction error are likely outliers and get replaced with reconstructed values.
""")

# Build and train autoencoder
@st.cache_data
def train_autoencoder(data, hidden_size=10, epochs=100):
    # Reshape data for autoencoder (samples, features)
    X = data.reshape(-1, 1)
    
    # Build autoencoder
    input_layer = keras.Input(shape=(1,))
    encoded = layers.Dense(hidden_size, activation='relu')(input_layer)
    decoded = layers.Dense(1, activation='linear')(encoded)
    
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train autoencoder
    history = autoencoder.fit(X, X, epochs=epochs, verbose=0, validation_split=0.2)
    
    return autoencoder, history

with st.spinner("Training autoencoder..."):
    autoencoder, ae_history = train_autoencoder(standardized_data)

# Get reconstructions and identify outliers
reconstructed = autoencoder.predict(standardized_data.reshape(-1, 1), verbose=0).flatten()
reconstruction_error = np.abs(standardized_data - reconstructed)
threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)

# Apply denoising
denoised_data = standardized_data.copy()
outlier_mask = reconstruction_error > threshold
denoised_data[outlier_mask] = reconstructed[outlier_mask]

# Visualize denoising effect
fig_denoise = make_subplots(rows=2, cols=1, 
                           subplot_titles=['Reconstruction Error', 'Before vs After Denoising'])

# Plot reconstruction error
fig_denoise.add_trace(
    go.Scatter(y=reconstruction_error, name='Reconstruction Error', line=dict(color='red')),
    row=1, col=1
)
fig_denoise.add_hline(y=threshold, line_dash="dash", line_color="black", 
                     annotation_text="Outlier Threshold", row=1, col=1)

# Plot before/after
fig_denoise.add_trace(
    go.Scatter(y=standardized_data, name='Original', opacity=0.7),
    row=2, col=1
)
fig_denoise.add_trace(
    go.Scatter(y=denoised_data, name='Denoised', line=dict(color='green')),
    row=2, col=1
)

fig_denoise.update_layout(height=600, title_text="Autoencoder Denoising Results")
st.plotly_chart(fig_denoise, use_container_width=True)

st.info(f"Detected and corrected {np.sum(outlier_mask)} outliers ({100*np.sum(outlier_mask)/len(data):.1f}% of data)")

# Step 3: Triple Exponential Smoothing (Holt-Winters)
st.subheader("Step 3: Triple Exponential Smoothing (Holt-Winters)")
st.markdown("""
**Components:**
- **Level (α):** Base value, responds to recent changes
- **Trend (β):** Direction and rate of change  
- **Seasonality (γ):** Repeating patterns (weekly cycles for COVID data)
""")

def holt_winters_smoothing(data, alpha=0.8, beta=0.2, gamma=0.5, season_length=7):
    """Apply Triple Exponential Smoothing"""
    n = len(data)
    
    # Initialize components
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    
    # Initialize seasonal components
    for i in range(season_length):
        seasonal[i] = data[i] - np.mean(data[:season_length])
    
    # Initialize level and trend
    level[0] = data[0]
    trend[0] = data[1] - data[0] if n > 1 else 0
    
    # Apply smoothing
    for i in range(1, n):
        season_idx = i % season_length
        
        # Update level
        level[i] = alpha * (data[i] - seasonal[season_idx]) + (1 - alpha) * (level[i-1] + trend[i-1])
        
        # Update trend  
        trend[i] = beta * (level[i] - level[i-1]) + (1 - beta) * trend[i-1]
        
        # Update seasonality
        seasonal[season_idx] = gamma * (data[i] - level[i]) + (1 - gamma) * seasonal[season_idx]
    
    # Combine components
    smoothed = level + seasonal
    
    return smoothed, level, trend, seasonal

smoothed_data, level, trend, seasonal = holt_winters_smoothing(denoised_data)

# Visualize smoothing components
fig_components = make_subplots(rows=4, cols=1, 
                              subplot_titles=['Original vs Smoothed', 'Level', 'Trend', 'Seasonal'])

fig_components.add_trace(go.Scatter(y=denoised_data, name='Denoised', opacity=0.5), row=1, col=1)
fig_components.add_trace(go.Scatter(y=smoothed_data, name='Smoothed', line=dict(color='blue')), row=1, col=1)
fig_components.add_trace(go.Scatter(y=level, name='Level', line=dict(color='green')), row=2, col=1)
fig_components.add_trace(go.Scatter(y=trend, name='Trend', line=dict(color='orange')), row=3, col=1)
fig_components.add_trace(go.Scatter(y=seasonal, name='Seasonal', line=dict(color='purple')), row=4, col=1)

fig_components.update_layout(height=800, title_text="Holt-Winters Decomposition")
st.plotly_chart(fig_components, use_container_width=True)

# Step 4: LSTM Prediction
st.subheader("Step 4: LSTM Neural Network Training")
st.markdown("""
**LSTM (Long Short-Term Memory)** networks are designed to learn long-term dependencies in sequential data.
They use gates to control information flow and can remember important patterns over time.
""")

# Prepare data for LSTM
def create_sequences(data, sequence_length):
    """Create input-output sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Split data
split_idx = int(train_split * len(smoothed_data))
train_data = smoothed_data[:split_idx]
test_data = smoothed_data[split_idx:]

# Create sequences
X_train, y_train = create_sequences(train_data, predict_ahead)
X_test, y_test = create_sequences(test_data, predict_ahead)

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
@st.cache_resource
def build_and_train_lstm(X_train, y_train, hidden_units, epochs):
    model = keras.Sequential([
        layers.LSTM(hidden_units, return_sequences=False, input_shape=(X_train.shape[1], 1)),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class StreamlitCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.6f}')
    
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, 
                       validation_split=0.2, callbacks=[StreamlitCallback()])
    
    progress_bar.empty()
    status_text.empty()
    
    return model, history

with st.spinner("Training LSTM model..."):
    lstm_model, lstm_history = build_and_train_lstm(X_train, y_train, hidden_units, lstm_epochs)

# Make predictions
train_predictions = lstm_model.predict(X_train, verbose=0)
test_predictions = lstm_model.predict(X_test, verbose=0)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_mape = mean_absolute_percentage_error(y_test, test_predictions) * 100

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training RMSE", f"{train_rmse:.4f}")
with col2:
    st.metric("Test RMSE", f"{test_rmse:.4f}")
with col3:
    st.metric("Test MAPE", f"{test_mape:.2f}%")

# Visualize results
st.subheader("📈 Prediction Results")

# Create comprehensive results plot
fig_results = go.Figure()

# Plot training data
train_indices = range(predict_ahead, len(train_data))
fig_results.add_trace(go.Scatter(
    x=train_indices,
    y=y_train,
    mode='lines',
    name='Training Actual',
    line=dict(color='blue', width=1)
))

fig_results.add_trace(go.Scatter(
    x=train_indices,
    y=train_predictions.flatten(),
    mode='lines',
    name='Training Predicted',
    line=dict(color='lightblue', width=1)
))

# Plot test data
test_indices = range(split_idx + predict_ahead, split_idx + predict_ahead + len(y_test))
fig_results.add_trace(go.Scatter(
    x=test_indices,
    y=y_test,
    mode='lines',
    name='Test Actual',
    line=dict(color='red', width=2)
))

fig_results.add_trace(go.Scatter(
    x=test_indices,
    y=test_predictions.flatten(),
    mode='lines',
    name='Test Predicted',
    line=dict(color='orange', width=2)
))

# Add vertical line to separate train/test
fig_results.add_vline(x=split_idx, line_dash="dash", line_color="gray", 
                     annotation_text="Train/Test Split")

fig_results.update_layout(
    title="LSTM Prediction Results (Standardized Scale)",
    xaxis_title="Time Index",
    yaxis_title="Standardized Cases",
    hovermode='x unified'
)

st.plotly_chart(fig_results, use_container_width=True)

# Convert back to original scale for interpretation
st.subheader("🔄 Results in Original Scale")

# Transform predictions back to original scale
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_pred_original = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()

# Calculate metrics in original scale
rmse_original = np.sqrt(mean_squared_error(y_test_original, test_pred_original))
mape_original = mean_absolute_percentage_error(y_test_original, test_pred_original) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE (Original Scale)", f"{rmse_original:,.0f} cases")
with col2:
    st.metric("MAPE (Original Scale)", f"{mape_original:.2f}%")

# Plot in original scale
fig_original = go.Figure()

fig_original.add_trace(go.Scatter(
    x=test_indices,
    y=y_test_original,
    mode='lines',
    name='Actual Cases',
    line=dict(color='red', width=2)
))

fig_original.add_trace(go.Scatter(
    x=test_indices,
    y=test_pred_original,
    mode='lines',
    name='Predicted Cases',
    line=dict(color='orange', width=2)
))

fig_original.update_layout(
    title="Final Predictions in Original Scale",
    xaxis_title="Time Index",
    yaxis_title="Daily Cases",
    hovermode='x unified'
)

st.plotly_chart(fig_original, use_container_width=True)

# Model interpretation
st.header("🧠 Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Progress")
    train_loss_fig = px.line(
        x=range(len(lstm_history.history['loss'])),
        y=lstm_history.history['loss'],
        title="Training Loss Over Time",
        labels={'x': 'Epoch', 'y': 'Loss'}
    )
    if 'val_loss' in lstm_history.history:
        train_loss_fig.add_scatter(
            x=range(len(lstm_history.history['val_loss'])),
            y=lstm_history.history['val_loss'],
            name='Validation Loss'
        )
    st.plotly_chart(train_loss_fig, use_container_width=True)

with col2:
    st.subheader("Prediction Error Analysis")
    errors = y_test_original - test_pred_original
    error_fig = px.histogram(errors, title="Prediction Error Distribution",
                           labels={'value': 'Prediction Error (cases)', 'count': 'Frequency'})
    st.plotly_chart(error_fig, use_container_width=True)

# Summary and insights
st.header("📋 Summary & Key Insights")

st.markdown(f"""
### Model Performance Summary

**Pipeline Components:**
1. **Autoencoder Denoising**: Corrected {np.sum(outlier_mask)} outliers ({100*np.sum(outlier_mask)/len(data):.1f}% of data)
2. **Holt-Winters Smoothing**: Captured trends and weekly seasonality patterns
3. **LSTM Network**: Learned complex temporal dependencies with {hidden_units} hidden units

**Final Metrics:**
- **RMSE**: {rmse_original:,.0f} daily cases
- **MAPE**: {mape_original:.2f}% (lower is better)
- **Prediction Horizon**: {predict_ahead} days ahead

### Educational Takeaways

**Why This Approach Works:**
- **Denoising** removes data quality issues that could mislead the model
- **Smoothing** helps the LSTM focus on true patterns rather than noise
- **Sequential Learning** captures complex temporal dependencies that traditional methods miss

**When to Use This Method:**
- Time series with clear trends and seasonal patterns
- Noisy data that benefits from preprocessing
- Need for multi-step ahead forecasting
- When interpretability of components is valuable

**Limitations to Consider:**
- Requires sufficient training data (typically 100+ observations)
- Performance degrades for predictions far into the future
- May not adapt quickly to sudden pattern changes
- Computational complexity higher than simple methods
""")

# Download section
st.header("💾 Export Results")
if st.button("Generate Downloadable Report"):
    results_df = pd.DataFrame({
        'time_index': test_indices,
        'actual_cases': y_test_original,
        'predicted_cases': test_pred_original,
        'prediction_error': y_test_original - test_pred_original
    })
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="covid_predictions.csv",
        mime="text/csv"
    )
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.dates as mdates

# Load the datasets
sp500 = pd.read_csv('SP500 (2).csv')
fedfunds = pd.read_csv('FEDFUNDS (2).csv')
djia = pd.read_csv('DJIA (2).csv')
nasdaqcom = pd.read_csv('NASDAQCOM (2).csv')
nasdaq100 = pd.read_csv('NASDAQ100 (1).csv')
gdpc1 = pd.read_csv('GDPC1.csv')
unrate = pd.read_csv('UNRATE.csv')
dcoilwtico = pd.read_csv('DCOILWTICO.csv')
dgs10 = pd.read_csv('DGS10.csv')
m2v = pd.read_csv('M2V.csv')
wm2ns = pd.read_csv('WM2NS.csv')

# Available datasets for selection
datasets_dict = {
    'SP500': sp500,
    'FEDFUNDS': fedfunds,
    'DJIA': djia,
    'NASDAQCOM': nasdaqcom,
    'NASDAQ100': nasdaq100,
    'GDPC1': gdpc1,
    'UNRATE': unrate,
    'DCOILWTICO': dcoilwtico,
    'DGS10': dgs10,
    'M2V': m2v,
    'WM2NS': wm2ns
}

# Ask the user to select datasets
print("Available datasets for analysis: ", list(datasets_dict.keys()))
selected_datasets = input("Enter datasets you want to include in the analysis, separated by commas (or type 'all' to select all datasets): ")

# Check if user input is 'all' and select all datasets if true
if selected_datasets.lower() == 'all':
    selected_datasets = list(datasets_dict.keys())
else:
    selected_datasets = selected_datasets.split(', ')

# Filter and merge the selected datasets
merged_data = datasets_dict[selected_datasets[0]]
for dataset_name in selected_datasets[1:]:
    merged_data = pd.merge(merged_data, datasets_dict[dataset_name], on='DATE', how='outer')

# Rename the columns and standardize date column
merged_data.rename(columns={'DATE': 'Date'}, inplace=True)
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Filter data between 1960 and 2024
mask = (merged_data['Date'] >= '1960-01-01') & (merged_data['Date'] <= '2024-12-31')
filtered_data = merged_data.loc[mask].reset_index(drop=True)

# Convert the necessary columns to numeric, forcing errors to NaN
for column in filtered_data.columns[1:]:
    filtered_data[column] = pd.to_numeric(filtered_data[column], errors='coerce')

# Fill missing values using forward fill
filtered_data.fillna(method='ffill', inplace=True)

# Drop any remaining rows with NaN values after filling
filtered_data.dropna(inplace=True)

# Feature Selection: Select key columns (selected datasets only)
selected_columns = selected_datasets
filtered_data = filtered_data[['Date'] + selected_columns]

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(filtered_data[selected_columns])

# Create the training dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step):
        X.append(data[i:(i+time_step), :])  # Include all features
        y.append(data[i + time_step, 0])    # Predicting the first selected dataset (e.g. SP500)
    return np.array(X), np.array(y)

# Create the dataset (Ensure X has 3D shape: (samples, time steps, features))
X, y = create_dataset(scaled_data, time_step=60)
print("Shape of X:", X.shape)  # Debugging
print("Shape of y:", y.shape)

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Check if data is valid for training
if X_train.size == 0 or X_test.size == 0:
    print("Error: Not enough data to train the model. Please check the dataset selection and merging process.")
else:
    # Model creation: LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))  # Increase Dropout to avoid overfitting
    model.add(LSTM(50))
    model.add(Dropout(0.3))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))

    # Make predictions
    predicted_prices = model.predict(X_test)

    # Inverse transform predictions to original scale
    pad_array = np.zeros((predicted_prices.shape[0], len(selected_columns)-1))
    predicted_full = np.concatenate((predicted_prices, pad_array), axis=1)
    predicted_prices_actual = scaler.inverse_transform(predicted_full)[:, 0]

    # Prepare future predictions for a custom number of weeks
    def predict_future_data(model, last_sequence, future_steps=8):  # Default set to 8 weeks
        future_predictions = []
        for _ in range(future_steps):
            prediction = model.predict(np.expand_dims(last_sequence, axis=0))
            future_predictions.append(prediction[0, 0])
            # Create the next input sequence by padding prediction to match features
            padded_prediction = np.zeros(last_sequence.shape[1])
            padded_prediction[0] = prediction[0, 0]  # Insert predicted value for first feature (SP500)
            next_sequence = np.concatenate((last_sequence[1:], np.expand_dims(padded_prediction, axis=0)), axis=0)
            last_sequence = next_sequence
        return np.array(future_predictions)

    # Set prediction period (in weeks)
    prediction_period_weeks = int(input("Enter the number of weeks for the prediction period: "))

    # Use the last available data as the seed for future predictions
    last_sequence = X_test[-1]
    future_predictions = predict_future_data(model, last_sequence, future_steps=prediction_period_weeks)

    # Inverse transform future predictions
    future_pad_array = np.zeros((future_predictions.shape[0], len(selected_columns)-1))
    future_predictions_full = np.concatenate((future_predictions.reshape(-1, 1), future_pad_array), axis=1)
    future_predictions_actual = scaler.inverse_transform(future_predictions_full)[:, 0]

    # Create a DataFrame to display the predicted data
    prediction_df = pd.DataFrame({
        'Date': pd.date_range(start=filtered_data['Date'].iloc[-1] + pd.DateOffset(weeks=1), periods=prediction_period_weeks, freq='W'),
        'Predicted Prices': future_predictions_actual
    })

    # Display the predictions in tabular format
    print(prediction_df)

    # Save the predicted data to a CSV file
    prediction_df.to_csv('predicted_data.csv', index=False)
    print("Predicted data saved to 'predicted_data.csv'.")

    # Plotting the results
    plt.figure(figsize=(14, 7))
    # Plot actual SP500 prices
    plt.plot(filtered_data['Date'], filtered_data[selected_columns[0]], label='Actual Prices', color='blue')

    # Plot predicted prices (on test data)
    test_dates = filtered_data['Date'].iloc[len(filtered_data) - len(y_test):]
    plt.plot(test_dates, predicted_prices_actual, label='Predicted Prices', color='red')

    # Plot future predictions (for the selected prediction period)
    plt.plot(prediction_df['Date'], prediction_df['Predicted Prices'], label=f'Future Predictions ({prediction_period_weeks} weeks)', color='green')

    # Formatting
    plt.title(f'{selected_columns[0]} Price Prediction and Future Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Adjust x-axis for better readability
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())  # Major ticks every week
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    plt.show()

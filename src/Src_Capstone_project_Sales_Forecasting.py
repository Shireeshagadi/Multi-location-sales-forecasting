# %%
#  Preliminary analysis
import pandas as pd

# Load datasets
items = pd.read_csv('c:/data/items.csv')
restaurants = pd.read_csv('c:/data/resturants.csv')
sales = pd.read_csv('c:/data/sales.csv')
merged_df = sales.merge(items, left_on='item_id', right_on='id', how='inner')
final_df = merged_df.merge(restaurants, left_on='store_id', right_on='id', how='inner')
final_df = final_df[['date', 'item_id', 'item_count', 'price', 'name_x', 'kcal', 'id_y', 'name_y']]
final_df = final_df.rename(columns={'name_x': 'item names', 'item_id': 'item id', 'item_count': 'item count', 'kcal': 'kcal values', 'id_y': 'store id', 'name_y': 'store name'})

# Aggregate Sales Data by Date

# Create a new column for total daily sales
daily_sales = final_df.groupby('date')['price'].sum().reset_index()
daily_sales.rename(columns={'price': 'total_sales'}, inplace=True)

# Merge total_sales back to the original DataFrame
final_df = final_df.merge(daily_sales, on='date', how='left')

# Ensure the date column matches the format in data frame
final_df['date'] = pd.to_datetime(final_df['date'], format='%d-%m-%Y')
final_df['date'] = pd.to_datetime(final_df['date'], dayfirst=True)  # Assumes day is the first part of the date
#sales fluctuate across different days of the week

final_df['date'] = pd.to_datetime(final_df['date'])

# Add a column for day of the week
final_df['day_of_week'] = final_df['date'].dt.day_name()



# %%
# Aggregate Sales by Day of the Week
# Group sales by day of the week
day_wise_sales = final_df.groupby('day_of_week').agg({'price': 'sum'}).reset_index()
day_wise_sales.rename(columns={'price': 'total_sales'}, inplace=True)

# Sort the days of the week 
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_wise_sales['day_of_week'] = pd.Categorical(day_wise_sales['day_of_week'], categories=ordered_days, ordered=True)
day_wise_sales = day_wise_sales.sort_values('day_of_week')

# Extract the Month
# Ensure 'date' is in datetime format
final_df['date'] = pd.to_datetime(final_df['date'])

# Extract the month name
final_df['month'] = final_df['date'].dt.month_name()

# Extract the Month
# Ensure 'date' is in datetime format
final_df['date'] = pd.to_datetime(final_df['date'])

# Extract the month name
final_df['month'] = final_df['date'].dt.month_name()



# %%
# Aggregate Sales Data by Month
# Aggregate sales by month
month_wise_sales = final_df.groupby('month').agg({'price': 'sum'}).reset_index()
month_wise_sales.rename(columns={'price': 'total_sales'}, inplace=True)

# Sort months in chronological order
ordered_months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
month_wise_sales['month'] = pd.Categorical(month_wise_sales['month'], categories=ordered_months, ordered=True)
month_wise_sales = month_wise_sales.sort_values('month')



# %%
# Extract Quarter Information
# Ensure 'date' is in datetime format
final_df['date'] = pd.to_datetime(final_df['date'])

# Extract the quarter
final_df['quarter'] = final_df['date'].dt.quarter

# Aggregate Sales by Quarter
# Group sales by quarter
quarterly_sales = final_df.groupby('quarter').agg({'price': 'mean'}).reset_index()
quarterly_sales.rename(columns={'price': 'average_sales'}, inplace=True)


# %%
# Group total sales by restaurant
restaurant_sales = final_df.groupby('store name').agg({'price': 'sum'}).reset_index()
restaurant_sales.rename(columns={'price': 'total_sales'}, inplace=True)

# Identify the restaurant with the most sales
top_restaurant = restaurant_sales.loc[restaurant_sales['total_sales'].idxmax()]



# %%
# Analyze Sales by Year
# Extract year from date
final_df['year'] = final_df['date'].dt.year

# Group sales by restaurant and year
yearly_sales = final_df.groupby(['store name', 'year']).agg({'price': 'sum'}).reset_index()
yearly_sales.rename(columns={'price': 'yearly_sales'}, inplace=True)



# %%
# Analyze Sales by Month
# Extract month from date
final_df['month'] = final_df['date'].dt.month_name()

# Group sales by restaurant and month
monthly_sales = final_df.groupby(['store name', 'month']).agg({'price': 'sum'}).reset_index()
monthly_sales.rename(columns={'price': 'monthly_sales'}, inplace=True)



# %%
# Analyze Sales by Day
# Group sales by restaurant and day
daily_sales = final_df.groupby(['store name', 'date']).agg({'price': 'sum'}).reset_index()
daily_sales.rename(columns={'price': 'daily_sales'}, inplace=True)



# %%
# Machine learning
# Identify the Most Popular Items Overall
# Group by item names to calculate total count sold
popular_items = final_df.groupby('item names').agg({'item count': 'sum'}).reset_index()
popular_items.rename(columns={'item count': 'total_count'}, inplace=True)

# Sort by total count in descending order
popular_items = popular_items.sort_values('total_count', ascending=False)



# %%
# Identify Stores Where Popular Items are Sold
# Find stores where popular items are sold
stores_selling_popular_items = final_df[final_df['item names'].isin(popular_items['item names'].head(5))]
popular_items_with_stores = stores_selling_popular_items[['item names', 'store name']].drop_duplicates()



# %%
# Identify the Most Popular Item at Each Store
# Group by store name and item names to calculate total count
store_item_sales = final_df.groupby(['store name', 'item names']).agg({'item count': 'sum'}).reset_index()
store_item_sales.rename(columns={'item count': 'total_count'}, inplace=True)

# Find the most popular item for each store
most_popular_item_per_store = store_item_sales.loc[store_item_sales.groupby('store name')['total_count'].idxmax()]



# %%
# Identify the Store with the Highest Sales Volume
# Group by store name and calculate total item count
store_sales_volume = final_df.groupby('store name').agg({'item count': 'sum'}).reset_index()
store_sales_volume.rename(columns={'item count': 'total_volume'}, inplace=True)

# Find the store with the highest sales volume
top_volume_store = store_sales_volume.loc[store_sales_volume['total_volume'].idxmax()]



# %%
# Calculate Average Money Made per Day
# Group by store name and date to calculate daily sales
daily_sales = final_df.groupby(['store name', 'date']).agg({'price': 'sum'}).reset_index()
daily_sales.rename(columns={'price': 'daily_revenue'}, inplace=True)

# Calculate the average daily revenue for each store
store_avg_daily_revenue = daily_sales.groupby('store name').agg({'daily_revenue': 'mean'}).reset_index()
store_avg_daily_revenue.rename(columns={'daily_revenue': 'avg_daily_revenue'}, inplace=True)

# Find the store making the most money per day
top_revenue_store = store_avg_daily_revenue.loc[store_avg_daily_revenue['avg_daily_revenue'].idxmax()]

# Group Items by Restaurant and Find the Most Expensive Item
# Group by store name and item names to find the most expensive item at each restaurant
expensive_items = final_df.loc[final_df.groupby('store name')['price'].idxmax(), ['store name', 'item names', 'price', 'kcal values']]


# %%
 # Forecasting using machine learning algorithms
 #  Build and compare linear regression, random forest, and XGBoost models for predictions
 # Generate Features
 # Generate time-based features
final_df['date'] = pd.to_datetime(final_df['date'])  # Ensure datetime format
final_df['day_of_week'] = final_df['date'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)
final_df['quarter'] = final_df['date'].dt.quarter       # Quarter of the year
final_df['month'] = final_df['date'].dt.month           # Month
final_df['year'] = final_df['date'].dt.year             # Year
final_df['day_of_month'] = final_df['date'].dt.day      # Day of the month



# %%

# Split the Data into Training and Testing Sets
train_data = final_df[final_df['date'] < '2021-07-01']
test_data = final_df[final_df['date'] >= '2021-07-01']

# Separate predictors and target variable (e.g., item_count)
X_train = train_data[['day_of_week', 'quarter', 'month', 'year', 'day_of_month']]
y_train = train_data['item count']
X_test = test_data[['day_of_week', 'quarter', 'month', 'year', 'day_of_month']]
y_test = test_data['item count']



# %%
# Train and Test Models
# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and compute RMSE
lr_predictions = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))


# %%
# Random Forest
from sklearn.ensemble import RandomForestRegressor

# Train random forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Predict and compute RMSE
rf_predictions = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))


# %%
# XGBoost
from xgboost import XGBRegressor

# Train XGBoost model
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# Predict and compute RMSE
xgb_predictions = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

# Compare Model Performance
best_model = min([lr_rmse, rf_rmse, xgb_rmse], key=lambda x: x)

# %%
# Generate Future Dates and Features
# Generate future dates for the next year
future_dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

# Create features for future dates
future_features = pd.DataFrame({
    'day_of_week': future_dates.dayofweek,
    'quarter': future_dates.quarter,
    'month': future_dates.month,
    'year': future_dates.year,
    'day_of_month': future_dates.day
})



# %%
# Use the XGBoost Model to Make Predictions
# Make predictions for the future dates
future_predictions = xgb_model.predict(future_features)

# Combine dates with predictions into a DataFrame
future_forecasts = pd.DataFrame({
    'date': future_dates,
    'predicted_sales': future_predictions
})



# %%
#  Deep learning
#  Forecasting using deep learning algorithms
# Use sales amount (price) for predictions
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Select sales amount and date
sales_data = final_df[['date', 'price']].groupby('date').sum().reset_index()

# Ensure the data is sorted by date
sales_data.sort_values('date', inplace=True)

# Normalize the sales data (required for LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
sales_data['normalized_price'] = scaler.fit_transform(sales_data[['price']])



# %%
# Split data into training and testing sets
train_size = int(len(sales_data) * 0.8)
train_data = sales_data.iloc[:train_size]
test_data = sales_data.iloc[train_size:]



# %%
# Generate synthetic data for 12 months if no test data exists
synthetic_dates = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')

# Generate random sales prices for the synthetic data
synthetic_sales = np.random.uniform(low=price_min, high=price_max, size=len(synthetic_dates))

# Create a DataFrame with synthetic data
synthetic_data = pd.DataFrame({'date': synthetic_dates, 'price': synthetic_sales})


# %%
from sklearn.preprocessing import MinMaxScaler

# Normalize synthetic sales data

scaler = MinMaxScaler(feature_range=(0, 1))
synthetic_data['normalized_price'] = scaler.fit_transform(synthetic_data[['price']])

# Check the updated normalized values
combined_data = pd.concat([final_df, synthetic_data], ignore_index=True)
combined_data.sort_values('date', inplace=True)

# %%
# Create Input Sequences for LSTM
def create_sequences(data, time_steps=7):
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:(i + time_steps)])
        labels.append(data[i + time_steps])
    return np.array(sequences), np.array(labels)

# Prepare sequences
time_steps = 7
train_sequences, train_labels = create_sequences(train_data[['normalized_price']].values, time_steps)
test_sequences, test_labels = create_sequences(test_data[['normalized_price']].values, time_steps)

# Reshape for LSTM input

train_sequences = train_sequences.reshape(869, 7, 1)  # Reshape to (samples, timesteps, features)
test_sequences = test_sequences.reshape(213, 7, 1)



# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# Build the LSTM model
model = Sequential([Input(shape=(7,1)),
    LSTM(50, activation='relu', return_sequences=False), # 50 units, input shape: (7 timesteps, 1 feature)
    Dense(1)  # Dense layer for output
])

# Compile the model
model.compile(optimizer='adam', loss='mse')
model.summary()

# %%
# Train the Model
history = model.fit(
    train_sequences, train_labels,
    validation_data=(test_sequences, test_labels),
    epochs=50,  # Number of epochs
    batch_size=32  # Mini-batch size
)



# %%
# Make Predictions
predictions = model.predict(test_sequences)
# Compare predictions with true labels
for i in range(len(test_labels)):
    # Extract a scalar value correctly from an array efore converting to float
    true_label = float(test_labels[i].item())  # Use .item() to get the scalar value from the array
    predicted_label = float(predictions[i][0].item())  # For predictions, extract scalar value properl
    

# %%
# Assuming `test_labels` are the true values and `predictions` are the predicted values
def calculate_mape(true_values, predicted_values):
    # Avoid division by zero
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    non_zero_indices = true_values != 0

    # MAPE calculation
    mape = np.mean(np.abs((true_values[non_zero_indices] - predicted_values[non_zero_indices]) / true_values[non_zero_indices])) * 100
    return mape

# Example usage
mape = calculate_mape(test_labels, predictions)



# %%
# Initialize future predictions
future_predictions = []

# Use the last sequence for forecasting
last_sequence = full_sequences[-1].reshape(1, time_steps, 1)

# Generate predictions for the next 90 days
for _ in range(90):
    pred = model.predict(last_sequence, verbose=0)  # Predict the next value
    future_predictions.append(pred[0, 0])  # Append the prediction to the list
    # Update the sequence with the new prediction
    last_sequence = np.append(last_sequence[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

# Convert NumPy floats to Python floats
future_predictions = [float(pred) for pred in future_predictions]

# Format predictions by rounding to 2 decimal places
formatted_predictions = [round(pred, 2) for pred in future_predictions]



# %%
from datetime import datetime, timedelta

# Generate dates for the next 90 days
# Use the last date in the dataset as the start date
start_date = final_df['date'].max()  # This is already a Timestamp
future_dates = [start_date + timedelta(days=i) for i in range(1, 91)]

# Create a DataFrame with dates and predictions
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': formatted_predictions})



# %%
#Gated Recurrent Unit (GRU) model. GRU is a type of recurrent neural network (RNN) that is simpler than LSTM but often performs well in forecasting tasks.
#Data Preparation
#Sort Data by Date
final_df = final_df.sort_values(by='date')

# %%
# Select Features and Labels
features = final_df[['day_of_week', 'month', 'quarter', 'day_of_month']]
target = final_df['total_sales']

# %%
# Normalize Data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

# %%
# Create Sequences
import numpy as np

def create_sequences(data, target, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(target[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 7
X, y = create_sequences(features_scaled, target_scaled, time_steps)

# %%
# Train-Test Split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# %%
# Build the GRU Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,GRU, Dense

# Define the GRU model with an explicit Input layer
model = Sequential([
    Input(shape=(7, 4)),  # Explicitly specify 7 time steps and 4 features
    GRU(50, activation='relu'),
    Dense(1)  # Final/output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# %%
# Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# %%
# Evaluate the Model
from sklearn.metrics import mean_absolute_percentage_error

# Make predictions on the test data
y_pred = model.predict(X_test)

# Inverse scale the predictions and actual values
y_test_rescaled = scaler.inverse_transform(y_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)


# %%
import tensorflow as tf
import numpy as np

# Ensure 'pred' is a valid NumPy array with expected shape
pred_array = np.array(pred).flatten()  # Flatten to ensure it's a 1D array
pred_array = np.resize(pred_array, (4,))  # Resize to have exactly 4 elements
new_feature = tf.constant(pred_array, dtype=tf.float64)  # Match dtype to 'last_sequence'

new_feature = tf.reshape(new_feature, (1, 1, 4))  # Reshape to required format

# Reshape 'last_sequence' to match batch dimension and data type
last_sequence = tf.constant(X_test[-1], dtype=tf.float64)  # Ensure data type consistency, Convert 'X_test[-1]' to TensorFlow constant
last_sequence = tf.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

# Concatenate 'last_sequence' and 'new_feature' with matching data types and shapes

# Concatenate tensors with matching shapes and data types
last_sequence = tf.concat([last_sequence[:, 1:, :], new_feature], axis=1)

future_predictions = []

# Forecast for 90 days (3 months)
for _ in range(90):
    # Predict the next value based on the last sequence
    pred = model.predict(last_sequence)

    # Extract the predicted value and update the sequence
    # Process and update 'new_feature'
    pred_array = np.array(pred).flatten()
    pred_array = np.resize(pred_array, (4,))
    new_feature = tf.constant(pred_array, dtype=tf.float64)
    new_feature = tf.reshape(new_feature, (1, 1, 4))  # Ensure proper shape
    last_sequence = tf.concat([last_sequence[:, 1:, :], new_feature], axis=1)

    # Add the prediction to the results
    future_predictions.append(pred[0, 0])

# Rescale predictions using the scaler
future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))




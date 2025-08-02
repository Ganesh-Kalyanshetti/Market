import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
from flask import Flask, render_template, request



def main(para):
    
    # Download historical stock from library
    stock_data = yf.download(para, start='2024-01-01', end='2025-01-01')
    stock_data = stock_data[['Close']]  

    if stock_data.empty:
        raise ValueError(f"No stock data found for ticker '{para}'. Please check the ticker symbol.")

    
    plt.figure(figsize=(14, 5))
    plt.plot(stock_data)


    plt.title(f'{para} Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')

   
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    original_plot_url = base64.b64encode(img.getvalue()).decode()

    
    plt.clf()
    plt.close()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    # Create the training data set
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Create the training datasets
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')


    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Get the predicted stock prices
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train = scaler.inverse_transform([y_train])
    y_test = scaler.inverse_transform([y_test])

    # Plot the results
    plt.figure(figsize=(14, 5))
    plt.plot(stock_data.index, stock_data['Close'], label='Actual Prices')
    plt.plot(stock_data.index[time_step:len(train_predict) + time_step], train_predict, label='Training Predictions')
    plt.plot(stock_data.index[len(train_predict) + (time_step * 2) + 1:len(stock_data) - 1], test_predict, label='Testing Predictions')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{para} Stock Trained and Tested Graph')
    plt.legend()

    # Save the predicted stock prices 
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    prediction_plot_url = base64.b64encode(img.getvalue()).decode()

    
    plt.clf()
    plt.close()

    
    last_60_days = scaled_data[-time_step:]  # Getting  the last 60 days from the scaled data
    last_60_days_scaled = last_60_days.reshape(1, time_step, 1)  

    predicted_price_scaled = model.predict(last_60_days_scaled)  

    predicted_price = scaler.inverse_transform(predicted_price_scaled)  

    
    next_day = stock_data.index[-1] + pd.Timedelta(days=1)
    predicted_df = pd.DataFrame({'Close': predicted_price[0][0]}, index=[next_day])
    extended_data = pd.concat([stock_data, predicted_df])

    # Plot the results with the next day's predicted price
    plt.figure(figsize=(14, 5))
    plt.plot(extended_data.index[:-1], extended_data['Close'][:-1], label='Actual Prices')  # Actual prices in blue
    plt.plot(extended_data.index[-2:], extended_data['Close'][-2:], label='Predicted Next Day Price', color='orange')  # Predicted price in red
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{para} Price Prediction with Next Day')
    plt.legend()

    # Save the next day's prediction plot
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    next_day_plot_url = base64.b64encode(img.getvalue()).decode()

    
    plt.clf()
    plt.close()

    
    future_predictions = []
    current_input = last_60_days.reshape(1, time_step, 1)

    for _ in range(10):
        next_pred_scaled = model.predict(current_input)
        future_predictions.append(next_pred_scaled[0, 0])
        next_pred_scaled_reshaped = next_pred_scaled.reshape(1, 1, 1)
        current_input = np.append(current_input[:, 1:, :], next_pred_scaled_reshaped, axis=1)

   
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    
    future_dates = [stock_data.index[-1] + pd.Timedelta(days=i) for i in range(1, 11)]
    future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Close'])

    # extend the original data 
    extended_data = pd.concat([stock_data, future_df])

    # plot the results with the next 10 days 
    plt.figure(figsize=(14, 5))
    plt.plot(extended_data.index[:-10], extended_data['Close'][:-10], label='Actual Prices')  # Actual prices in blue
    plt.plot(extended_data.index[-11:], extended_data['Close'][-11:], label='Predicted Next 10 Days', color='green')  # Predicted prices in red
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{para} Price Prediction with Next 10 Days')
    plt.legend()

    # Save the next 10 days' prediction plot
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    ten_days_plot_url = base64.b64encode(img.getvalue()).decode()

    
    plt.clf()
    plt.close()

    
    return original_plot_url, prediction_plot_url, next_day_plot_url, ten_days_plot_url



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    global userinp
    userinp = ''
    return render_template('index.html')

@app.route('/h', methods=['GET', 'POST'])
def hello_world3():
    global userinp
    userinp = request.form.get('query')
    return render_template('index3.html', user_input=userinp)

@app.route('/goo', methods=['POST'])
def hello_world2():
    global userinp
    userinp = request.form.get('query')

    original_plot_url, prediction_plot_url, next_day_plot_url, ten_days_plot_url = main(userinp)
    return render_template('index2.html', original_plot_url=original_plot_url, prediction_plot_url=prediction_plot_url, next_day_plot_url=next_day_plot_url, ten_days_plot_url=ten_days_plot_url, user_input=userinp)

# prevent from reloading
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5003)

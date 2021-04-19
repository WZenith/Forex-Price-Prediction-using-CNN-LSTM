#!/usr/bin/env python
# coding: utf-8

# # PREDICTION OF FOREX PRICES  USING THE CNN-LSTM 
# # CURRENCY PAIR: GBPUSD

# IMPORTING REQUIRED PYTHON LYBRARIES AND SETTING DESIRED PLOTTING STYLE
import numpy as np
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 
import os
plt.style.use('seaborn-whitegrid')

# IMPORTING HISTORY DATASETS OF GBP VS USD FOREX
#df=yf.download(tickers='GBPUSD=X',start='2003-12-31',interval ='1d')
#df.to_csv("GBPUSD_csv")
url = r"GBPUSD_csv" 
## Read dataset to pandas dataframe
df = pd.read_csv(url, index_col = 'Date')
df.head()
df.tail()
#get the number of rows and columns in the data set
df.shape


# PREDICTION IS DONE USING CLOSING PRICE ONLY, SO TAKING OUT ONLY THE CLOSE PRICES
data = df.filter(['Close']).values
data

# PLOTTING THE CLOSE PRICES OF HISTORY DATASETS
plt.figure(figsize = (20,8))
plt.plot(data,'b',label = 'Original')
plt.xlabel("Days")
plt.ylabel('Price')
plt.title("EURUSD_1d")
plt.legend()

# DATA PREPROCESSING USING SCIKIT LEARN PYTHON LIBRARY
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)
scaled_data

# DATA SPLITTING INTO TRAINING, AND TESTING DATA SETS
training_size = int(len(scaled_data)*0.80) #Training size is 80% of the given data
print("Training_size:",training_size)
x_train_1 = scaled_data[0:training_size,:]
print(len(x_train_1))
test_data_1= scaled_data[training_size:,:1]
print(len(test_data_1))
print(len(x_train_1)), print(test_data_1)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
       # find the end of this pattern
       end_ix = i + n_steps
       # check if we are beyond the sequence
       if end_ix > len(sequence)-1:
          break
       # gather input and output parts of the pattern
       seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
       X.append(seq_x)
       y.append(seq_y)
    return np.array(X), np.array(y)

#Split into samples
time_step = 120
x_train, y_train = split_sequence(x_train_1, time_step)
x_test, y_test = split_sequence(test_data_1, time_step)
print(x_train.shape),print(y_train.shape)
print(x_test[-1]), print(y_test[-2:])

# reshape input to be [samples, time steps, features] which is required for LSTM
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)
x_train.shape, x_test.shape

# BUILDING A CNN-LSTM MODEL USING KERAS
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D ,MaxPooling1D, Dropout, Flatten, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.utils.vis_utils import plot_model
from keras.metrics import RootMeanSquaredError as rmse
from keras import optimizers

# define model
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2,activation='relu',padding = 'same',input_shape=(120,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss= 'mse' , metrics = [rmse()])
# Display model summary
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# TRAINING THE MODEL FOR 250 EPOCHS
history = model.fit(x_train, y_train, epochs = 250, validation_data = (x_test,y_test), batch_size=32, verbose=1)

# MODEL EVALUATION
history.history.keys()

### Plotting iteration-loss graph for training as well as validation
plt.figure(figsize = (10,6))
plt.plot(history.history['loss'],label='Training Loss',color='b')
plt.plot(history.history['val_loss'],label='Validation-loss',color='g')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title('LOSS')
plt.legend()

### Plotting iteration-rmse graph for training as well as validation
plt.figure(figsize = (10,6))
plt.plot(history.history['root_mean_squared_error'],label='Training RMSE',color='b')
plt.plot(history.history['val_root_mean_squared_error'],label='Validation-RMSE',color='g')
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title('ITERATION-RMSE')
plt.legend()

#evaluate training data
model.evaluate(x_train,y_train, batch_size = 32)

#evaluate testing data
model.evaluate(x_test,y_test, batch_size = 32)

# PREDICTION USING TRAINING DATA

# prediction using training data
train_predict = model.predict(x_train)
plot_y_train = y_train.reshape(-1,1)

# Actual vs predicted training data graph
plt.figure(figsize=(22,7))
plt.plot(scaler.inverse_transform(plot_y_train),color = 'b', label = 'Original')
plt.plot(scaler.inverse_transform(train_predict),color='red', label = 'Predicted')
plt.title("Actual Vs Predicted Graph Using Training Data")
plt.xlabel("Days")
plt.ylabel('Prices')
plt.legend()
plt.show()

# PREDICTION USING TESTING DATA

# prediction using testing data
test_predict = model.predict(x_test)
plot_y_test = y_test.reshape(-1,1)

# Actual vs predicted testing data graph
plt.figure(figsize=(22,7))
plt.plot(scaler.inverse_transform(plot_y_test),color = 'b',  label = 'Original')
plt.plot(scaler.inverse_transform(test_predict),color='g', label = 'Predicted')
plt.title("Actual Vs Predicted Graph Using Testing Data")
plt.xlabel("Days")
plt.ylabel('Prices')
plt.legend()
plt.show()

#    COMPARING THE LAST 5 ACTUAL VALUES AND THE LAST 5 PREDICTED VALUES

last_actual_five = scaler.inverse_transform(y_test[-5:])
last_predicted_five = scaler.inverse_transform(test_predict[-5:])
compare = pd.DataFrame(last_actual_five, columns = ['Actual_Prices'])
compare['Predicted_Prices'] = last_predicted_five
print(compare)

# VISUALIZATION OF THE GRAPH OF HISTORY DATA SETS WITH PREDICTION OF TRAINING AND TESTING DATA

### Plotting 
# shift train predictions for plotting
look_back = time_step
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2):len(data), :] = test_predict
# plot baseline and predictions
plt.figure(figsize=(22,8))
plt.plot(data,color = 'b', label = 'Original')
plt.plot(scaler.inverse_transform(trainPredictPlot),color='red',label = 'Train Predictions')
plt.plot(scaler.inverse_transform(testPredictPlot),color='green', label = 'Testing Predictions')
plt.title("Original Price Vs Predictions")
plt.xlabel('Period')
plt.ylabel('Prices')
plt.legend()
plt.show()

# SAVING MODELS FOR THE APPLICATION PURPOSES
model.save('gbpusd.h5')
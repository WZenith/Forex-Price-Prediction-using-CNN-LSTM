# Import required libraries
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import streamlit as st 
from PIL import Image 
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly as py

image = Image.open("Manipur.png")
st.image(image,use_column_width=True)

# Choosing the currency pair
dataset_name = st.sidebar.selectbox('Select the currency pair',('EURUSD','GBPUSD','USDJPY'))

def get_dataset(name):
	if name == 'EURUSD':
		st.sidebar.image(Image.open("Details_EURUSD.png"))
		#df_1=yf.download(tickers='EURUSD=X',start='2003-12-31',interval ='1d')#Download from yfinance
		url = r"EURUSD_csv"
		## Read dataset to pandas dataframe
		df_1 = pd.read_csv(url, index_col = 'Date')
		return df_1
	elif name == 'GBPUSD':
		st.sidebar.image(Image.open("Details_GBPUSD.png"))	
		#df_2=yf.download(tickers='GBPUSD=X',start='2003-12-31',interval ='1d')#Donwload from yfinance
		url = r"GBPUSD_csv"
		## Read dataset to pandas dataframe
		df_2 = pd.read_csv(url, index_col = 'Date')
		return df_2
	elif name == 'USDJPY':
		st.sidebar.image(Image.open("Details_USDJPY.png"))
		#df_3=yf.download(tickers='USDJPY=X',start='2003-12-31',interval ='1d')#Download from yfinance
		url = r"USDJPY_csv"
		## Read dataset to pandas dataframe
		df_3 = pd.read_csv(url, index_col= 'Date')
		return df_3

# Getting Data
df = get_dataset(dataset_name)
# data preprocessing
data = df.filter(['Close']).values
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)
pred = scaled_data[-120:]

# Loading Saved Model
from keras.models import load_model

if dataset_name == 'EURUSD':
	test_saved_modelz = load_model('eurusd.h5')
elif dataset_name == 'GBPUSD':
	test_saved_modelz = load_model('gbpusd.h5')
else:
	test_saved_modelz = load_model('usdjpy.h5')

# demonstrate prediction for next 10 days
from numpy import array

def forcast_future(nextt, dtas):
	x_input = dtas.reshape(1,-1)
	temp_input = list(x_input)
	temp_input = temp_input[0].tolist()
	lst_output = []
	n_steps = 120
	future_days = nextt
	i = 0
	while(i<future_days):
		if(len(temp_input)>120):
			x_input=np.array(temp_input[1:])
			x_input=x_input.reshape(1,-1)
			x_input = x_input.reshape((1, n_steps, 1))
			yhat = test_saved_modelz.predict(x_input, verbose=0)
			temp_input.extend(yhat[0].tolist())
			temp_input=temp_input[1:]
			lst_output.extend(yhat[0].tolist())
			i=i+1
		else:
			x_input = x_input.reshape((1, n_steps,1))
			yhat = test_saved_modelz.predict(x_input, verbose=0)
			temp_input.extend(yhat[0].tolist())
			lst_output.extend(yhat[0].tolist())
			i=i+1
	lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))
	return lst_output


# this is the main function in which we define our webpage 
	# giving the webpage a title 

st.subheader("Department of Computer Science, Manipur University, Canchipur-795003") 
	
	
# here we define some of the front end elements of the web page like 
# the font and background color, the padding and the text to be displayed 
html_temp = """
<div style ="background-color:green;padding:13px"> 
<h2 style ="color:white;text-align:center;">Forex Price Prediction Using CNN-LSTM</h1> 
</div> 
"""
# this line allows us to display the front end aspects we have 
# defined in the above code 
st.markdown(html_temp, unsafe_allow_html = True) 
st.info("This app is developed for the purpose of MCA-5th semester project only.")


if dataset_name == 'EURUSD':
	st.subheader("EURUSD")
	image = Image.open("EURUSDphoto.jpg")
	st.image(image,use_column_width=True)
elif dataset_name == 'GBPUSD':
	st.subheader("GBPUSD")
	image = Image.open("GBPUSDphoto.jpg")
	st.image(image,use_column_width=True)
elif dataset_name == 'USDJPY':
	st.subheader("USDJPY")
	image = Image.open("USDJPYphoto.jpg")
	st.image(image,use_column_width=True)


# Displaying Datasets
st.write("## HISTORICAL DATASETS")
shows = st.radio("Choose 'Show' if you want to see the Historical Datasets",("Hide",'Show'))
if shows == 'Show':
	st.write(df)
	st.write("No. of Rows = ",len(df))
	st.success('successfull')
	st.info("This Historical Price Datasets are downloaded from yfinance")

# Plotting Chart
st.write("## PRICE CHART (NOT LIVE)")
fg = px.line(df,y='Close',labels = {'x':'Time', 'y':'Prices'}, title = 'PRICE CHART: PERIOD = 1 DAY')
show = st.radio("Choose 'show' if you want to see the Price Chart",("Hide",'Show'))
if show == 'Show':
	st.plotly_chart(fg)
	st.success('successfull')

# PREDICT FUTURE PRICE
st.write("## PREDICTION FUTURE PRICES")
value = st.radio("Choose 'Show' if you want to see the prediction prices",('Hide','Show'))
if value == 'Show':
	nextt = st.slider('How many next days do you want to predict?',1,10,2)
	st.write("""### Predicting Close Prices for the next""",nextt,"days")
	future_pred = forcast_future(nextt,pred)
	st.write(future_pred)
	st.success("successfully predicted")
	
	day_new = np.arange(1,120+1)
	day_pred = np.arange(120+1,120+nextt+1)
	abc = data[-120:].reshape(120,)
	xyz = future_pred.reshape(len(future_pred),)
	trace1 = go.Scatter(x = day_new, y = abc, mode = 'lines', name = 'Previous Prices')
	trace2 = go.Scatter(x = day_pred, y = xyz, mode = 'lines', name = 'Predicted Prices', marker = dict(color = 'green'))
	dat = [trace1,trace2]
	layout = go.Layout(title = "FUTURE PRICE CHART", xaxis = {'title':'No. of Days'}, yaxis = {'title':'Prices'})
	fig = go.Figure(data = dat, layout = layout)
	st.plotly_chart(fig)
	st.warning("DISCLAIMER: The predicted prices are not recommended to use in the business or trading. This app is developed for the purpose of MCA-5th semester project only.")
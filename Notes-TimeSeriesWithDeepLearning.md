<h1>Notes: develop time series forecasting model via deep learning algorithms</h1>

# Step 1: prepare data

Perhaps to fit a Convolutional Neural Network (CNN) or Long Short-TermMemory (LSTM) Neural Network model, the first step is to transform a 2D data into a 3D structure. 

A simple time series has the following format: 
```
original series: [10, 20, 30, 40, 50, 60, 70, 80, 90]

2D series:
x1, x2, x3 y
[10 20 30] 40
[20 30 40] 50
[30 40 50] 60
[40 50 60] 70
[50 60 70] 80
[60 70 80] 90
```

<h1>Notes: develop time series forecasting model via deep learning algorithms</h1>

# Prepare data for deep learning model

Perhaps to fit a Convolutional Neural Network (CNN) or Long Short-TermMemory (LSTM) Neural Network model, the first step is to transform a 2D data into a 3D structure. 

An example of a simple time series and its 2D format structure `[samples, features]`, where the input shape is (6,3) and output (6,): 
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

CNN, LSTM, or other deep learning architecture require time series data to have a _3D format_ structure `[samples, timesteps, features]`, in which the three dimensions are:
1. Samples. One sequence is one sample. A batch is comprised of one or more samples.
2. Time Steps. One time step is one point of observation in the sample. One sample is comprised of multiple time steps.
3. Features. One feature is one observation at a time step. One time step is comprised of one or more features.


# LSTM - multivariate

A time series may have more than one observation for each time step. Multivariate time series comes in two flavor:
1. multiple input series

   A numerical example of multiple input series is as follow, where the input shape for each sample is (3,2).
   ```
    [[10 15]
     [20 25]
     [30 35]] 65
    [[20 25]
     [30 35]
     [40 45]] 85
   ```
3. multiple parallel series.

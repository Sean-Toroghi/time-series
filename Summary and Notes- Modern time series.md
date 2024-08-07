<h1>Stochastic time series - from foudnation to edge approaches</h1>

Last update: 07/2024

# Time series - overview





















































---
<h1> Cyclic time serie </h1>
A time series can be constituted of a set of the multi-dimensional feature vectors ordered according to the time occurrence of the vectors.

In a holistic view, there are three different ways to model dynamic contents of time series: deterministic, chaotic and stochastic model. 
- With a deterministic models, time series can be expressed by a closed mathematical formula. A deterministic system is completely described by a linear time invariant differential equation. . 
- Chaotic time series are constituted of a nonlinear differential equation, with an unknown initial condition. Here the output time series is the solution to a nonlinear differential equation with an unknown initial condition. Initial condition is cruitial in forecasting a chaotic time series. By knowing the initial conditions, one can completely find the output time series. This makes the initial condition in chaotic time series system an important piece of infrmation.
- In stochastic models, an output time series is assumed to have resulted from a nonlinear and time dependent differential equations. We cannot fully define a linear time invariant differential equation for the system, so the solution of the system is not uniquely identified. A complete solution cannot be found for the output, and therefore, an optimization technique must be invoked for the system identification.

To formulate a model time series analysis, in simplest form a set of events occuring in the past and present are used to predict the future events. Any $m$ dimensional series then can be decomposed into $m$ univariate series. It is often the case that the time series is sliced into the temporal windows of L samples, and the model parameters are found by utilising contents of each window, sliding over the time series with an overlap of V samples between each two successive windows. Root mean square of the predicted error, defined as the difference between actual values of time series and the predicted ones, can be employed as an informative metric to find an understanding about the model capability in identifying time series.

__Variation in time series__

There are two types of variation in time series analysis:
- Variation of the time series values for a certain subject with respect to time, t
- Variation of the time series value at a certain time point with respect to the subjects, i





A group of stationary time series whose temporal statistics are equal to their population statistics (counter moments) is known as ergodic time series. 

__Cyclic time series__

A type of stochastic time-series, its values resemble repetitive contents, but cannot be categorized as periodic time series. A periodict time series is predictable, which is not possible in the case of stochastic time series. In these cases, even though regularity cannot be observed in the value of time series, meaning that the values are not exactly repeated at a certain, priory known points of the time. Nevertheless, certain patterns are repetitively seen over a time span. These types of time series are named as cyclic time series. Electrocardiogram is an example of cyclic time series.

A cyclic model for processing a time series assumes that the time series resembles random behaviour within the cycles, and also the cycle duration by itself is also a random variable. In many practical cases, an auxiliary signal is recorded along with the time series synchronously. The auxiliary signal helps to identify the onset and the endpoint information of the time series.

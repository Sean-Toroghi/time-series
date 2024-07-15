# Multi-variate and multi-step forecasting 

A multivariate - mulsti-step forecasting means forecasting n-steps int he future, using a sequence of length m, with multiple features. There are several approaches to solve this problem, among which are:
- employ deep-learning model, which could generate a vector as output. This is a good approach, since deep-learning model captures interdependencies among variables.
- recusrsive approach, which employs a model such as neural network to generate one-step-ahead forecast and use it recursively to get the multi-step output.
- employ multiple models, each generate a single output for each forecasting horizon.
- combine recursive and muilti-model approach: in which multiple models are trained for each forecast horizon, and feeding the previous forecast as input for the next model.

## global models
Global model forecasting focus is on handling multiple related time-series and caputring relavent information across them. 


The main idea of a global model forecasting approach is there are common patterns across different related time-series. We can employ nueral netweok methods such as LSTM to handle this task. The main difference, compare with single time series forecasting, is how the data is preprocessed. 

Comparing the preprocessing for single and multiple time series forecasting
- single time series: the model input format is [batch_size, sequence_length, num_features], while the output format is [batch_size, horizon].
- multiple time-serie forecasting: model input and output have the same dimension. However, the input aggregates the information across multiple time series. This increase the ability of the model to learn new patterns/dependencies. Also, the output is forecasts of multiple time series simultaneously. 

## Hyperparameter tuning with Ray Tune

__Steps__
1. Define the search space: First, define the hyperparameter space you want to explore.
2. Configure Ray Tune: Initialize the Tune experiment with the desired settings, such as the number of trials, resources, and so on.
3. Run the optimization: Execute the experiment by passing the training function and the defined search space.
4. Analyze the results: Utilize Ray Tune’s tools to analyze the results and identify the best hyperparameters.

## Other model architectures

__Methods__
- N-Beats
- DeepAR with GluonTS
- Transformer with NeuralForecast
- Temporal Fusion Transformer with GluonTS
- Informer model with NeuralForecast
  
### N-Beats model
N-BEATS originally designed to handle problems that have several univariate time series. It is available as off-the-shelf model in `pytorch_forecasting`. 

__Steps__

1. Define N-BEATS requires to create an instance with the following parameters:
  ```python
  from pytorch_forecasting import NBeats
  model = NBeats.from_dataset(
    dataset =
    , stack_type =
    , num_blocks =
    , num_block_layers = 
    , widths =
    , sharing =
    , backcast_loss_ratio =
    ,)
  ```
  - dataset: The TimeSeriesDataSet instance that contains the training set.
  - stack_types: The mode you want to run N-BEATS on. A trend and seasonality type of stack enables the model to be interpretable, while a ['generic'] setup is usually more accurate.
  - num_blocks: A block is the cornerstone of the N-BEATS model. It contains a set of fully connected layers that model the time series.
  - num_block_layers: The number of fully connected layers in each block.
  widths: The width of the fully connected layers in each block.
  - sharing: A Boolean parameter that denotes whether the weights are shared blocks per stack. In the interpretable mode, this parameter should be set to True.
  - backcast_loss_ratio: The relevance of the backcast loss in the model. Backcasting (predicting the input sample) is an important mechanism in the training of N-BEATS. This parameter balances the loss of the backcast with the loss of the forecast.



# Time-series classification

# Overview of time series 


__Resampling__: change the frequency of time series

It is usedul for either of the following two scenarios:
- changing the frequency
- dealing with irregular dataset

[reference](https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects)

---

__missing values__

- one common imputation approach is to replace missing values with central value (median, mean) `df.fillna(df.mean())`
- another option is fill with the last known value: `df.ffill()`
- also we can impute with the next known value: `df.bfill()`

[reference](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html)

---

__Decomposing time series__

It is the process of splitting a time series into its components such as trend and seasonality.
- trend can be estimated by computing moving average
- seasonality can be estimated by averaging the values of each period.

- Option 1- use statmodel library:
```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(x=series_daily,
                            model='additive',
                            period=365)
```

- option 2 (assumption single season): use `STL()` function - Seasonal and Trend decomposition using LOESS (STL) in the statmodels libarary:
```python
from statsmodels.tsa.seasonal import STL
result = STL(endog=series_daily, period=365).fit()
```

- option 3 (multiple seasons): use `MSTL()` function
```python
from statsmodels.tsa.seasonal import MSTL
result = MSTL(endog=series_daily, periods=(7, 365)).fit()
```

---

__Auto-correlation__

 The autocorrelation function quantifies the linear relationship between a time series and a lagged version of itself.

 [Reference](https://otexts.com/fpp3/components.html)

 ---

 __Stationary__

A time series is stationary if its statistical properties do not change. Two property of a time series that violate stationary property are seasonality and variance.

1. seanonality
   
    Since seasonality could break the stationary property of a time series, we remove the seasonality to improve the performance of a time series forecasting model. This could be achived by computing differencing as follow:
    
    Mehods 1:
    1. Estimate the number of differencing steps required for stationarity. Example: augmented Dickey-Fuller test, or the KPSS test.
    2. Apply the required number of differencing operations.
    
    Method 2:
    1. differencing apply over seasonal periods.
    2. seasonal differencing involves computing the difference between consecutive observations of the same seasonal period.
    
    [References](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ndiffs.html)
    
2. variance:

    Another property of time series that violates stationary property is variance. A time series with variance is called heteroskedasticity, and could be detected with a having a long-tail distribution.
    
    To remove heteroskedasticity, we need to follow these two steps
    1. examine if the data is heteroskedasticity. Example White test or the Breusch-Pagan test
    2. apply ransformation such as logarithm to stablize variance
  
  
  [Reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html)

---
---

# Univariate forecasting

## Create baseline
Baseline is used to evaluate if more sophisticated models actuly effective in learning from data, compare with the naive models. Also, they could be helpful in revealing a flaw in a complex model. Furthermore, naive models help to understand the data.

__Some libraries__
- [GluonTS](https://ts.gluon.ai/dev/api/gluonts/gluonts.model.seasonal_naive.html)
- [sktime](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.naive.NaiveForecaster.html)
- [pytorch_forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.baseline.Baseline.html)

To create a baseline for forecasting, we can employ several approaches among which are:
- __Naive__: the simplest method is naive approach, in which we assume the next observation will be the same as the last one.
- __Seasonal naive__: this approach assumes the next observation will be similar to the previous observation from the same season.
- __Mean__: in this apprach, we take the average of the past observation as the prediction for the future



## ARIMA and SARIMA

__ARIMA__
ARIMA models explain a time series based on its past values. They combine aspects of autoregressive, integrated, and moving average models:
- AR part: a regression where the next value of the time series is modeled based on the previous p lags.
- I part: to make data stationary (ARIMA is defined for stationary data), the I part represents the number of differencing operations (d) required to make the series stationary.  
- MA part: is another regression where the next value of the series is modeled based on the past q errors.

__Determine ARIMA parameters: p,d,q__

To determine ARIMA parameters we need to employ explantory methods such as checking the autocorrelation function (ACF) and the partial autocorrelation function (PACF) plots.

__Note__: ARIMA assumes model is stationary.

__SARIMA__

SARIMA is an alternative method to ARIMA, for non-stationary time series with a seasonal component. 










































---
---


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

References
- [pytorch forecasting library](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nbeats.NBeats.html)
- [NHiTS mehthods](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nhits.NHiTS.html#pytorch_forecasting.models.nhits.NHiTS)
- [N-BEATSx: a varient of N-BEATS that handles exogenous variables](https://nixtla.github.io/neuralforecast/models.nbeatsx.html)
- 
N-BEATS originally designed to handle problems that have several univariate time series. It is available as off-the-shelf model in `pytorch_forecasting`. 

It is based on two main components:
1. a double stack of residual connections (forecasting and backcasting).
2. a deep stack of densely connected layers.


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
2. Define call back and pass the model to the pytorch-lightning trainer
  ```python
  import lightning.pytorch as pl
  from lightning.pytorch.callbacks import EarlyStopping
  early_stop_callback = EarlyStopping(monitor="val_loss",
      min_delta=1e-4,
      patience=10,
      verbose=False,
      mode="min")
  trainer = pl.Trainer(
      max_epochs=30,
      accelerator="auto",
      enable_model_summary=True,
      gradient_clip_val=0.01,
      callbacks=[early_stop_callback],
      )
  # train the model
  trainer.fit(
      model,
      train_dataloaders=datamodule.train_dataloader(),
      val_dataloaders=datamodule.val_dataloader(),
      )
  # save model
  best_model_path = trainer.checkpoint_callback.best_model_path
  best_model = NBeats.load_from_checkpoint(best_model_path)
  # evaluate model
  predictions = best_model.predict(datamodule.test.to_dataloader(batch_size=1, shuffle=False))
  actuals = torch.cat(
      [y[0] for x, y in iter(
          datamodule.test.to_dataloader(batch_size=1, 
              shuffle=False))])
  print((actuals - predictions.cpu()).abs().mean())
  # visualize the prediction
  raw_predictions = best_model.predict
    (datamodule.val_dataloader(),
    mode="raw",
    return_x=True)
  best_model.plot_interpretation(x=raw_predictions[1],
      output=raw_predictions[0],
      idx=0)
  ```

### Optimizing the learning rate
`lightning.pytorch` libarary has a moddule to optimize the model, which can be used to optimize learning rate: `from lightning.pytorch.tuner import Tuner`.

Turner acts as a wrapper of the trainer object, and then use `lr_optim()` method to optimize learning rate. 

Example:

```python
from lightning.pytorch.tuner import Tuner
import lightning.pytorch as pl
from pytorch_forecasting import NBeats
trainer = pl.Trainer(accelerator="auto", gradient_clip_val=0.01)
tuner = Tuner(trainer)
model = NBeats.from_dataset(
    dataset=datamodule.training,
    stack_types=['trend', 'seasonality'],
    num_blocks=[3, 3],
    num_block_layers=[4, 4],
    widths=[256, 2048],
    sharing=[True],
    backcast_loss_ratio=1.0,
)

lr_optim = tuner.lr_find(model,
    train_dataloaders=datamodule.train_dataloader(),
    val_dataloaders=datamodule.val_dataloader(),
    min_lr=1e-5)

lr_optim.suggestion()
fig = lr_optim.plot(show=True, suggest=True)
fig.show()
```

### GluonTS: probabisting time series modeling with PyTorch
`glounts` is library with array of utilities for tiume series data processing and model evaluation.



### DeepAR model
DeepAR is a SOTA forecasting method by Amazon, that is very powerful when there’s a need to generate forecasts for multiple related time series.. It utilizes autoregressive recurent networks and suitable for tasks that benefits from longer horizons, such as demand forecasting. 

With RNN architecture, DeepAR leverage LSTM orGRU to model a time series data. DeepAR, in addition to forecasting, can generates probabilistic forecasts, which mean it can provide a distribution over possible future values. This feature allows for further investigate the uncertaintity associated with the predictions. Furthermore, it considers commonalities between series, when working with multiple related time series. DeepAR can be trained to generate forecast for all series on global dataset, or trained on each time series separatetly (less efficient).

DeepAR is sprecifically designed to make forecast in the following scenarios:
- we have multiple related time series
- dataset with seasonality or recurring patterns
- it is important to generate probabilitic forecast (predict point estimate and also uncertatinty intervals).
- perform very good for tasks similar to demand forecasting, stock price prediction, and web traffic prediction.

```python
model = [
    VanillaTransformer(
        h=HORIZON,
        input_size=N_LAGS,
        max_steps=100,
        val_check_steps=5,
        early_stop_patience_steps=3,
    ),
]
nf = NeuralForecast(models=model, freq="D")
Y_df = df[df["unique_id"] == 0]
Y_train_df = Y_df.iloc[:-2*HORIZON]
Y_val_df = Y_df.iloc[-2*HORIZON:-HORIZON]
training_df = pd.concat([Y_train_df, Y_val_df])
nf.fit(df=training_df, val_size=HORIZON)

# visualize the forecast
forecasts = nf.predict()
Y_df = df[df["unique_id"] == 0]
Y_hat_df = forecasts[forecasts.index == 0].reset_index()
Y_hat_df = Y_test_df.merge(Y_hat_df, how="outer", 
    on=["unique_id", "ds"])
plot_df = pd.
    concat([Y_train_df, Y_val_df, Y_hat_df]).set_index("ds")
plot_df = plot_df.iloc[-150:]
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
plot_df[["y", "VanillaTransformer"]].plot(ax=ax, linewidth=2)
ax.set_title("First Time Series Forecast with Transformer", fontsize=22)
ax.set_ylabel("Value", fontsize=20)
ax.set_xlabel("Timestamp [t]", fontsize=20)
ax.legend(prop={"size": 15})
ax.grid()
plt.show()
```

__Steps__
1. preprocess data
   ```python
   from gluonts.dataset.common import ListDataset
   from gluonts.dataset.common import FieldName
   train_ds = ListDataset(
       [
           {FieldName.TARGET: entry["target"], 
               FieldName.START: entry["start"]}
           for entry in dataset.train
       ],
       freq=dataset.metadata.freq,
    )
    ```
   
2. define DeepAR estimator.
  ```python
  from gluonts.torch.model.deepar import DeepAREstimator
  N_LAGS=7
  HORIZON=7
  estimator = DeepAREstimator(
      prediction_length=HORIZON, # number of forecasting steps
      context_length=N_LAGS, # number of steps to consider when forming input seq
      freq=dataset.metadata.freq,
      trainer_kwargs={"max_epochs": 100},
  )
  ```

3. train model
  ```python
  predictor = estimator.train(train_ds)
  ```
4. make prediction
  ```python
  forecast_it, ts_it = make_evaluation_predictions(
      dataset=dataset.test,
      predictor=predictor,
      num_samples=100,
  )
  forecasts = list(forecast_it)
  tss = list(ts_it)
  fig, ax = plt.subplots(1, 1, figsize=(10, 6))
  ts_entry = tss[0]
  ax.plot(ts_entry[-150:].to_timestamp())
  forecasts[0].plot(show_label=True, ax=ax, intervals=())
  ax.set_title("Forecast with DeepAR")
  ax.legend()
  plt.tight_layout()
  plt.show ()
  ```

---

### Transformer-based forecasting method: NueralForecast

NueralForecast is a vanila transformer model for forecasting task. A transfomer model process entire input sequence at once. This is the main difference of a transfomer-based model compared with seq-to-seq approach such as LSTM. 

The transformers model processes a time series data by encoding the sequence using the self-attention mechanism. This is a great advantage in the case in which dataset carries patterns/dependencies over a long horizon, or when the relevance of past data changes over time (dynamic dataset). The multi-head attention mechanism allows the Transformer to focus on different time steps and features concurrently, making it especially powerful for complex time series with multiple interacting patterns and seasonality.

Data preparation for `nueralforecast` library: each observation consists of three peices of information: timestamp, time-series identifier, and corresponding value.

Model creation [Ref.](https://nixtla.github.io/neuralforecast/models.vanillatransformer.html): creating/initializing a model requires to define a group of parameters, including forecasting steps, number of training sets, and early stopping.

### Transformer-based forecating model: temporal fusion transformer (TFT) 

TFT is an attention-based architecture, developed by Google. It uses its recurrent layers to learn temporal relasionships at different scales, combined with self-attention layers. It also has feature selection capability, and uses quantile loss function. 

Some of the advantages of utilizing this model:
- temporal processing
- attention mechanism, which enables the model to dynamically assign importance to different time steps
- gating mechanism: its gated residual network feature provides flexibility in the modeling process, adapting to the complexity of the data.
- variable selection networks: it has capability to determine the relevance of each covariate to the forecsat, by weighting the input features' importance and filter out noise.
- static covariate encoders, which encodes static information into multiple context vectors.
- quantile prediction, provides a range of possible outcomes.
- interpretable output, by providing feature importance.

__Applications__
- if there is a need to explain how the predictions are produced (utilizing variable network seelection and tempral multi-head attention layer).
- it acts as both forecasting method and analytical tool.


Building the model - example:
```python
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
N_LAGS = 7
HORIZON = 7
estimator = TemporalFusionTransformerEstimator(
    prediction_length=HORIZON,
    context_length=N_LAGS,
    freq=dataset.metadata.freq,
    trainer_kwargs={"max_epochs": 100},
)
#make prediction
from gluonts.evaluation import make_evaluation_predictions

predictor = estimator.train(train_ds)
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor,
    num_samples=100,
)

# visualize prediction
import matplotlib.pyplot as plt
ts_entry = tss[0]
ax.plot(ts_entry[-150:].to_timestamp())
forecasts[0].plot(show_label=True, ax=ax, intervals=())
ax.set_title("Forecast with Temporal Fusion Transformer")
ax.legend()
plt.tight_layout()
plt.show()
```

### Transformer-based forecasting model: Informer

Informer is a Transformer-based forecasting model, tailored for long-term forecasting. IT has an improved attention mechism that reduces computational cost. 

The input data requires three set of information: timestamp, ID of the time series, and value of the corresponding time series.

Implementation example [ref.](https://nixtla.github.io/neuralforecast/models.informer.html):
```python
# define model
from neuralforecast.core import NeuralForecast
from neuralforecast.models import Informer
N_LAGS = 7
HORIZON = 7
model = [Informer(h=HORIZON,
    input_size=N_LAGS,
    max_steps=1000,
    val_check_steps=25,
    early_stop_patience_steps=10)]
nf = NeuralForecast(models=model, freq='D')

# Train model
nf.fit(df=df, val_size=val_size)

# make prediction
forecasts = nf.predict()
forecasts.head()
```
### Compare the performance of different models (transformer-based)
To compare the preformance of different models, we can employ cross-validation, train with train and validations sets, and evaluate on test set. The model with highest performance (based on its score on test set) is picked, and trained on the whole dataset as the final solution. 


Example: compare performance of multiple trasnformer-based models
```python
# define models for the comparison

from neuralforecast.models import Informer, VanillaTransformer
models = [
    Informer(h=HORIZON,
        input_size=N_LAGS,
        max_steps=1000,
        val_check_steps=10,
        early_stop_patience_steps=15),
    VanillaTransformer(h=HORIZON,
        input_size=N_LAGS,
        max_steps=1000,
        val_check_steps=10,
        early_stop_patience_steps=15),
]

# perform comparison
from neuralforecast.core import NeuralForecast
nf = NeuralForecast(
    models=models,
    freq='D')
cv = nf.cross_validation(df=df,
    val_size=val_size,
    test_size=test_size,
    n_windows=None)

# compute MAE
from neuralforecast.losses.numpy import mae
mae_informer = mae(cv['y'], cv['Informer'])
mae_transformer = mae(cv['y'], cv['VanillaTransformer'])


```

# point forecasting - probabilistic time series forecasting


---

## Other refrences

- [TimeShap](https://github.com/feedzai/timeshap)
- [Temporal fusion transformer (TFT)](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html#Interpret-model)
- [Pytorch forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials.html)
- [Nueral forest library](https://nixtla.github.io/neuralforecast/core.html)
- []()
- []()
- []()
- []()
# Time-series classification

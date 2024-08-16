<h1>Time series forecasting: Code depository</h1>

__Table of contents__
- [FeatureConfig](#FeatureConfig) -  [Link](https://github.com/Sean-Toroghi/time-series/blob/main/codes/FeatureConfig.py)


---
# <a name= 'FeatureConfig'>FeatureConfig</a>

[Link](https://github.com/Sean-Toroghi/time-series/blob/main/codes/FeatureConfig.py)

__Description__ A Python dataclass for preocessing data for a machine learning model. It has a method `get_X_y` that returns tuple of (features, target, original_target).

__Methods:__
- `get_X_y`
  - Inputs:
    - df: A DataFrame that contains all the necessary columns, including the target, if available
    - categorical: A Boolean flag for including categorical features or not
    - exogenous: A Boolean flag for including exogenous features or not
  - Outputs: tuple of (features, target, original_target)
 

__Inputs:__
- date: A mandatory column that sets the name of the column with date in the DataFrame.
- target: A mandatory column that sets the name of the column with target in the DataFrame.
- original_target: If target contains a transformed target (log, differenced, and so on), original_target specifies the name of the column with the target without transformation. This is essential in calculating metrics such as MASE, which relies on training history. If not given, it is assumed that target and original_target are the same.
- continuous_features: A list of continuous features.
- categorical_features: A list of categorical features.
- boolean_features: A list of Boolean features. Boolean features are categorical but only have two unique values.
- index_cols: A list of columns that are set as a DataFrame index while preprocessing. Typically, we would give the datetime and, in some cases, the unique ID of a time series as indices.
- exogenous_features: A list of exogenous features. The features in the DataFrame may be from the feature engineering process, such as the lags or rolling features, but also external sources such as the temperature data in our dataset. This is an optional field that lets us bifurcate the exogenous features from the rest of the features. The items in this list should be a subset of continuous_features, categorical_features, or boolean_features.

---

# <a name= ''></a>


---

# <a name= ''> # </a>

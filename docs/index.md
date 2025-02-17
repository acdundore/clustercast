![Clustercast Logo](img/clustercast_logo_with_words.png)

# Overview

`clustercast` is a Python library for ML-based time series forecasting that supports both single-series (local) forecasting and multi-series (global) forecasting for grouped/hierarchical time series. The "cluster" in the name comes from the technique of modeling grouped (clustered) time series with shared traits concurrently. It provides two main forecasting approaches:

- **Direct Forecasting**: Trains separate models for each forecast step, using direct multi-step forecasting
- **Recursive Forecasting**: Trains a single model and uses its one-step-ahead predictions recursively for multi-step forecasting

Key features that differentiate `clustercast` from other libraries include:

- Support for grouped time series that share common categories using a global architecture
- Ability to perform stationarity tests for all time series simultaneously (ADF and KPSS)
- Differencing and Box-Cox transformations
- Automatic calculation of lag features
- Several types of seasonality features (Fourier, one-hot, and ordinal), which all support multiple seasonalities
- Sample weighting to emphasize recent observations
- Prediction intervals via Conformal Quantile Regression (with direct forecaster) or bootstrapped residuals (with recursive forecaster)
- Model-agnostic architecture supporting any custom ML model, with `LightGBM` as the default
- Ability to tune the hyperparameters of the built-in `LightGBM` model
- Handling of exogenous variables
- Support of missing data (for both individual features and entire timesteps)

Both forecasters share a similar API and handle data preprocessing automatically, making it easy to experiment with different forecasting approaches/parameters with minimal changes to code. The library is designed for flexibility and ease of use, allowing users to quickly implement sophisticated forecasting solutions while maintaining control over model parameters and preprocessing options.

---

## Installation

```bash
pip install clustercast
```

---

## Usage

The forecasting models in `clustercast` are simple to use, but come with many additional parameters allowing the user to implement sophisticated preprocessing techniques with ease. 
An example using the recursive forecasting class is shown below.
In this example, there are 12 distinct time series representing sales over time for different regions and product categories.

```python
# imports
from clustercast.datasets import load_store_sales
from clustercast import RecursiveForecaster

# load store sales data
data = load_store_sales()
print(data)
```

```profile
     ID         YM   Region         Category      Sales
0     1 2015-01-01  Central        Furniture    506.358
1     2 2015-01-01  Central  Office Supplies    996.408
2     3 2015-01-01  Central       Technology     31.200
3     4 2015-01-01     East        Furniture    199.004
4     5 2015-01-01     East  Office Supplies    112.970
..   ..        ...      ...              ...        ...
424   8 2017-12-01    South  Office Supplies   5302.324
425   9 2017-12-01    South       Technology   2910.754
426  10 2017-12-01     West        Furniture  14391.752
427  11 2017-12-01     West  Office Supplies   9166.328
428  12 2017-12-01     West       Technology   8545.118

[429 rows x 5 columns]
```

```python
# create the forecasting model
model = RecursiveForecaster(
    data=data, # provide the full dataset
    endog_var='Sales', # the sales column will be forecasted
    id_var='ID', # indicates the different time series identifier column
    group_vars=['Region', 'Category'], # group features that differentiate the time series
    timestep_var='YM', # indicates the timestep column
    lags=12, # include lags 1 through 12
    seasonality_ordinal=[12], # include an ordinal seasonality feature
)

# fit the model
model.fit()

# make predictions out to 12 steps ahead
forecast = model.predict(steps=12)
print(forecast)
```

```profile
     ID         YM   Region         Category     Forecast
0     1 2018-01-01  Central        Furniture  3249.188111 
1     2 2018-01-01  Central  Office Supplies  2484.753879
2     3 2018-01-01  Central       Technology  3015.802614
3     4 2018-01-01     East        Furniture  1845.889868 
4     5 2018-01-01     East  Office Supplies  3785.740747
..   ..        ...      ...              ...          ...
139   8 2018-12-01    South  Office Supplies  4050.859234
140   9 2018-12-01    South       Technology  3080.471316
141  10 2018-12-01     West        Furniture  9342.107224
142  11 2018-12-01     West  Office Supplies  7727.692540
143  12 2018-12-01     West       Technology  9912.039919

[144 rows x 5 columns]
```
The output contains forecasts out to 12 months ahead for each of the 12 unique time series.
This is a fairly basic example implementation without much thought put into preprocessing or model tuning. 
This example can be easily extended and optimized using the built-in functionality in `clustercast`!

---

## License

`clustercast` was created by Alex Dundore. It is licensed under the terms of the MIT license.

---

## Credits and Dependencies

`clustercast` is powered by [`numpy`](https://numpy.org/), [`pandas`](https://pandas.pydata.org/), [`statsmodels`](https://www.statsmodels.org/stable/index.html#), and [`LightGBM`](https://lightgbm.readthedocs.io/en/latest/index.html).

# Recursive Forecaster

The `RecursiveForecaster` class implements recursive multi-step forecasting, supporting both single-series (local) forecasting and multi-series (global) forecasting for grouped or hierarchical time series. It trains a single model and uses the model's predictions as inputs for subsequent predictions, iterating through the forecast horizon. The class handles a variety of time series preprocessing techniques natively, including differencing, Box-Cox transformations, seasonality features, and lag calculations. The `RecursiveForecaster` provides both point forecasts and prediction intervals, with optional Conformal Quantile Regression (CQR) for state-of-the-art interval coverage. This flexible forecaster is well-suited for time series with complex patterns and multiple relevant predictors.

---

> *class* clustercast.**RecursiveForecaster**(*data, endog_var, id_var, timestep_var, group_vars=[], exog_vars=[], boxcox=1, differencing=False, include_level=True, include_timestep=False, lags=1, sample_weight_halflife=None, seasonality_fourier={}, seasonality_onehot=[], seasonality_ordinal=[], lgbm_kwargs={'verbose': -1}, base_regressor=None*)

---

### Parameters

Once the model class has been instantiated, the parameters may not be changed.

| Parameter | Description |
|---|---|
| data : `pd.DataFrame` | The input data containing the time series, IDs, timesteps, and any <br>grouping or exogenous variables. |
| endog_var : `str` | The name of the target variable to forecast. |
| id_var : `str` | The name of the column containing unique identifiers for each time <br>series. |
| timestep_var : `str` | The name of the column containing the time steps. The timestep <br>values may either be datetimes, integers, or floats. |
| group_vars : `list` | List of column names containing categorical variables used to group <br>the time series. |
| exog_vars  : `list` | List of column names containing exogenous variables to use as <br>predictors. |
| boxcox : `float` or `int` | The Box-Cox transformation parameter. Use `1` for no transformation. <br>A value of `0` will perform a log transformation. |
| differencing : `bool` | Whether to apply first-order differencing to the target variable. |
| include_level : `bool` | Whether to include the level of the target variable as a feature. <br>When `True`, this is only included when differencing is applied. |
| include_timestep : `bool` | Whether to include the time step as a feature. If `True`, an integer <br>index starting at zero is mapped to each unique timestep <br>chronologically and passed to the regressor. Use with caution, as <br>this may make your model prone to overfitting. |
| lags : `int` or `list` | The number of lags, or a list of specific lag values, to use as features. |
| sample_weight_halflife : `int` | The halflife, in number of timesteps, used to calculate sample weights <br>during the model fit (more recent timesteps have a heavier weight). If <br>`None`, all samples are weighted equally. |
| seasonality_fourier : `dict` | Dictionary with periods as the keys and number of Fourier terms as <br>values. |
| seasonality_onehot : `list` | List of periods for one-hot encoded seasonality features. |
| seasonality_ordinal : `list` | List of periods for ordinal encoded seasonality features. |
| lgbm_kwargs : `dict` | Additional keyword arguments to pass to LGBMRegressor, if no <br>custom base regressor is used. |
| base_regressor : `class` | Alternative regressor class to use instead of LGBMRegressor. You can <br>create an custom wrapper for any statistical or machine learning <br>regressor if certain criteria are met. See the examples page for more <br>information. |

---

### Methods

**.fit**(*alpha=None*)
	
>Creates and fits the forecasting model.
Trains a single model that will be used recursively for multi-step forecasting. The model can
optionally be used to generate prediction intervals via bootstrapped residuals.

> | Argument | Description |
|---|---|
| alpha : `float` | Miscoverage rate for prediction intervals (e.g., 0.05 for 95% <br>intervals). If `None`, only point forecasts are produced. |

**.predict**(*steps=1, exog_data=None, bootstrap_iter=500*)
	
>Generates forecasts for multiple steps ahead.
Makes predictions up to the specified number of steps ahead by using one-step-ahead forecasts as inputs for subsequent timesteps.
Optionally generates prediction intervals via bootstrapped residuals.

> | Argument | Description |
|---|---|
| steps : `int` | Number of steps ahead to forecast. Default is `1`. |
| exog_data : `pd.DataFrame` | Future values of exogenous variables. Must contain the same columns <br>as the exogenous variables used during fitting, along with the ID<br> and timestep variables. Although this is an optional argument, <br>performance may be degraded if exogenous variables were used for <br>training and are not provided for prediction. |
| bootstrap_iter : `int` | Number of bootstrap iterations to use when generating prediction <br>intervals. It is recommended to use a bare minimum of `100` bootstrap <br>iterations. An excessive number of iterations will be computationally<br> intensive. Only used when alpha was specified during fitting. Default<br> is `500`. |

> | Returns | Description |
|---|---|
| forecasts : `pd.DataFrame` | DataFrame containing the forecasts and optionally prediction intervals<br> for each time series at each forecast horizon. |

**.stationarity_test**(*test='both'*)
	
>Tests for stationarity of each time series before and after the model's transformations.
Performs either Augmented Dickey-Fuller (ADF) test, Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test, or both.
The ADF test has a null hypothesis of non-stationarity, while KPSS has a null hypothesis of stationarity.

> | Argument | Description |
|---|---|
| test : `str` | Which stationarity test(s) to perform: <br>-`'adf'`: Augmented Dickey-Fuller test only <br>-`'kpss'`: KPSS test only <br>-`'both'`: Both ADF and KPSS tests (default) |

> | Returns | Description |
|---|---|
| results : `pd.DataFrame` | DataFrame containing test results with columns: <br>- ID variable <br>- 'Raw ADF p-value': ADF test p-value on raw data (optional) <br>- 'Raw KPSS p-value': KPSS test p-value on raw data (optional) <br>- 'Transformed ADF p-value': ADF test p-value after transform (optional) <br>- 'Transformed KPSS p-value': KPSS test p-value after transform (optional)|

---

### Example

```python
# imports
from clustercast.datasets import load_store_sales
from clustercast import RecursiveForecaster

# load store sales data
data = load_store_sales()
print(data)
```

```profile
     ID         YM   Region         Category      Sales    exog_1    exog_2
0     1 2015-01-01  Central        Furniture    506.358  1.764052  3.248691
1     2 2015-01-01  Central  Office Supplies    996.408  0.400157 -1.223513
2     3 2015-01-01  Central       Technology     31.200  0.978738 -1.056344
3     4 2015-01-01     East        Furniture    199.004  2.240893 -2.145937
4     5 2015-01-01     East  Office Supplies    112.970  1.867558  1.730815
..   ..        ...      ...              ...        ...       ...       ...
424   8 2017-12-01    South  Office Supplies   5302.324 -0.130107  4.109248
425   9 2017-12-01    South       Technology   2910.754  0.093953  0.106819
426  10 2017-12-01     West        Furniture  14391.752  0.943046 -0.958314
427  11 2017-12-01     West  Office Supplies   9166.328 -2.739677  0.700334
428  12 2017-12-01     West       Technology   8545.118 -0.569312  0.034329

[429 rows x 7 columns]
```

```python
# show the future data for the exogenous variables (either known or forecasted)
print(future_exog)
```

```profile
     ID    exog_1     exog_2           YM
0     1  1.331587  -1.930131   2018-01-01
1     2  0.715279   2.056548   2018-01-01
2     3 -1.545400   0.457260   2018-01-01
3     4 -0.008384   0.890275   2018-01-01
4     5  0.621336  -2.273204   2018-01-01
..   ..       ...        ...          ...
283   8  2.010783   0.686925   2019-12-01
284   9 -0.096784   3.092061   2019-12-01
285  10  0.422202   1.380162   2019-12-01
286  11 -0.225462  -4.091707   2019-12-01
287  12 -0.637943   0.668934   2019-12-01

[288 rows x 4 columns]
```

```python
# create the forecasting model
model = RecursiveForecaster(
    data=data, # provide the full dataset
    endog_var='Sales', # the sales column will be forecasted
    id_var='ID', # indicates the different time series identifier column
    group_vars=['Region', 'Category'], # group features that differentiate the time series
    timestep_var='YM', # indicates the timestep column
    exog_vars=['exog_1', 'exog_2'], # indicates the exogenous features to use
    boxcox=0.5, # boxcox transformation with lambda = 0.5
    differencing=False, # do not difference the data
    include_level=False, # do not include a level feature
    include_timestep=False, # do not include timestep as a feature
    lags=12, # include lags 1 through 12
    sample_weight_halflife=12, # decay the sample weights over the period of a month
    seasonality_ordinal=[12], # include an ordinal seasonality feature
    lgbm_kwargs={'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 30, 'reg_lambda': 0.03, 'verbose':-1},
)

# fit the model
model.fit(alpha=0.10)

# make predictions out to 12 steps ahead, pass the future exog data,
# and use 500 bootstrap iterations to produce prediction intervals
forecast = model.predict(steps=12, exog_data=future_exog, bootstrap_iter=500)
print(forecast)
```

```profile
     ID         YM   Region         Category     Forecast  Forecast_0.050  Forecast_0.950  
0     1 2018-01-01  Central        Furniture  3249.188111      823.035650     5335.111385  
1     2 2018-01-01  Central  Office Supplies  2484.753879      731.983805     5578.788326  
2     3 2018-01-01  Central       Technology  3015.802614     1504.573711    13206.456765  
3     4 2018-01-01     East        Furniture  1845.889868      471.868242     5599.483712  
4     5 2018-01-01     East  Office Supplies  3785.740747     1534.934691     6936.820658  
..   ..        ...      ...              ...          ...             ...             ... 
139   8 2018-12-01    South  Office Supplies  4050.859234     1729.549101    10691.168582  
140   9 2018-12-01    South       Technology  3080.471316     1000.518659    12845.554813 
141  10 2018-12-01     West        Furniture  9342.107224     7607.681676    13520.501073 
142  11 2018-12-01     West  Office Supplies  7727.692540     3622.107959    12863.080882  
143  12 2018-12-01     West       Technology  9912.039919     5914.329473    12409.999083  

[144 rows x 7 columns]
```

---

### References

[Grouped Time Series: Hyndman, "Forecasting: Principles and Practice"](https://otexts.com/fpp3/hts.html#grouped-time-series)
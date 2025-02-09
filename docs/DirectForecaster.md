# Direct Forecaster

The `DirectForecaster` class implements direct multi-step forecasting, supporting both single-series (local) forecasting and multi-series (global) forecasting for grouped or hierarchical time series. It trains separate models for each forecast horizon, using LightGBM as the default base regressor but allowing for custom ML models. The class handles a variety of time series preprocessing techniques natively, including differencing, Box-Cox transformations, seasonality features, and lag calculations. The `DirectForecaster` provides both point forecasts and prediction intervals, with optional Conformal Quantile Regression (CQR) for state-of-the-art interval coverage. This flexible forecaster is well-suited for time series with complex patterns and multiple relevant predictors.

---

> *class* clustercast.**DirectForecaster**(*data, endog_var, id_var, timestep_var, group_vars=[], exog_vars=[], boxcox=1, differencing=False, include_level=True, include_timestep=False, lags=1, sample_weight_halflife=None, seasonality_fourier={}, seasonality_onehot=[], seasonality_ordinal=[], lgbm_kwargs={'verbose': -1}, base_regressor=None*)

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

**.fit**(*max_steps=1, alpha=None, cqr_cal_size='auto'*)
	
>Creates and fits the forecasting model up to a defined forecast horizon.
Trains separate base regressors for each lookahead timestep up to `max_steps` ahead.
Also supports prediction intervals with Conformalized Quantile Regression.

> | Argument | Description |
|---|---|
| max_steps : `int` | Maximum number of timesteps ahead to forecast. |
| alpha : `float` | Miscoverage rate for prediction intervals (e.g., 0.05 for 95% <br>intervals). If `None`, only point forecasts are produced. |
| cqr_cal_size : `str`, <br>`int`, or `float` | Size of the calibration set for CQR: <br>-`'auto'`: Automatically determine size based on data. Uses a minimum of a <br>full season's data (using the largest season) or 20% of the data. <br>-`int`: Number of time steps to use. <br>-`float`: Fraction of the total time steps to use in (0, 1).<br>-`None`, no CQR calibration is performed and standard quantile regression <br>is used. |

**.predict**(*steps=1*)
	
>Generates forecasts and prediction intervals, if applicable.
Makes predictions using the trained models for each lookahead timestep up to the specified
number of steps ahead. If prediction intervals were enabled during fitting, also generates
prediction intervals.

> | Argument | Description |
|---|---|
| steps : `int` | Number of steps ahead to forecast. If this value is greater than the maximum <br>number of timesteps the forecaster was trained on during the fit method, the <br>fit method is called again with the lengthened forecast horizon. |

> | Returns | Description |
|---|---|
| forecasts : `pd.DataFrame` | DataFrame containing the forecasts with columns: <br>- ID variable <br>- Timestep variable <br>- Group variables (if applicable) <br>- 'Forecast': Point forecasts <br>- 'Forecast_{alpha/2}' and 'Forecast_{1-alpha/2}': Lower and upper <br>prediction interval bounds (if alpha was specified during fitting). |

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
# show the training data
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
# import the model class
from clustercast import DirectForecaster

# create the forecasting model
model = DirectForecaster(
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

# fit the model with a maximum forecast horizon of 12 steps and a 90% prediction interval
model.fit(max_steps=12, alpha=0.10, cqr_cal_size='auto')

# make predictions out to 12 steps ahead and show the results
forecast = model.predict(steps=12)
print(forecast)
```

```profile
     ID         YM   Region         Category     Forecast  Forecast_0.050  Forecast_0.950
0     1 2018-01-01  Central        Furniture  3249.188111     1066.250237    13043.484485 
1     2 2018-01-01  Central  Office Supplies  2484.753879      547.091754     7594.467308 
2     3 2018-01-01  Central       Technology  3015.802614      215.583127    26040.987499
3     4 2018-01-01     East        Furniture  1845.889868     1236.899833     7945.113132 
4     5 2018-01-01     East  Office Supplies  3785.740747     2163.662184    12222.113959  
..   ..        ...      ...              ...          ...             ...             ...
139   8 2018-12-01    South  Office Supplies  5828.376613     1727.170471     6891.555979
140   9 2018-12-01    South       Technology  3745.596868     1424.952497    14552.931506 
141  10 2018-12-01     West        Furniture  8873.877963     1040.328678    20655.901796
142  11 2018-12-01     West  Office Supplies  7705.059564     1176.046296    15412.515219  
143  12 2018-12-01     West       Technology  6902.883468     1191.380446    19659.924231   

[144 rows x 7 columns]
```

---

### References

[Romano et. al., "Conformalized Quantile Regression"](https://arxiv.org/abs/1905.03222)

[Grouped Time Series: Hyndman, "Forecasting: Principles and Practice"](https://otexts.com/fpp3/hts.html#grouped-time-series)
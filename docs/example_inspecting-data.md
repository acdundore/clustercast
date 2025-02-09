# Inspecting Transformed Data

In this example, we will use the same store sales dataset from other examples.
Instead of making predictions, we will use some built-in properties of the forecasting models to inspect the transformed training data.
It is good practice to manually inspect the transformed dataset before inferencing to ensure it aligns with user expectations.
Although this model uses the `DirectForecaster` class, the same process can be completed for `RecursiveForecaster`.

---

## Data Preparation

```python
# import store sales data
data = pd.read_csv('store_sales_grouped.csv', parse_dates=['YM'])
print(data)

# keep only certain data for training
data_train = data.loc[
    data['YM'] < dt.datetime(year=2018, month=1, day=1)
]
```

```profile
     ID         YM   Region    Category     Sales
0     1 2015-01-01  Central   Furniture   506.358
1     1 2015-02-01  Central   Furniture   439.310
2     1 2015-03-01  Central   Furniture  3639.290
3     1 2015-04-01  Central   Furniture  1468.218
4     1 2015-05-01  Central   Furniture  2304.382
..   ..        ...      ...         ...       ...
568  12 2018-08-01     West  Technology  6230.788
569  12 2018-09-01     West  Technology  5045.440
570  12 2018-10-01     West  Technology  4651.807
571  12 2018-11-01     West  Technology  7584.580
572  12 2018-12-01     West  Technology  8064.524

[573 rows x 5 columns]
```

---

## Direct Forecaster

First, we can create the direct forecaster model.
We will use the following parameters.
Although these are likely not optimal, they will be useful for explaining how the feature transformations work.

- A Box-Cox transform parameter of 0.5
- Differencing
- Inclusion of a level feature
- 3 lag features
- Two ordinal seasonality features: one for 12 months (yearly), and one for 3 months (quarterly)

```python
# create the forecasting model
model = DirectForecaster(
    data=data_train,
    endog_var='Sales',
    id_var='ID',
    group_vars=['Region', 'Category'],
    boxcox=0.5,
    differencing=True,
    include_level=True,
    timestep_var='YM',
    lags=3,
    seasonality_ordinal=[12, 3],
)

# fit the model; data transformation is performed within the fit method
model.fit(max_steps=3)
```

First, let's take a look at the timestep inferred by the model.
The forecasting classes work with both datetime and non-datetime timesteps, but it is good practice to inspect.
In this case, we are working with monthly data.
It appears that the timestep delta inferred by the model is correct.

```python
# show the inferred timestep
print(model._inferred_timestep)
```

```profile
<DateOffset: months=1>
```

Next, let's look at the transformed data.
We will sort it by series ID and timestep in order to make it easier to understand, since there are multiple time series in the dataset.

There are several things to note about the transformed data:

- There are multiple columns that begin with an underscore. These will not be used for training, but are included to make it easier for the user to double-check the transformations. In this case, the `Sales` column is put through a Box-Cox transformation (`_endog_boxcox` column). Then, differencing is applied (`_endog_differenced` column). The final transformed endogenous variable is stored in the `endog` column, which will be used for training.
- The endogenous variable after being Box-Cox transformed is stored in the `endog_level` column, as specified by the `include_level` argument.
- There are 3 lag variables included in the transformed data: `endog_lag_1`, `endog_lag_2`, and `endog_lag_3`. These are lags of the final `endog` column.
- Because we specified for the `DirectForecaster` to fit 3 lookahead models (`max_steps`=3), there were 3 target columns calculated: `endog_lookahead_1`, `endog_lookahead_2`, and `endog_lookahead_3`.
- There are two seasonality columns: `season_ordinal_p12` and `season_ordinal_p3`.
- There are onehot-encoded columns for both `Region` and `Category` (the two grouping variables).


```python
# display the transformed data
print(model._data_trans.sort_values(by=['ID', 'YM']))
```

```profile
            YM  ID   Region    Category     Sales      endog  _endog_boxcox  endog_level  _endog_differenced  endog_lookahead_1  endog_lookahead_2  endog_lookahead_3  endog_lag_1  endog_lag_2  endog_lag_3  season_ordinal_p12  season_ordinal_p3  Region_Central  Region_East  Region_South  Region_West  Category_Furniture  Category_Office Supplies  Category_Technology
0   2015-01-01   1  Central   Furniture   506.358        NaN      43.049218    43.049218                 NaN          -3.082088          75.620414          31.611542          NaN          NaN          NaN                   1                  1               1            0             0            0                   1                         0                    0
12  2015-02-01   1  Central   Furniture   439.310  -3.082088      39.967130    39.967130           -3.082088          78.702502          34.693629          54.061657          NaN          NaN          NaN                   2                  2               1            0             0            0                   1                         0                    0
24  2015-03-01   1  Central   Furniture  3639.290  78.702502     118.669632   118.669632           78.702502         -44.008872         -24.640844           9.307329    -3.082088          NaN          NaN                   3                  3               1            0             0            0                   1                         0                    0
36  2015-04-01   1  Central   Furniture  1468.218 -44.008872      74.660759    74.660759          -44.008872          19.368028          53.316202          16.867761    78.702502    -3.082088          NaN                   4                  1               1            0             0            0                   1                         0                    0
48  2015-05-01   1  Central   Furniture  2304.382  19.368028      94.028787    94.028787           19.368028          33.948174          -2.500268         -33.884375   -44.008872    78.702502    -3.082088                   5                  2               1            0             0            0                   1                         0                    0
..         ...  ..      ...         ...       ...        ...            ...          ...                 ...                ...                ...                ...          ...          ...          ...                 ...                ...             ...          ...           ...          ...                 ...                       ...                  ...
383 2017-08-01  12     West  Technology  4075.004 -29.493989     125.687180   125.687180          -29.493989          52.117225         -14.274518          18.847880   -39.848737    60.500935    63.291658                   8                  2               0            0             0            1                   0                         0                    1
395 2017-09-01  12     West  Technology  8081.406  52.117225     177.804405   177.804405           52.117225         -66.391742         -33.269344           5.086028   -29.493989   -39.848737    60.500935                   9                  3               0            0             0            1                   0                         0                    1
407 2017-10-01  12     West  Technology  3214.608 -66.391742     111.412662   111.412662          -66.391742          33.122398          71.477770                NaN    52.117225   -29.493989   -39.848737                  10                  1               0            0             0            1                   0                         0                    1
419 2017-11-01  12     West  Technology  5367.131  33.122398     144.535061   144.535061           33.122398          38.355372                NaN                NaN   -66.391742    52.117225   -29.493989                  11                  2               0            0             0            1                   0                         0                    1
431 2017-12-01  12     West  Technology  8545.118  38.355372     182.890432   182.890432           38.355372                NaN                NaN                NaN    33.122398   -66.391742    52.117225                  12                  3               0            0             0            1                   0                         0                    1

[432 rows x 24 columns]
```

As previously mentioned, not all columns in the transformed data will be used for training.
If we want to see the columns that will be used for training in the X data, we can do the following.
Note that the timestep variable, series IDs, original group variables, and intermediate endogenous variable transformations are not included.

```python
# display the columns that will be used for training
for c in model._X_cols:
    print(c)
```

```profile
endog
endog_level
endog_lag_1
endog_lag_2
endog_lag_3
season_ordinal_p12
season_ordinal_p3
Region_Central
Region_East
Region_South
Region_West
Category_Furniture
Category_Office Supplies
Category_Technology
```
# Datasets

`clustercast` includes two example datasets that can be imported and used to experiment with the forecasting classes.

---

## Airline Passengers Dataset

> *function* clustercast.datasets.**load_airline_passengers**()

This function returns the well-known airline passengers dataset as a pandas dataframe.
The airline passengers dataset records monthly airline passengers over a period of many years.
This dataset is great for learning about time series forecasting because it is non-stationary and exhibits strong seasonality.

### Example

```python
from clustercast.datasets import load_airline_passengers

# load in the dataset
data = load_airline_passengers()
print(data)
```

```profile
            YM  Passengers
0   1949-01-01         112
1   1949-02-01         118
2   1949-03-01         132
3   1949-04-01         129
4   1949-05-01         121
..         ...         ...
139 1960-08-01         606
140 1960-09-01         508
141 1960-10-01         461
142 1960-11-01         390
143 1960-12-01         432

[144 rows x 2 columns]
```

---

## Store Sales Dataset

> *function* clustercast.datasets.**load_store_sales**()

This function returns a superstore sales dataset as a pandas dataframe.
This dataset is a modified version of an open dataset found on Kaggle, linked in the references below.
The store sales dataset is a good example for global forecasting of grouped time series, as there are 12
closely related time series that have shared and overlapping attributes (store region and product category).

### Example

```python
from clustercast.datasets import load_store_sales

# load in the dataset
data = load_store_sales()
print(data)
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

### References

[Airline Passengers Dataset (Kaggle)](https://www.kaggle.com/datasets/erogluegemen/airline-passengers)

[Superstore Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting?select=train.csv)

[Grouped Time Series: Hyndman, "Forecasting: Principles and Practice"](https://otexts.com/fpp3/hts.html#grouped-time-series)
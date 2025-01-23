import pandas as pd 
import numpy as np 
import datetime as dt 
from lightgbm import LGBMRegressor 

from collections import Counter
from itertools import product
import re 
import warnings 


class _GroupForecaster():
    def __init__(self, data, endog_var, id_var, group_vars, timestep_var, exog_vars=[], boxcox=1, differencing=False, lags=1, seasonality_fourier={}, seasonality_onehot=[], seasonality_ordinal=[]):
        self.data = data 
        self._data_trans = None
        self.endog_var = endog_var 
        self.id_var = id_var 
        self.group_vars = group_vars 
        self.exog_vars = exog_vars 
        self.timestep_var = timestep_var
        self.boxcox = boxcox
        self.differencing = differencing
        self.seasonality_fourier = seasonality_fourier
        self.seasonality_onehot = seasonality_onehot
        self.seasonality_ordinal = seasonality_ordinal

        if type(lags) == list:
            self.lags = lags
        else:
            self.lags = list(range(1, lags + 1))

        # infer the timestep and get all timestep values in range
        self._infer_timestep()
        self._all_timesteps = self._get_all_timesteps(self.data)


    def _infer_timestep(self):
        # get all unique timestep values
        unique_timesteps = self.data[self.timestep_var].sort_values().unique()

        # get the difference between all timesteps, then get the most common timestep
        deltas = (unique_timesteps - np.roll(unique_timesteps, shift=1))[1:]
        timestep_mode = Counter(deltas).most_common(1)[0][0]

        # check if the timestep mode is numeric; if not, treat it as a datetime/timestamp
        if pd.api.types.is_numeric_dtype(timestep_mode):
            self._inferred_timestep = timestep_mode 
        else:
            # define bounds for different common timesteps in terms of seconds
            timestep_mode_in_sec = timestep_mode.total_seconds()
            yearly_bounds = [363 * 24 * 60 * 60, 366 * 24 * 60 * 60]
            quarterly_bounds = [3 * 29 * 24 * 60 * 60, 3 * 32 * 24 * 60 * 60]
            monthly_bounds = [29 * 24 * 60 * 60, 32 * 24 * 60 * 60]
            weekly_bounds = [0.99 * 7 * 24 * 60 * 60, 1.01 * 7 * 24 * 60 * 60]
            daily_bounds = [.99 * 24 * 60 * 60, 1.01 * 24 * 60 * 60]
            hourly_bounds = [.99 * 60 * 60, 1.01 * 60 * 60]
            minute_bounds = [.99 * 60, 1.01 * 60]

            # infer the timestep using a date offset
            if yearly_bounds[0] < timestep_mode_in_sec < yearly_bounds[1]:
                self._inferred_timestep = pd.tseries.offsets.DateOffset(years=1)
            elif quarterly_bounds[0] < timestep_mode_in_sec < quarterly_bounds[1]:
                self._inferred_timestep = pd.tseries.offsets.DateOffset(months=3)
            elif monthly_bounds[0] < timestep_mode_in_sec < monthly_bounds[1]:
                self._inferred_timestep = pd.tseries.offsets.DateOffset(months=1)
            elif weekly_bounds[0] < timestep_mode_in_sec < weekly_bounds[1]:
                self._inferred_timestep = pd.tseries.offsets.DateOffset(weeks=1)
            elif daily_bounds[0] < timestep_mode_in_sec < daily_bounds[1]:
                self._inferred_timestep = pd.tseries.offsets.DateOffset(days=1)
            elif hourly_bounds[0] < timestep_mode_in_sec < hourly_bounds[1]:
                self._inferred_timestep = pd.tseries.offsets.DateOffset(hours=1)
            elif minute_bounds[0] < timestep_mode_in_sec < minute_bounds[1]:
                self._inferred_timestep = pd.tseries.offsets.DateOffset(minutes=1)
            else:
                # if timestep is less than munitely, treat it as decimal seconds
                self._inferred_timestep = pd.tseries.offsets.DateOffset(seconds=timestep_mode_in_sec)


    def _get_all_timesteps(self, data):
        # get all unique timestep values
        unique_timesteps = data[self.timestep_var].sort_values().unique()

        # fill in any missing timesteps and store to list
        max_timestep = max(unique_timesteps)
        all_timesteps = []
        current_timestep = min(unique_timesteps)
        while current_timestep <= max_timestep:
            all_timesteps.append(current_timestep)
            current_timestep += self._inferred_timestep

        return all_timesteps


    def _transform_data(self, data, all_timesteps, lookaheads=1):
        # create a dataframe with the group values for all IDs
        id_group_key = data[[self.id_var] + self.group_vars].drop_duplicates()

        # create new dataframe that includes all dates for all time series IDs
        all_date_id_combos = list(product(all_timesteps, data[self.id_var].unique()))
        data_trans = pd.DataFrame(all_date_id_combos, columns=[self.timestep_var, self.id_var])

        # join the group values onto the newly filled data
        data_trans = pd.merge(left=data_trans, right=id_group_key, on=self.id_var, how='left')
        
        # join the endog and exog variabes onto the newly filled data
        data_trans = pd.merge(
            left=data_trans, 
            right=data[[self.id_var, self.endog_var, self.timestep_var] + self.exog_vars], 
            on=[self.id_var, self.timestep_var], 
            how='left'
        )

        # creating the endog column with the raw endogenous variable values
        data_trans['endog'] = data_trans[self.endog_var]

        # applying boxcox transformation if necessary:
        if self.boxcox == 0:
            data_trans['_endog_boxcox'] = np.log(data_trans['endog'] + 1)
        else:
            data_trans['_endog_boxcox'] = ((data_trans['endog'] + 1) ** self.boxcox - 1) / self.boxcox

        # create a new transformed version of the data that will contain all generated features 
        data_trans['endog'] = data_trans['_endog_boxcox']

        # difference the data if necessary
        if self.differencing:
            # differencing the data
            data_trans['_endog_differenced'] = data_trans['endog'] - data_trans.groupby(self.id_var)['endog'].shift(1)
            # overwrite the final endog variable if necessary
            data_trans['endog'] = data_trans['_endog_differenced']

        # generate lookahead targets
        for lookahead in range(1, lookaheads + 1):
            # get the future endog for the lookahead
            if self.differencing:
                data_trans[f'endog_lookahead_{str(lookahead)}'] = data_trans.groupby(self.id_var)['_endog_boxcox'].shift(-lookahead) - data_trans['_endog_boxcox']
            else:
                data_trans[f'endog_lookahead_{str(lookahead)}'] = data_trans.groupby(self.id_var)['_endog_boxcox'].shift(-lookahead)

        # generate lags
        for lag in self.lags:
            data_trans[f'endog_lag_{str(lag).zfill(len(str(max(self.lags))))}'] = data_trans.groupby(self.id_var)['endog'].shift(lag)

        # creating placeholder dataframe for seasonality features
        seasonality_features = pd.DataFrame({self.timestep_var: all_timesteps})
        n = np.arange(start=0, stop=len(all_timesteps), step=1)

        # calculating fourier seasonality features if necessary
        for period, n_terms in self.seasonality_fourier.items():
            for t in range(1, n_terms + 1):
                seasonality_features[f'season_fourier_p{period}_h{t}_sin'] = np.sin(n * (t * 2 * np.pi / period))
                seasonality_features[f'season_fourier_p{period}_h{t}_cos'] = np.cos(n * (t * 2 * np.pi / period))

        # calculating onehot seasonality features if necessary
        for period in self.seasonality_onehot:
            for i in range(1, period + 1):
                seasonality_features[f'season_onehot_p{period}_{str(i).zfill(len(str(period)))}'] = (n % period == i - 1).astype(int)

        # calculating ordinal seasonality features if necessary
        for period in self.seasonality_ordinal:
            for i in range(1, period + 1):
                seasonality_features[f'season_ordinal_p{period}'] = (n % period + 1).astype(int)

        # merging seasonality features with the transformed data
        data_trans = pd.merge(left=data_trans, right=seasonality_features, how='left', on=self.timestep_var)

        # onehot encode the grouping features
        for group in self.group_vars:
            for group_val in list(data_trans[group].dropna().unique()):
                data_trans[f'{group}_{group_val}'] = (data_trans[group] == group_val).astype(int)

        # store a list of all training features
        target_cols = [c for c in data_trans if 'endog_lookahead_' in c]
        training_cols = [c for c in data_trans if c not in [self.id_var, self.timestep_var, self.endog_var] + self.group_vars + target_cols and not re.fullmatch(r'^_endog_.*', c)]
        self._X_cols = training_cols

        return data_trans


    def _reverse_transform_preds(self, y_pred, data_pred):
        # undo any differencing
        if self.differencing:
            y_pred = data_pred['_endog_boxcox'] + y_pred 

        # reverse boxcox transformation
        if self.boxcox == 0:
            y_pred = np.exp(y_pred) - 1
        else:
            y_pred = (y_pred * self.boxcox + 1) ** (1 / self.boxcox) - 1

        return y_pred
        

class DirectForecaster(_GroupForecaster):
    def __init__(self, data, endog_var, id_var, group_vars, timestep_var, exog_vars=[], boxcox=1, differencing=False, lags=1, seasonality_fourier={}, seasonality_onehot=[], seasonality_ordinal=[]):
        super().__init__(
            data=data,
            endog_var=endog_var, 
            id_var=id_var, 
            group_vars=group_vars, 
            timestep_var=timestep_var, 
            exog_vars=exog_vars, 
            boxcox=boxcox, 
            differencing=differencing, 
            lags=lags, 
            seasonality_fourier=seasonality_fourier, 
            seasonality_onehot=seasonality_onehot, 
            seasonality_ordinal=seasonality_ordinal
        )

    def fit(self, max_steps=1, alpha=None):
        # transform the data
        self._data_trans = self._transform_data(data=self.data, all_timesteps=self._all_timesteps, lookaheads=max_steps)

        # calculate alpha
        if type(alpha) == float:
            self._alphas = [alpha / 2, 1 - alpha / 2]
        elif alpha == None:
            self._alphas = []
        else:
            self._alphas = alpha

        # create lists to store the trained models
        self._predictors = []
        self._pi_predictors = []

        # make a prediction for each lookahead
        for step in range(1, max_steps + 1):
            # create predictor object
            current_predictor = LGBMRegressor(verbose=-1, objective='quantile', alpha=0.5)
            current_pi_predictors = {}
            for a in self._alphas:
                current_pi_predictors[a] = LGBMRegressor(verbose=-1, objective='quantile', alpha=a)

            # define the target for the current lookahead and drop any rows with blank targets
            target = f'endog_lookahead_{str(step)}'
            data_train = self._data_trans.dropna(subset=target)

            # get the X and y training data
            X_train = data_train[self._X_cols]
            y_train = data_train[target]

            # train the model and store it
            current_predictor.fit(X_train, y_train)
            self._predictors.append(current_predictor)

            # train any prediction interval models and store them
            for a in self._alphas:
                current_pi_predictors[a].fit(X_train, y_train)

            self._pi_predictors.append(current_pi_predictors)

    def predict(self, steps=1):
        # instantiate a list to store the predictions
        pred_data_list = []

        # make a prediction for each lookahead
        for step in range(1, steps + 1):
            # define the prediction data
            data_pred = self._data_trans.loc[self._data_trans[self.timestep_var] == max(self._data_trans[self.timestep_var])]

            # get the X data for prediction
            X_pred = data_pred[self._X_cols]

            # train the model and make predictions
            y_pred = self._predictors[step - 1].predict(X_pred)
            pi_preds = []
            for p in self._pi_predictors[step - 1].values():
                pi_preds.append(p.predict(X_pred))

            # reverse transform the predictions as necessary
            y_pred = self._reverse_transform_preds(y_pred, data_pred)
            for i in range(len(pi_preds)):
                pi_preds[i] = self._reverse_transform_preds(pi_preds[i], data_pred)

            # store the prediction data
            current_pred_data = data_pred[[self.id_var, self.timestep_var] + self.group_vars].copy()
            current_pred_data['Forecast'] = y_pred 
            for a, pi_pred in zip(self._alphas, pi_preds):
                current_pred_data[f'Forecast_{a:.3f}'] = pi_pred

            current_pred_data[self.timestep_var] += step * self._inferred_timestep
            pred_data_list.append(current_pred_data)

        # transform the predictions to a single dataframe
        prediction_data = pd.concat(pred_data_list, axis=0).reset_index(drop=True)

        return prediction_data
    

class RecursiveForecaster(_GroupForecaster):
    def __init__(self, data, endog_var, id_var, group_vars, timestep_var, exog_vars=[], boxcox=1, differencing=False, lags=1, seasonality_fourier={}, seasonality_onehot=[], seasonality_ordinal=[]):
        super().__init__(
            data=data,
            endog_var=endog_var, 
            id_var=id_var, 
            group_vars=group_vars, 
            timestep_var=timestep_var, 
            exog_vars=exog_vars, 
            boxcox=boxcox, 
            differencing=differencing, 
            lags=lags, 
            seasonality_fourier=seasonality_fourier, 
            seasonality_onehot=seasonality_onehot, 
            seasonality_ordinal=seasonality_ordinal
        )

    def fit(self, alpha=None):
        # transform the data
        self._data_trans = self._transform_data(data=self.data, all_timesteps=self._all_timesteps, lookaheads=1)

        # calculate alpha
        if type(alpha) == float:
            self._alphas = [alpha / 2, 1 - alpha / 2]
        elif alpha == None:
            self._alphas = []
        else:
            self._alphas = alpha

        # create predictor object
        self._predictor = LGBMRegressor(verbose=-1, objective='quantile', alpha=0.5)

        # define the target for the current lookahead and drop any rows with blank targets
        target = f'endog_lookahead_1'
        data_train = self._data_trans.dropna(subset=target)

        # get the X and y training data
        X_train = data_train[self._X_cols]
        y_train = data_train[target]

        # train the model and store it
        self._predictor.fit(X_train, y_train)

        # store the in-sample residuals for each time series in a dictionary
        in_sample_residuals = y_train - self._predictor.predict(X_train)
        in_sample_residuals_df = pd.DataFrame({'ID': data_train[self.id_var], 'Residuals': in_sample_residuals})
        self._in_sample_residuals_dict = {}
        for id in data_train[self.id_var].unique():
            self._in_sample_residuals_dict[id] = np.array(in_sample_residuals_df.loc[in_sample_residuals_df['ID'] == id, 'Residuals'])


    def predict(self, steps=1, exog=None, bootstrap_iter=100):
        # instantiate a list to store the predictions
        pred_data_list = []

        # check to see if the model needs exogenous variables to be passed
        if len(self.exog_vars) > 0 and exog is None:
                warnings.warn('The model was fit on exogenous features, but none were passed to the predict method.', UserWarning)

        # make a prediction for each lookahead
        for step in range(1, steps + 1):
            # define the prediction data
            if step == 1:
                data = self.data
            if step != 1:
                data = pd.concat([data, current_pred_data.rename(columns={'Forecast': self.endog_var})], axis=0)
            
            all_timesteps = self._get_all_timesteps(data)
            self._data_trans = self._transform_data(data=data, all_timesteps=all_timesteps, lookaheads=1)

            data_pred = self._data_trans.loc[self._data_trans[self.timestep_var] == max(self._data_trans[self.timestep_var])]

            # check to see if exogenous variables were passed to the predict method
            if exog is not None:
                # merge the new exogenous variables onto the prediction dataframe, and change the old exogenous feature name for removal
                data_pred = pd.merge(left=data_pred, right=exog, on=[self.timestep_var, self.id_var], how='left', suffixes=(None, '__FUTURE__'))

                # for each exogenous variable, infill it with the future data if it exists
                for exog_var in self.exog_vars:
                    future_exog_var = f"{exog_var}__FUTURE__"
                    if future_exog_var in data_pred.columns:
                        # where the original variable is null, replace with future values
                        data_pred[exog_var] = data_pred[exog_var].fillna(data_pred[future_exog_var])

            # get the X data for prediction
            X_pred = data_pred[self._X_cols]

            # train the model and make predictions
            y_pred = self._predictor.predict(X_pred)

            # reverse transform the predictions as necessary
            y_pred = self._reverse_transform_preds(y_pred, data_pred)

            # store the prediction data
            current_pred_data = data_pred[[self.id_var, self.timestep_var] + self.group_vars].copy()
            current_pred_data['Forecast'] = y_pred 
            current_pred_data[self.timestep_var] += self._inferred_timestep
            pred_data_list.append(current_pred_data)

        # transform the predictions to a single dataframe
        prediction_data = pd.concat(pred_data_list, axis=0).reset_index(drop=True)

        # perform bootstrapping if prediction intervals need to be generated
        if len(self._alphas) > 0:
            # create a list to store the bootstrapped prediction data
            bootstrap_pred_data_list = []

            # repeat the bootstrapping for the specified number of iterations
            for b_iter in range(bootstrap_iter):
                # make a prediction for each lookahead
                for step in range(1, steps + 1):
                    # define the prediction data
                    if step == 1:
                        data = self.data
                    if step != 1:
                        data = pd.concat([data, current_pred_data.rename(columns={'Forecast': self.endog_var})], axis=0)
                    
                    # get all timesteps and transform the data
                    all_timesteps = self._get_all_timesteps(data)
                    data_trans = self._transform_data(data=data, all_timesteps=all_timesteps, lookaheads=1)

                    # get the most recent batch of data for prediction
                    data_pred = data_trans.loc[data_trans[self.timestep_var] == max(data_trans[self.timestep_var])]

                    # check to see if exogenous variables were passed to the predict method
                    if exog is not None:
                        # merge the new exogenous variables onto the prediction dataframe, and change the old exogenous feature name for removal
                        data_pred = pd.merge(left=data_pred, right=exog, on=[self.timestep_var, self.id_var], how='left', suffixes=(None, '__FUTURE__'))

                        # for each exogenous variable, infill it with the future data if it exists
                        for exog_var in self.exog_vars:
                            future_exog_var = f"{exog_var}__FUTURE__"
                            if future_exog_var in data_pred.columns:
                                # where the original variable is null, replace with future values
                                data_pred[exog_var] = data_pred[exog_var].fillna(data_pred[future_exog_var])

                    # get the X data for prediction
                    X_pred = data_pred[self._X_cols]

                    # train the model and make predictions
                    y_pred = self._predictor.predict(X_pred)

                    # add on randomly sampled residuals from the fitted values (from respective time series)
                    for i, id in enumerate(data_pred[self.id_var]):
                        y_pred[i] += np.random.choice(a=self._in_sample_residuals_dict[id], size=1)[0]

                    # reverse transform the predictions as necessary
                    y_pred = self._reverse_transform_preds(y_pred, data_pred)

                    # store the prediction data
                    current_pred_data = data_pred[[self.id_var, self.timestep_var] + self.group_vars].copy()
                    current_pred_data['Forecast'] = y_pred 
                    current_pred_data[self.timestep_var] += self._inferred_timestep
                    bootstrap_pred_data_list.append(current_pred_data)

            # transform the predictions to a single dataframe
            bootstrap_prediction_data = pd.concat(bootstrap_pred_data_list, axis=0).reset_index(drop=True)

        # calculate percentiles from the bootstrapped predictions
        for a in self._alphas:
            current_pi = bootstrap_prediction_data[[self.id_var, self.timestep_var, 'Forecast']].groupby([self.id_var, self.timestep_var]).aggregate({'Forecast': lambda x: x.quantile(a)}).reset_index()
            current_pi = current_pi.rename(columns={'Forecast': f'Forecast_{a:.3f}'})
            prediction_data = pd.merge(left=prediction_data, right=current_pi, on=[self.id_var, self.timestep_var], how='left')

        return prediction_data
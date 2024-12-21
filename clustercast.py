class grouped_forecaster():
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

        # fill in any missing timesteps and store to list
        max_timestep = max(unique_timesteps)
        self._all_timesteps = []
        current_timestep = min(unique_timesteps)
        while current_timestep <= max_timestep:
            self._all_timesteps.append(current_timestep)
            current_timestep += self._inferred_timestep


    def _transform_data(self, lookaheads=1):
        # create a dataframe with the group values for all IDs
        id_group_key = self.data[[self.id_var] + self.group_vars].drop_duplicates()

        # create new dataframe that includes all dates for all time series IDs
        all_date_id_combos = list(product(self._all_timesteps, self.data[self.id_var].unique()))
        filled_data = pd.DataFrame(all_date_id_combos, columns=[self.timestep_var, self.id_var])

        # join the group values onto the newly filled data
        filled_data = pd.merge(left=filled_data, right=id_group_key, on=self.id_var, how='left')
        
        # join the endog and exog variabes onto the newly filled data
        filled_data = pd.merge(
            left=filled_data, 
            right=self.data[[self.id_var, self.endog_var, self.timestep_var] + self.exog_vars], 
            on=[self.id_var, self.timestep_var], 
            how='left'
        )

        # create a wide form table containing the endog variable for all groups
        endog_table = filled_data.pivot(index=self.timestep_var, columns=self.id_var, values=self.endog_var) #.reset_index()

        # create a new transformed version of the data that will contain all generated features 
        merge_cols = [self.timestep_var, self.id_var]
        longform_endog = endog_table.reset_index().melt(id_vars=self.timestep_var, value_name='_endog_standard')
        self._data_trans = pd.merge(left=filled_data, right=longform_endog, on=merge_cols, how='left')

        # applying boxcox transformation if necessary:
        if self.boxcox == 0:
            endog_boxcox_table = np.log(endog_table + 1)
        else:
            endog_boxcox_table = ((endog_table + 1) ** self.boxcox - 1) / self.boxcox

        # create a new transformed version of the data that will contain all generated features 
        longform_endog = endog_boxcox_table.reset_index().melt(id_vars=self.timestep_var, value_name='_endog_boxcox')
        self._data_trans = pd.merge(left=self._data_trans, right=longform_endog, on=merge_cols, how='left')
        self._data_trans['endog'] = self._data_trans['_endog_boxcox']

        # difference the data if necessary
        if self.differencing:
            # differencing the data
            endog_diff_table = endog_boxcox_table - endog_boxcox_table.shift(1)

            # create a new transformed version of the data that will contain all generated features 
            longform_endog = endog_diff_table.reset_index().reset_index().melt(id_vars=self.timestep_var, value_name='_endog_differenced')
            self._data_trans = pd.merge(left=self._data_trans, right=longform_endog, on=merge_cols, how='left')

            # overwrite the final endog variable if necessary
            self._data_trans['endog'] = self._data_trans['_endog_differenced']

        # generate lookahead targets
        for lookahead in range(1, lookaheads + 1):
            # get the future endog for the lookahead
            if self.differencing:
                lookahead_endog = (endog_boxcox_table.shift(-lookahead) - endog_boxcox_table).reset_index()
            else:
                lookahead_endog = endog_boxcox_table.shift(-lookahead).reset_index()

            # melt the lookaheads back into long form and rename the timestep variable so it is not multilevel
            longform_lookahead = lookahead_endog.melt(id_vars=self.timestep_var, value_name=f'endog_lookahead_{str(lookahead)}')

            # merge this onto the transformed data
            self._data_trans = pd.merge(left=self._data_trans, right=longform_lookahead, on=merge_cols, how='left')
        
        # generate lags
        for lag in self.lags:
            # lag the endog variable in wide form
            lagged_endog = endog_boxcox_table.shift(lag).reset_index()

            # melt the lagged endog back to long form and rename the timestep variable so it is not multilevel
            longform_lag = lagged_endog.melt(id_vars=self.timestep_var, value_name=f'endog_lag_{str(lag).zfill(len(str(max(self.lags))))}')

            # merge this onto the transformed data
            self._data_trans = pd.merge(left=self._data_trans, right=longform_lag, on=merge_cols, how='left')

        # creating placeholder dataframe for seasonality features
        seasonality_features = pd.DataFrame({self.timestep_var: self._all_timesteps})
        n = np.arange(start=0, stop=len(self._all_timesteps), step=1)

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
        self._data_trans = pd.merge(left=self._data_trans, right=seasonality_features, how='left', on=self.timestep_var)

        # onehot encode the grouping features
        for group in self.group_vars:
            for group_val in list(self._data_trans[group].dropna().unique()):
                self._data_trans[f'{group}_{group_val}'] = (self._data_trans[group] == group_val).astype(int)

        # store a list of all training features
        target_cols = [c for c in self._data_trans if 'endog_lookahead_' in c]
        training_cols = [c for c in self._data_trans if c not in [self.id_var, self.timestep_var, self.endog_var] + self.group_vars + target_cols and not re.fullmatch(r'^_endog_.*', c)]
        self._X_cols = training_cols

    def forecast_direct(self, steps=1, alphas=[]):
        # transform the data
        self._transform_data(lookaheads=steps)

        # instantiate a list to store the predictions
        pred_list = []

        # make a prediction for each lookahead
        for step in range(1, steps + 1):
            # create predictor object
            predictor = LGBMRegressor(verbose=-1)
            pi_predictors = []
            for a in alphas:
                pi_predictors.append(LGBMRegressor(verbose=-1, objective='quantile', alpha=a))

            # define the target for the current lookahead and drop any rows with blank targets
            target = f'endog_lookahead_{str(step)}'
            data_train = self._data_trans.dropna(subset=target)

            # define the prediction data
            data_pred = self._data_trans.loc[self._data_trans[self.timestep_var] == max(self._data_trans[self.timestep_var])]

            # separate into X and y data
            X_train = data_train[self._X_cols]
            y_train = data_train[target]
            X_pred = data_pred[self._X_cols]

            # train the model and make predictions
            predictor.fit(X_train, y_train)
            y_pred = predictor.predict(X_pred)
            pi_preds = []
            for p in pi_predictors:
                p.fit(X_train, y_train)
                pi_preds.append(p.predict(X_pred))

            # reverse transform the predictions as necessary
            if self.differencing:
                y_pred = data_pred['_endog_boxcox'] + y_pred 
                for i in range(len(pi_preds)):
                    pi_preds[i] = data_pred['_endog_boxcox'] + pi_preds[i]

            if self.boxcox == 0:
                y_pred = np.exp(y_pred) - 1
                for i in range(len(pi_preds)):
                    pi_preds[i] = np.exp(pi_preds[i]) - 1
            else:
                y_pred = (y_pred * self.boxcox + 1) ** (1 / self.boxcox) - 1
                for i in range(len(pi_preds)):
                    pi_preds[i] = (pi_preds[i] * self.boxcox + 1) ** (1 / self.boxcox) - 1

            # store the prediction data
            prediction_data = data_pred[[self.timestep_var] + self.group_vars].copy()
            prediction_data['Forecast'] = y_pred 
            for a, pi_pred in zip(alphas, pi_preds):
                prediction_data[f'Forecast_{a:.3f}'] = pi_pred
            prediction_data[self.timestep_var] += step * self._inferred_timestep
            pred_list.append(prediction_data)

        # transform the predictions to a single dataframe
        prediction_data = pd.concat(pred_list, axis=0)

        return prediction_data
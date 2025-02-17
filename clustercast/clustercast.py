import pandas as pd 
import numpy as np 
import datetime as dt 
from lightgbm import LGBMRegressor 
from statsmodels.tsa.stattools import adfuller, kpss

from collections import Counter
from itertools import product
import re 
import warnings 


class _GroupForecaster():
    def __init__(self, data, endog_var, id_var, timestep_var, group_vars=[], exog_vars=[], boxcox=1, differencing=False, include_level=True, include_timestep=False, lags=1, sample_weight_halflife=None, seasonality_fourier={}, seasonality_onehot=[], seasonality_ordinal=[], lgbm_kwargs={'verbose': -1}, base_regressor=None):
        # validate input types
        self._validate_inputs(
            data=data,
            group_vars=group_vars,
            exog_vars=exog_vars,
            boxcox=boxcox,
            differencing=differencing,
            include_level=include_level,
            include_timestep=include_timestep,
            lags=lags,
            seasonality_fourier=seasonality_fourier,
            seasonality_onehot=seasonality_onehot,
            seasonality_ordinal=seasonality_ordinal,
            lgbm_kwargs=lgbm_kwargs,
            base_regressor=base_regressor,
            sample_weight_halflife=sample_weight_halflife
        )
        
        # setting the class attributes
        self.data = data 
        self._data_trans = None
        self.endog_var = endog_var 
        self.id_var = id_var 
        self.timestep_var = timestep_var
        self.group_vars = group_vars 
        self.exog_vars = exog_vars 
        self.boxcox = boxcox
        self.differencing = differencing
        self.include_level = include_level
        self.include_timestep = include_timestep
        self.sample_weight_halflife = sample_weight_halflife
        self.seasonality_fourier = seasonality_fourier
        self.seasonality_onehot = seasonality_onehot
        self.seasonality_ordinal = seasonality_ordinal
        self.lgbm_kwargs = lgbm_kwargs
        self.base_regressor = base_regressor

        # convert lags to a list if necessary
        if type(lags) == list:
            self.lags = lags
        else:
            self.lags = list(range(1, lags + 1))

        # infer the timestep and get all timestep values in range
        self._infer_timestep()
        self._all_timesteps = self._get_all_timesteps(self.data)

        # check to make sure that there are not duplicate entries at any timestep for each series
        if len(self.data) != len(self.data.drop_duplicates(subset=[self.id_var, self.timestep_var])):
            raise ValueError('At least one timestep has multiple values within particular series.')
        
    # method to validate all input types and values
    @staticmethod
    def _validate_inputs(**kwargs):
        # type validations
        if not isinstance(kwargs['data'], pd.DataFrame):
            raise TypeError('data must be a pandas DataFrame.')
            
        if not isinstance(kwargs['group_vars'], list):
            raise TypeError('group_vars must be a list.')
            
        if not isinstance(kwargs['exog_vars'], list):
            raise TypeError('exog_vars must be a list.')
            
        if not isinstance(kwargs['boxcox'], (float, int)):
            raise TypeError('boxcox must be a scalar numerical value.')
            
        if not isinstance(kwargs['differencing'], bool):
            raise TypeError('differencing must be a boolean.')
            
        if not isinstance(kwargs['include_level'], bool):
            raise TypeError('include_level must be a boolean.')
            
        if not isinstance(kwargs['include_timestep'], bool):
            raise TypeError('include_timestep must be a boolean.')
            
        # validate lags
        if not isinstance(kwargs['lags'], (int, list)):
            raise TypeError('lags must be an integer or list of integers.')
        if isinstance(kwargs['lags'], list) and not all(isinstance(x, int) for x in kwargs['lags']):
            raise TypeError('All lags must be integers.')
        if isinstance(kwargs['lags'], int) and kwargs['lags'] <= 0:
            raise ValueError('lags must be positive.')
        if isinstance(kwargs['lags'], list) and any(lag <= 0 for lag in kwargs['lags']):
            raise ValueError('All lags must be positive.')
            
        # validate seasonality parameters
        if not isinstance(kwargs['seasonality_fourier'], dict):
            raise TypeError('seasonality_fourier must be a dictionary.')
        if kwargs['seasonality_fourier'] and not all(
            isinstance(k, int) and isinstance(v, int) 
            for k, v in kwargs['seasonality_fourier'].items()
        ):
            raise TypeError('seasonality_fourier must contain integer key:value pairs.')
            
        if not isinstance(kwargs['seasonality_onehot'], list):
            raise TypeError('seasonality_onehot must be a list.')
        if kwargs['seasonality_onehot'] and not all(isinstance(x, int) for x in kwargs['seasonality_onehot']):
            raise TypeError('seasonality_onehot must contain integers.')
            
        if not isinstance(kwargs['seasonality_ordinal'], list):
            raise TypeError('seasonality_ordinal must be a list.')
        if kwargs['seasonality_ordinal'] and not all(isinstance(x, int) for x in kwargs['seasonality_ordinal']):
            raise TypeError('seasonality_ordinal must contain integers.')
            
        # validate remaining parameters
        if not isinstance(kwargs['lgbm_kwargs'], dict):
            raise TypeError('lgbm_kwargs must be a dictionary.')
            
        if kwargs['base_regressor'] is not None and not isinstance(kwargs['base_regressor'], type):
            raise TypeError('base_regressor must be None or a class object.')
            
        if (kwargs['sample_weight_halflife'] is not None and 
            not isinstance(kwargs['sample_weight_halflife'], (int, float))):
            raise TypeError('sample_weight_halflife must be None or a positive number.')
        if kwargs['sample_weight_halflife'] is not None and kwargs['sample_weight_halflife'] <= 0:
            raise ValueError('sample_weight_halflife must be positive.')


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
            daily_bounds = [0.99 * 24 * 60 * 60, 1.01 * 24 * 60 * 60]
            hourly_bounds = [0.99 * 60 * 60, 1.01 * 60 * 60]
            minute_bounds = [0.99 * 60, 1.01 * 60]

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


    def _transform_data(self, data, all_timesteps, lookaheads=1, bootstrap_iter_col=[]):
        # create a dataframe with the group values for all IDs
        id_group_key = data[[self.id_var] + self.group_vars].drop_duplicates()

        # create new dataframe that includes all dates for all time series IDs (and all bootstrap iterations if applicable)
        if len(bootstrap_iter_col) > 0:
            all_date_id_combos = list(product(all_timesteps, data[self.id_var].unique(), data[bootstrap_iter_col[0]].unique()))
        else:
            all_date_id_combos = list(product(all_timesteps, data[self.id_var].unique()))
        data_trans = pd.DataFrame(all_date_id_combos, columns=[self.timestep_var, self.id_var] + bootstrap_iter_col)

        # join the group values onto the newly filled data
        data_trans = pd.merge(left=data_trans, right=id_group_key, on=self.id_var, how='left')
        
        # join the endog and exog variabes onto the newly filled data
        data_trans = pd.merge(
            left=data_trans, 
            right=data[[self.id_var, self.endog_var, self.timestep_var] + self.exog_vars + bootstrap_iter_col], 
            on=[self.id_var, self.timestep_var] + bootstrap_iter_col, 
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

        # if you are differencing and want a current level feature, add it now
        if self.differencing and self.include_level:
            data_trans['endog_level'] = data_trans['endog']

        # difference the data if necessary
        if self.differencing:
            # differencing the data
            data_trans['_endog_differenced'] = data_trans['endog'] - data_trans.groupby([self.id_var] + bootstrap_iter_col)['endog'].shift(1)
            # overwrite the final endog variable if necessary
            data_trans['endog'] = data_trans['_endog_differenced']

        # generate lookahead targets
        for lookahead in range(1, lookaheads + 1):
            # get the future endog for the lookahead
            if self.differencing:
                data_trans[f'endog_lookahead_{str(lookahead)}'] = data_trans.groupby([self.id_var] + bootstrap_iter_col)['_endog_boxcox'].shift(-lookahead) - data_trans['_endog_boxcox']
            else:
                data_trans[f'endog_lookahead_{str(lookahead)}'] = data_trans.groupby([self.id_var] + bootstrap_iter_col)['_endog_boxcox'].shift(-lookahead)

        # generate lags
        for lag in self.lags:
            data_trans[f'endog_lag_{str(lag).zfill(len(str(max(self.lags))))}'] = data_trans.groupby([self.id_var] + bootstrap_iter_col)['endog'].shift(lag)

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

        # add a feature that tracks the timesteps if necessary
        if self.include_timestep:
            timestep_increment_feature = pd.DataFrame({self.timestep_var: all_timesteps, 'Timestep Index': range(len(all_timesteps))})
            data_trans = pd.merge(left=data_trans, right=timestep_increment_feature, how='left', on=self.timestep_var)

        # onehot encode the grouping features
        for group in self.group_vars:
            for group_val in list(data_trans[group].dropna().unique()):
                data_trans[f'{group}_{group_val}'] = (data_trans[group] == group_val).astype(int)

        # store a list of all training features
        target_cols = [c for c in data_trans.columns if 'endog_lookahead_' in c]
        forbidden_cols = [self.id_var, self.timestep_var, self.endog_var] + self.group_vars + target_cols + bootstrap_iter_col
        forbidden_cols += [c for c in data_trans.columns if re.fullmatch(r'^_endog_.*', c)] # adding any intermediate endogenous calculation
        training_cols = [c for c in data_trans.columns if c not in forbidden_cols]
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
    

    def _calculate_sample_weights(self, data):
        # Get the most recent timestep in the entire dataset
        max_timestep = data[self.timestep_var].max()
        
        # Convert inferred timestep to timedelta if it's a dateoffset
        if isinstance(self._inferred_timestep, pd.tseries.offsets.DateOffset):
            reference_dt = dt.datetime(year=2000, month=1, day=1, hour=0, minute=0)
            inferred_timedelta = (reference_dt + self._inferred_timestep) - reference_dt
        else:
            inferred_timedelta = self._inferred_timestep
        
        # Calculate time differences in number of timesteps
        time_diff = (max_timestep - data[self.timestep_var]) / inferred_timedelta

        # Calculate weights using exponential decay
        weights = np.exp(-np.log(2) * np.array(time_diff) / self.sample_weight_halflife)

        return weights
    

    def _determine_cqr_cal_size(self, cqr_cal_size):
        # determine maximum season length
        max_season_length = 0
        if self.seasonality_fourier != {}:
            max_season_length = max(max_season_length, max(list(self.seasonality_fourier.keys())))
        if self.seasonality_onehot != []:
            max_season_length = max(max_season_length, max(self.seasonality_onehot))
        if self.seasonality_ordinal != []:
            max_season_length = max(max_season_length, max(self.seasonality_ordinal))

        # determine cutoff timestep for CQR if necessary
        if cqr_cal_size == 'auto':
            # determine length of 20% of all timesteps
            length_20_dataset = int(np.ceil(len(self._all_timesteps) * 0.20))

            # set the CQR calibration set size to be max of largest season length vs dataset proportion
            cqr_cal_size = max(max_season_length, length_20_dataset)
        elif type(cqr_cal_size) == int:
            if cqr_cal_size <= 0:
                raise ValueError(f'If cqr_cal_size is an integer, it must be positive.')
        elif type(cqr_cal_size) == float:
            if not 0 < cqr_cal_size < 1:
                raise ValueError(f'If cqr_cal_size is a float, it must be in the interval (0, 1).')
            cqr_cal_size = int(np.ceil(len(self._all_timesteps) * cqr_cal_size))
        else:
            raise ValueError(f'cqr_cal_size must be either \'auto\', an integer, or a float.')

        # throw a warning if the calculated CQR size is less than a single season and override it
        if cqr_cal_size < max_season_length:
            warnings.warn(f'cqr_cal_size was set to less than the length of the largest seasonality ({max_season_length} timesteps). A full season length was used instead.', UserWarning)
            cqr_cal_size = max_season_length

        return cqr_cal_size
    

    def stationarity_test(self, test='both'):
        # argument validation
        if test not in ['adf', 'kpss', 'both']:
            raise ValueError('Either \'adf\', \'kpss\', or \'both\' must be passed to the test argument.')
        
        # create a list to store results
        row_list = []
        
        # get transformed data
        data_trans = self._transform_data(self.data, self._all_timesteps, lookaheads=1)
        
        # perform test for each time series ID
        for id in self.data[self.id_var].unique():
            # instantiate a dict to store the data, and track the ID
            current_row = {}
            current_row['ID'] = id
            
            # test raw data
            raw_ts = self.data.loc[self.data[self.id_var]==id, self.endog_var].dropna()
            if test == 'adf' or test == 'both':
                current_row['Raw ADF p-value'] = adfuller(raw_ts)[1]
            if test == 'kpss' or test == 'both':
                current_row['Raw KPSS p-value'] = kpss(raw_ts)[1]
            
            # test transformed data
            trans_ts = data_trans.loc[data_trans[self.id_var]==id, 'endog'].dropna()
            if test == 'adf' or test == 'both':
                current_row['Transformed ADF p-value'] = adfuller(trans_ts)[1]
            if test == 'kpss' or test == 'both':
                current_row['Transformed KPSS p-value'] = kpss(trans_ts)[1]

            # store the row
            row_list.append(current_row)
            
        # Compile results into dataframe
        results = pd.DataFrame(row_list)
        
        return results


class DirectForecaster(_GroupForecaster):
    """
    A forecaster that makes direct multi-step predictions using LightGBM.

    This class implements direct multi-step forecasting by training separate models for each forecast horizon.
    It supports both point forecasts and prediction intervals, with optional conformal quantile regression (CQR)
    for improved interval coverage.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the time series and any additional variables.

    endog_var : str
        The name of the target variable to forecast.

    id_var : str
        The name of the column containing unique identifiers for each time series.

    timestep_var : str
        The name of the column containing the time steps.

    group_vars : list, optional (default=[])
        List of column names containing categorical variables used to group the time series.

    exog_vars : list, optional (default=[])
        List of column names containing exogenous variables to use as predictors.

    boxcox : float or int, optional (default=1)
        The Box-Cox transformation parameter. Use 1 for no transformation.

    differencing : bool, optional (default=False)
        Whether to apply first-order differencing to the target variable.

    include_level : bool, optional (default=True)
        Whether to include the level of the target variable as a feature. When True, this is only included
        when differencing is applied.

    include_timestep : bool, optional (default=False)
        Whether to include the time step as a feature. If True, an integer index starting at zero is mapped
        to each unique timestep chronologically and passed to the regressor.

    lags : int or list, optional (default=1)
        The number of lags to use as features, or a list of specific lag values.

    sample_weight_halflife : int, optional (default=None)
        The halflife, in number of timesteps, used to calculate sample weights during the model fit (more recent
        timesteps have a heavier weight). If None, all samples are weighted equally.

    seasonality_fourier : dict, optional (default={})
        Dictionary with periods as the keys and number of Fourier terms as values.

    seasonality_onehot : list, optional (default=[])
        List of periods for one-hot encoded seasonality features.
    
    seasonality_ordinal : list, optional (default=[])
        List of periods for ordinal encoded seasonality features.
    
    lgbm_kwargs : dict, optional (default={'verbose': -1})
        Additional keyword arguments to pass to LGBMRegressor.
    
    base_regressor : class, optional (default=None)
        Alternative regressor class to use instead of LGBMRegressor. You can create an custom wrapper for
        any statistical or machine learning regressor if the following criteria are met:
        -   The regressor must be capable of quantile regression, and must have an option to pass an "alpha"
            parameter that defines the quantile for prediction (similar to LGBMRegressor implementation).
        -   The class must have fit and predict methods with arguments of (X, y) similar to the scikit-learn implementation.
        
        An example is shown below of a valid custom class (using XGBoost):
            
            class custom_regressor():
                def __init__(self, alpha=0.5): # provide the quantile as an argument
                    # create a custom XGBoost regressor using pinball loss with a specified quantile
                    self.regressor = XGBRegressor(alpha=0.1, objective='reg:quantileerror', quantile_alpha=alpha)

                def fit(self, X, y): # standard fit method
                    self.regressor.fit(X, y)
                    return self 
                
                def predict(self, X, y=None): # standard predict method
                    return self.regressor.predict(X)
    """

    def __init__(self, data, endog_var, id_var, timestep_var, group_vars=[], exog_vars=[], boxcox=1, differencing=False, include_level=True, include_timestep=False, lags=1, sample_weight_halflife=None, seasonality_fourier={}, seasonality_onehot=[], seasonality_ordinal=[], lgbm_kwargs={'verbose': -1}, base_regressor=None):
        super().__init__(
            data=data,
            endog_var=endog_var, 
            id_var=id_var, 
            timestep_var=timestep_var, 
            group_vars=group_vars, 
            exog_vars=exog_vars, 
            boxcox=boxcox, 
            differencing=differencing, 
            include_level=include_level,
            include_timestep=include_timestep,
            lags=lags, 
            sample_weight_halflife=sample_weight_halflife,
            seasonality_fourier=seasonality_fourier, 
            seasonality_onehot=seasonality_onehot, 
            seasonality_ordinal=seasonality_ordinal,
            lgbm_kwargs=lgbm_kwargs,
            base_regressor=base_regressor
        )


    def fit(self, max_steps=1, alpha=None, cqr_cal_size='auto'):
        """
        Create and fit the forecasting models up to a defined forecast horizon.

        Trains separate models for each lookahead timestep up to max_steps ahead. For each horizon,
        trains a point forecast model and optionally prediction interval models with CQR calibration.

        Parameters
        ----------
        max_steps : int, optional (default=1)
            Maximum number of steps ahead to forecast.
        
        alpha : float, optional (default=None)
            Miscoverage rate for prediction intervals (e.g., 0.05 for 95% intervals).
            If None, only point forecasts are produced.
        
        cqr_cal_size : str, int, or float, optional (default='auto')
            Size of the calibration set for CQR:
            -   'auto': Automatically determine size based on data. Uses a minimum of a full season's data
                (using the largest season) or 20% of the data.
            -   int: Number of time steps to use.
            -   float: Fraction of the total time steps to use in (0, 1).
            If None, no CQR calibration is performed and standard quantile regression is used.

        Returns
        ------
        None
        """
        # argument validation
        if not isinstance(max_steps, int) and max_steps <= 0:
            raise TypeError('The max_steps argument must be a positive integer.')
        if alpha is not None and (not isinstance(alpha, float) or not 0 < alpha < 1):
            raise TypeError('The alpha argument must either be None or a float between 0 and 1.')
        if cqr_cal_size is not None and cqr_cal_size != 'auto' and not isinstance(cqr_cal_size, (int, float)):
            raise TypeError('The cqr_cal_size argument must either be None, \'auto\', a positive integer, or a float between 0 and 1.')
        
        # determine CQR calibration set size
        if cqr_cal_size is not None:
            cqr_cal_size = self._determine_cqr_cal_size(cqr_cal_size) 

        # transform the data
        self._data_trans = self._transform_data(data=self.data, all_timesteps=self._all_timesteps, lookaheads=max_steps)

        # calculate alpha
        if type(alpha) == float:
            self._alpha = alpha
        else:
            self._alpha = None

        # create lists to store the trained models
        self._predictors = []
        self._pi_lo_predictors = []
        self._pi_hi_predictors = []
        self._cqr_Q = []

        # make a prediction for each lookahead
        for step in range(1, max_steps + 1):
            # create predictor object
            if self.base_regressor is None:
                current_predictor = LGBMRegressor(**self.lgbm_kwargs).set_params(**{'objective': 'quantile', 'alpha': 0.5})
                current_pi_lo_predictor = LGBMRegressor(**self.lgbm_kwargs).set_params(**{'objective': 'quantile', 'alpha': (self._alpha / 2)})
                current_pi_hi_predictor = LGBMRegressor(**self.lgbm_kwargs).set_params(**{'objective': 'quantile', 'alpha': (1 - self._alpha / 2)})
            else:
                current_predictor = self.base_regressor(alpha=0.5)
                current_pi_lo_predictor = self.base_regressor(alpha=(self._alpha / 2))
                current_pi_hi_predictor = self.base_regressor(alpha=(1 - self._alpha / 2))

            # define the target for the current lookahead and drop any rows with blank targets
            target = f'endog_lookahead_{str(step)}'
            data_train = self._data_trans.dropna(subset=target)
            
            # throw a warning if the training dataset is empty after transformation
            if len(data_train) == 0:
                raise ValueError(f'No training data available after transformation for step {step} due to insufficient historical data for the specified lags and lookahead.')

            # raise a warning if you are losing a significant amount of data for training after transformation
            if len(data_train) < 0.50 * len(self.data):
                lost_proportion = (1 - len(data_train) / len(self.data)) * 100
                warnings.warn(f'{lost_proportion:.1f}% of data was lost after transformation for step {step}. Severe performance degradation may occur.', UserWarning)

            # get the X and y training data
            X_train = data_train[self._X_cols]
            y_train = data_train[target]

            # Calculate sample weights if halflife is specified
            if self.sample_weight_halflife is None:
                current_predictor.fit(X_train, y_train)
            else:
                sample_weights = self._calculate_sample_weights(data_train)
                current_predictor.fit(X_train, y_train, sample_weight=sample_weights)

            # train the model and store it
            self._predictors.append(current_predictor)

            # fitting prediction interval models
            if cqr_cal_size is not None:
                # calculate cqr_cal_size cutoff timestep for current lookahead
                cqr_cal_cutoff = np.array(data_train[self.timestep_var].unique())[-cqr_cal_size]

                # create training and calibration sets for CQR
                X_train_pi = data_train.loc[data_train[self.timestep_var] < cqr_cal_cutoff, self._X_cols]
                y_train_pi = data_train.loc[data_train[self.timestep_var] < cqr_cal_cutoff, target]
                X_cal_pi = data_train.loc[data_train[self.timestep_var] >= cqr_cal_cutoff, self._X_cols]
                y_cal_pi = data_train.loc[data_train[self.timestep_var] >= cqr_cal_cutoff, target]
                ids_cal_pi = data_train.loc[data_train[self.timestep_var] >= cqr_cal_cutoff, self.id_var]

                # train the prediction interval models and store them
                if self.sample_weight_halflife is None:
                    current_pi_lo_predictor.fit(X_train_pi, y_train_pi)
                    current_pi_hi_predictor.fit(X_train_pi, y_train_pi)
                else:
                    sample_weights = self._calculate_sample_weights(data_train.loc[data_train[self.timestep_var] < cqr_cal_cutoff])
                    current_pi_lo_predictor.fit(X_train_pi, y_train_pi, sample_weight=sample_weights)
                    current_pi_hi_predictor.fit(X_train_pi, y_train_pi, sample_weight=sample_weights)
                
                # make low and high predictions
                y_cal_pi_pred_lo = current_pi_lo_predictor.predict(X_cal_pi)
                y_cal_pi_pred_hi = current_pi_hi_predictor.predict(X_cal_pi)
                y_cal_data = pd.DataFrame({self.id_var: ids_cal_pi, 'true': y_cal_pi, 'lo': y_cal_pi_pred_lo, 'hi': y_cal_pi_pred_hi})

                # calculate CQR conformity scores
                current_Q = {} # dictionary that stores the conformity modifier for each series ID for a specific lookahead
                for id in data_train[self.id_var].unique():
                    # getting prediction data for current series ID
                    current_y_cal_pi = y_cal_data.loc[y_cal_data[self.id_var] == id, 'true']
                    current_y_cal_pi_pred_lo = y_cal_data.loc[y_cal_data[self.id_var] == id, 'lo']
                    current_y_cal_pi_pred_hi = y_cal_data.loc[y_cal_data[self.id_var] == id, 'hi']

                    # calculating conformity scores
                    current_E_lo = current_y_cal_pi_pred_lo - current_y_cal_pi
                    current_E_hi = current_y_cal_pi - current_y_cal_pi_pred_hi

                    # taking the empirical quantile of the high and low conformity scores and storing the conformity modifiers
                    empirical_quantile = min((1 - self._alpha / 2) * (1 + 1 / len(current_y_cal_pi)), 1)
                    current_Q_lo = np.quantile(current_E_lo, q=empirical_quantile)
                    current_Q_hi = np.quantile(current_E_hi, q=empirical_quantile)
                    current_Q[id] = [current_Q_lo, current_Q_hi]

                # store the conformity modifiers for all IDs for the current lookahead
                self._cqr_Q.append(current_Q)

            # train the prediction interval models on the full dataset (if using CQR, retrain on all data now that Q values are calculated)
            if self.sample_weight_halflife is None:
                current_pi_lo_predictor.fit(X_train, y_train)
                current_pi_hi_predictor.fit(X_train, y_train)
            else:
                sample_weights = self._calculate_sample_weights(data_train)
                current_pi_lo_predictor.fit(X_train, y_train, sample_weight=sample_weights)
                current_pi_hi_predictor.fit(X_train, y_train, sample_weight=sample_weights)

            # store the trained quantile regressors
            self._pi_lo_predictors.append(current_pi_lo_predictor)
            self._pi_hi_predictors.append(current_pi_hi_predictor)


    def predict(self, steps=1):
        """
        Generate forecasts for multiple steps ahead.

        Makes predictions using the trained models for each lookahead timestep up to the specified
        number of steps ahead. If prediction intervals were enabled during fitting, also generates
        interval forecasts.

        Parameters
        ----------
        steps : int, optional (default=1)
            Number of steps ahead to forecast. If this value is greater than the maximum number of
            timesteps the forecaster was trained on during the fit method, the fit method is called
            again with the lengthened forecast horizon.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the forecasts with columns:
            -   ID variable
            -   Timestep variable
            -   Group variables (if any)
            -   'Forecast': Point forecasts
            -   'Forecast_{alpha/2}' and 'Forecast_{1-alpha/2}': Lower and upper prediction
                interval bounds (if alpha was specified during fitting)
        """
        # argument validation
        if not isinstance(steps, int) and steps <= 0:
            raise TypeError('The steps argument must be a positive integer.')
        
        # instantiate a list to store the predictions
        pred_data_list = []

        # check to see if the intended number of steps to predict is larger than the number of lookahead models trained
        if steps > len(self._predictors):
            # warn the user that more models will need to be trained, then train them
            warnings.warn('The model has not yet been trained for this many prediction steps. Training more models now.', UserWarning)
            self.fit(max_steps=steps, alpha=self._alpha)

        # make a prediction for each lookahead
        for step in range(1, steps + 1):
            # define the prediction data
            data_pred = self._data_trans.loc[self._data_trans[self.timestep_var] == max(self._data_trans[self.timestep_var])]

            # get the X data for prediction
            X_pred = data_pred[self._X_cols]

            # train the model and make predictions
            y_pred = self._predictors[step - 1].predict(X_pred)
            y_pred_pi_lo = self._pi_lo_predictors[step - 1].predict(X_pred)
            y_pred_pi_hi = self._pi_hi_predictors[step - 1].predict(X_pred)

            # modify the predictions with CQR scores if necessary
            if len(self._cqr_Q) > 0:
                ids_pred = data_pred[self.id_var]
                y_pred_data = pd.DataFrame({self.id_var: ids_pred, 'true': y_pred, 'lo': y_pred_pi_lo, 'hi': y_pred_pi_hi})
                for id in data_pred[self.id_var].unique():
                    y_pred_data.loc[y_pred_data[self.id_var] == id, 'lo'] -= self._cqr_Q[step - 1][id][0]
                    y_pred_data.loc[y_pred_data[self.id_var] == id, 'hi'] += self._cqr_Q[step - 1][id][1]
                
                # set the prediction intervals to their modified versions
                y_pred_pi_lo = y_pred_data['lo']
                y_pred_pi_hi = y_pred_data['hi']

            # reverse transform the predictions as necessary
            y_pred = self._reverse_transform_preds(y_pred, data_pred)
            y_pred_pi_lo = self._reverse_transform_preds(y_pred_pi_lo, data_pred)
            y_pred_pi_hi = self._reverse_transform_preds(y_pred_pi_hi, data_pred)

            # store the prediction data
            current_pred_data = data_pred[[self.id_var, self.timestep_var] + self.group_vars].copy()
            current_pred_data['Forecast'] = y_pred
            alpha_lo = self._alpha / 2
            alpha_hi = 1 - (self._alpha / 2)
            current_pred_data[f'Forecast_{alpha_lo:.3f}'] = y_pred_pi_lo 
            current_pred_data[f'Forecast_{alpha_hi:.3f}'] = y_pred_pi_hi 
            current_pred_data[self.timestep_var] += step * self._inferred_timestep
            pred_data_list.append(current_pred_data)

        # transform the predictions to a single dataframe
        prediction_data = pd.concat(pred_data_list, axis=0).reset_index(drop=True)

        return prediction_data
    

class RecursiveForecaster(_GroupForecaster):
    """
    A forecaster that makes recursive multi-step predictions using LightGBM.

    This class implements recursive multi-step forecasting by iteratively using predictions as inputs
    for subsequent predictions. It supports both point forecasts and prediction intervals, with
    bootstrapped residuals used to generate intervals.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the time series and any additional variables.

    endog_var : str
        The name of the target variable to forecast.

    id_var : str
        The name of the column containing unique identifiers for each time series.

    timestep_var : str
        The name of the column containing the time steps.

    group_vars : list, optional (default=[])
        List of column names containing categorical variables used to group the time series.

    exog_vars : list, optional (default=[])
        List of column names containing exogenous variables to use as predictors.

    boxcox : float or int, optional (default=1)
        The Box-Cox transformation parameter. Use 1 for no transformation.

    differencing : bool, optional (default=False)
        Whether to apply first-order differencing to the target variable.

    include_level : bool, optional (default=True)
        Whether to include the level of the target variable as a feature. When True, this is only included
        when differencing is applied.

    include_timestep : bool, optional (default=False)
        Whether to include the time step as a feature. If True, an integer index starting at zero is mapped
        to each unique timestep chronologically and passed to the regressor.

    lags : int or list, optional (default=1)
        The number of lags to use as features, or a list of specific lag values.

    sample_weight_halflife : int, optional (default=None)
        The halflife, in number of timesteps, used to calculate sample weights during the model fit (more recent
        timesteps have a heavier weight). If None, all samples are weighted equally.

    seasonality_fourier : dict, optional (default={})
        Dictionary with periods as the keys and number of Fourier terms as values.

    seasonality_onehot : list, optional (default=[])
        List of periods for one-hot encoded seasonality features.
    
    seasonality_ordinal : list, optional (default=[])
        List of periods for ordinal encoded seasonality features.
    
    lgbm_kwargs : dict, optional (default={'verbose': -1})
        Additional keyword arguments to pass to LGBMRegressor.
    
    base_regressor : class, optional (default=None)
        Alternative regressor class to use instead of LGBMRegressor. You can create an custom wrapper for
        any statistical or machine learning regressor if the following criteria are met:
        -   The regressor must be capable of quantile regression, and must have an option to pass an "alpha"
            parameter that defines the quantile for prediction (similar to LGBMRegressor implementation).
        -   The class must have fit and predict methods with arguments of (X, y) similar to the scikit-learn implementation.
        
        An example is shown below of a valid custom class (using XGBoost):
            
            class custom_regressor():
                def __init__(self, alpha=0.5): # provide the quantile as an argument
                    # create a custom XGBoost regressor using pinball loss with a specified quantile
                    self.regressor = XGBRegressor(alpha=0.1, objective='reg:quantileerror', quantile_alpha=alpha)

                def fit(self, X, y): # standard fit method
                    self.regressor.fit(X, y)
                    return self 
                
                def predict(self, X, y=None): # standard predict method
                    return self.regressor.predict(X)
    """

    def __init__(self, data, endog_var, id_var, timestep_var, group_vars=[], exog_vars=[], boxcox=1, differencing=False, include_level=True, include_timestep=False, lags=1, sample_weight_halflife=None, seasonality_fourier={}, seasonality_onehot=[], seasonality_ordinal=[], lgbm_kwargs={'verbose': -1}, base_regressor=None):
        super().__init__(
            data=data,
            endog_var=endog_var, 
            id_var=id_var, 
            timestep_var=timestep_var, 
            group_vars=group_vars, 
            exog_vars=exog_vars, 
            boxcox=boxcox, 
            differencing=differencing, 
            include_level=include_level,
            include_timestep=include_timestep,
            lags=lags, 
            sample_weight_halflife=sample_weight_halflife,
            seasonality_fourier=seasonality_fourier, 
            seasonality_onehot=seasonality_onehot, 
            seasonality_ordinal=seasonality_ordinal,
            lgbm_kwargs=lgbm_kwargs,
            base_regressor=base_regressor
        )


    def fit(self, alpha=None):
        """
        Create and fit the forecasting model.

        Trains a single model that will be used recursively for multi-step forecasting. The model can
        optionally be used to generate prediction intervals via bootstrapped residuals.

        Parameters
        ----------
        alpha : float, optional (default=None)
            Miscoverage rate for prediction intervals (e.g., 0.05 for 95% intervals).
            If None, only point forecasts are produced.

        Returns
        ------
        None
        """

        # argument validation
        if alpha is not None and (not isinstance(alpha, float) or not 0 < alpha < 1):
            raise TypeError('The alpha argument must either be None or a float between 0 and 1.')
        
        # transform the data
        self._data_trans = self._transform_data(data=self.data, all_timesteps=self._all_timesteps, lookaheads=1)

        # calculate alpha
        if type(alpha) == float:
            self._alpha = alpha
        else:
            self._alpha = None

        # create predictor object
        if self.base_regressor is None:
            self._predictor = LGBMRegressor(**self.lgbm_kwargs).set_params(**{'objective': 'quantile', 'alpha': 0.5})
        else:
            self._predictor = self.base_regressor()

        # define the target for the current lookahead and drop any rows with blank targets
        target = f'endog_lookahead_1'
        data_train = self._data_trans.dropna(subset=target)

        # get the X and y training data
        X_train = data_train[self._X_cols]
        y_train = data_train[target]

        # Calculate sample weights if halflife is specified
        if self.sample_weight_halflife is None:
            self._predictor.fit(X_train, y_train)
        else:
            sample_weights = self._calculate_sample_weights(data_train)
            self._predictor.fit(X_train, y_train, sample_weight=sample_weights)

        # store the in-sample residuals for each time series in a dictionary
        in_sample_residuals = y_train - self._predictor.predict(X_train)
        in_sample_residuals_df = pd.DataFrame({'ID': data_train[self.id_var], 'Residuals': in_sample_residuals})
        self._in_sample_residuals_dict = {}
        for id in data_train[self.id_var].unique():
            self._in_sample_residuals_dict[id] = np.array(in_sample_residuals_df.loc[in_sample_residuals_df['ID'] == id, 'Residuals'])


    def _recursive_loop(self, steps, exog_data, bootstrap_iter, bootstrap=False):
        # create a list to store the bootstrapped prediction data
        pred_data_list = []

        # make predictions for each lookahead
        for step in range(1, steps + 1):
            # define the prediction data
            if step == 1:
                # if bootstrapping, create sets of the data for each bootstrap iteration that includes the iteration number; otherwise, just use the data
                if bootstrap:
                    data = pd.concat([self.data.copy().assign(_bootstrap_iter=i) for i in range(bootstrap_iter)], axis=0)
                else:
                    data = self.data
            else:
                data = pd.concat([data, current_pred_data.rename(columns={'Forecast': self.endog_var})], axis=0)
            
            # get all timesteps in the data
            all_timesteps = self._get_all_timesteps(data)

            # transform the data; include bootstrap iteration column if necessary
            if bootstrap:
                data_trans = self._transform_data(data=data, all_timesteps=all_timesteps, lookaheads=1, bootstrap_iter_col=['_bootstrap_iter'])
            else:
                data_trans = self._transform_data(data=data, all_timesteps=all_timesteps, lookaheads=1)

            # get the most recent batch of data for prediction
            data_pred = data_trans.loc[data_trans[self.timestep_var] == max(data_trans[self.timestep_var])]

            # check to see if exogenous variables were passed to the predict method
            if exog_data is not None:
                # merge the new exogenous variables onto the prediction dataframe, and change the old exogenous feature name for removal
                data_pred = pd.merge(left=data_pred, right=exog_data, on=[self.timestep_var, self.id_var], how='left', suffixes=(None, '__FUTURE__'))

                # for each exogenous variable, infill it with the future data if it exists
                for exog_var in self.exog_vars:
                    future_exog_var = f'{exog_var}__FUTURE__'
                    if future_exog_var in data_pred.columns:
                        # where the original variable is null, replace with future values
                        data_pred[exog_var] = data_pred[exog_var].fillna(data_pred[future_exog_var])

            # get the X data for prediction
            X_pred = data_pred[self._X_cols]

            # train the model and make predictions
            y_pred = self._predictor.predict(X_pred)

            # if bootstrapping, add on randomly sampled residuals from the fitted values from respective time series
            if bootstrap:
                for id in data_pred[self.id_var].unique():
                    # get indices for current ID, and calculate length of samples to be taken
                    id_mask = data_pred[self.id_var] == id
                    n_instances = id_mask.sum()

                    # sample residuals for each instance of this ID and add them to the predictions
                    sampled_residuals = np.random.choice(a=self._in_sample_residuals_dict[id], size=n_instances)
                    y_pred[id_mask] += sampled_residuals

            # reverse transform the predictions as necessary
            y_pred = self._reverse_transform_preds(y_pred, data_pred)

            # store the prediction data (including bootstrap iteration, if necessary)
            if bootstrap:
                current_pred_data = data_pred[[self.id_var, self.timestep_var] + self.group_vars + ['_bootstrap_iter']].copy()
            else:
                current_pred_data = data_pred[[self.id_var, self.timestep_var] + self.group_vars].copy()
            current_pred_data['Forecast'] = y_pred 
            current_pred_data[self.timestep_var] += self._inferred_timestep
            pred_data_list.append(current_pred_data)

        # transform the predictions to a single dataframe
        prediction_data = pd.concat(pred_data_list, axis=0).reset_index(drop=True)

        return prediction_data


    def predict(self, steps=1, exog_data=None, bootstrap_iter=500):
        """
        Generate forecasts for multiple steps ahead.

        Makes recursive multi-step predictions by using predictions as inputs for subsequent predictions.
        Optionally generates prediction intervals via bootstrapped residuals.

        Parameters
        ----------
        steps : int, optional (default=1)
            Number of steps ahead to forecast.

        exog_data : pandas.DataFrame, optional (default=None)
            Future values of exogenous variables. Must contain the same columns as the exogenous
            variables used during fitting, along with the ID and timestep variables.

        bootstrap_iter : int, optional (default=500)
            Number of bootstrap iterations to use when generating prediction intervals. It is recommended
            to use a bare minimum of 100 bootstrap iterations. An excessive number of iterations will
            be computationally intensive. Only used when alpha was specified during fitting.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the forecasts and optionally prediction intervals for each
            time series at each forecast horizon.
        """

        # argument validation
        if not isinstance(steps, int) and steps <= 0:
            raise TypeError('The steps argument must be a positive integer.')
        if exog_data is not None and not isinstance(exog_data, pd.DataFrame):
            raise TypeError('The data must be either None or a pandas DataFrame.')
        if not isinstance(bootstrap_iter, int) and bootstrap_iter < 10:
            raise TypeError('The steps argument must be a an integer and greater than or equal to 10.')
        
        # check to see if the model needs exogenous variables to be passed
        if len(self.exog_vars) > 0 and exog_data is None:
            warnings.warn('The model was fit on exogenous features, but none were passed to the predict method.', UserWarning)

        # perform recursive forecasting
        prediction_data = self._recursive_loop(steps=steps, exog_data=exog_data, bootstrap_iter=bootstrap_iter, bootstrap=False)

        # perform bootstrapping if prediction intervals need to be generated
        if self._alpha is not None:
            # perform recursive forecasting with bootstrapped residuals to generate many potential paths
            bootstrap_prediction_data = self._recursive_loop(steps=steps, exog_data=exog_data, bootstrap_iter=bootstrap_iter, bootstrap=True)

            # calculate percentiles from the bootstrapped predictions
            for a in [self._alpha / 2, 1 - self._alpha / 2]:
                current_pi = bootstrap_prediction_data[[self.id_var, self.timestep_var, 'Forecast']].groupby([self.id_var, self.timestep_var]).aggregate({'Forecast': lambda x: x.quantile(a)}).reset_index()
                current_pi = current_pi.rename(columns={'Forecast': f'Forecast_{a:.3f}'})
                prediction_data = pd.merge(left=prediction_data, right=current_pi, on=[self.id_var, self.timestep_var], how='left')

        return prediction_data
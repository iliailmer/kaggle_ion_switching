from typing import List, Union, Tuple
import xgboost as xgb
import pandas as pd
import scipy as sp
from functools import partial
import numpy as np
from lightgbm import Dataset
from xgboost import DMatrix
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold


def MacroF1MetricRegression(preds: np.ndarray, dtrain: Dataset):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(np.int16)
    score = f1_score(labels, preds, average='macro')
    return ('MacroF1Metric', score, True)


def MacroF1MetricRegressionXGB(preds: np.ndarray, dtrain: DMatrix):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(np.int16)
    score = f1_score(labels, preds, average='macro')
    return ('MacroF1MetricXGB', score)


def reduce_mem_usage(df: pd.DataFrame, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2  # just added
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    percent = 100 * (start_mem - end_mem) / start_mem
    print(
        'Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'
        .format(start_mem, end_mem, percent))
    return df


def train_lgb(params: dict,
              X: pd.DataFrame,
              y: pd.DataFrame,
              X_test: pd.DataFrame,
              oof_df: pd.DataFrame,
              sub: pd.DataFrame,
              features: list,
              feval: dict = {},
              objective: str = 'regression',
              num_boost_round: int = 1,
              early_stopping_rounds: int = 50,
              verbose: int = 10):
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    feat_importance_df = pd.DataFrame(index=features)
    fold = 0
    f1 = []
    for train_id, valid_id in tqdm(kfold.split(X, y)):
        fold += 1
        x_train, y_train = X.iloc[train_id, :], y.iloc[train_id]
        x_val, y_val = X.iloc[valid_id, :], y.iloc[valid_id]

        train_set = lgb.Dataset(x_train, y_train)
        valid_set = lgb.Dataset(x_val, y_val)

        model = lgb.train(params=params,
                          feval=feval[objective],
                          train_set=train_set,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          valid_sets=[train_set, valid_set],
                          verbose_eval=verbose)
        pred = model.predict(x_val, num_iteration=model.best_iteration)

        pred = np.round(np.clip(pred, 0, 10)).astype(np.int32)

        test_preds = model.predict(X_test, num_iteration=model.best_iteration)

        test_preds = np.round(np.clip(test_preds, 0, 10)).astype(np.int32)

        oof_df.loc[oof_df.iloc[valid_id].index, 'oof'] = pred
        sub[f'lgb_open_channels_fold_{fold}'] = test_preds

        f1.append(
            f1_score(oof_df.loc[oof_df.iloc[valid_id].index]['open_channels'],
                     oof_df.loc[oof_df.iloc[valid_id].index]['oof'],
                     average='macro'))
        feat_importance_df[
            f'lgb_importance_{fold}'] = model.feature_importance()

    print(f"Mean oof f1:{np.mean(f1)}, std: {np.std(f1)}")

    oof_f1 = f1_score(oof_df['open_channels'], oof_df['oof'], average='macro')
    oof_rmse = np.sqrt(
        mean_squared_error(oof_df['open_channels'], oof_df['oof']))

    return oof_df.copy(), feat_importance_df.copy(), sub.copy(
    ), oof_f1, oof_rmse


def train_xgb(params: dict,
              X: pd.DataFrame,
              y: pd.DataFrame,
              X_test: pd.DataFrame,
              oof_df: pd.DataFrame,
              features: List,
              sub: pd.DataFrame,
              feval: dict = {},
              verbose_eval=100,
              objective: str = 'regression',
              num_boost_round: int = 1,
              early_stopping_rounds: int = 50,
              sklearn_model=None):
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold = 0
    f1: list = []
    for train_id, valid_id in kfold.split(X, y):
        fold += 1
        x_train, y_train = X.iloc[train_id, :], y.iloc[train_id]
        x_val, y_val = X.iloc[valid_id, :], y.iloc[valid_id]

        train_set = xgb.DMatrix(x_train, y_train)
        valid_set = xgb.DMatrix(x_val, y_val)

        model = xgb.train(params,
                          train_set,
                          num_boost_round=num_boost_round,
                          evals=[(train_set, 'train'), (valid_set, 'val')],
                          feval=feval[objective],
                          verbose_eval=verbose_eval,
                          maximize=True,
                          early_stopping_rounds=early_stopping_rounds)
        pred = model.predict(xgb.DMatrix(x_val))
        pred = np.round(np.clip(pred, 0, 10)).astype(np.int32)

        test_preds = model.predict(xgb.DMatrix(X_test))
        test_preds = np.round(np.clip(test_preds, 0, 10)).astype(np.int32)

        oof_df.loc[oof_df.iloc[valid_id].index, 'oof'] = pred
        sub[f'xgb_open_channels_fold_{fold}'] = test_preds

        f1.append(
            f1_score(oof_df.loc[oof_df.iloc[valid_id].index]['open_channels'],
                     oof_df.loc[oof_df.iloc[valid_id].index]['oof'],
                     average='macro'))
        break

    # oof_f1 = f1_score(oof_df['open_channels'], oof_df['oof'], average='macro')
    # oof_rmse = np.sqrt(
    #     mean_squared_error(oof_df['open_channels'], oof_df['oof']))

    print(f"Mean oof f1:{np.mean(f1)}, std: {np.std(f1)}")
    return sub


def feature_eng(df: pd.DataFrame, bs: int = 500_000,
                bs_slice: int = 25_000,
                windows: Union[List, Tuple] = (10, 50)):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    df['batch'] = df.index // bs
    df['batch_index'] = df.index - (df.batch * bs)
    df['batch_slices'] = df['batch_index'] // bs_slice
    df['batch_slices2'] = df['batch'].astype(str).str.zfill(
        3) + '_' + df['batch_slices'].astype(str).str.zfill(3)

    for c in tqdm(['batch', 'batch_slices2']):
        df[f'batch_{bs//1000}k_max_{c}'] = df.groupby(
            [f'{c}'])['signal_undrifted'].transform(np.max)
        df[f'batch_{bs//1000}k_min_{c}'] = df.groupby(
            [f'{c}'])['signal_undrifted'].transform(np.min)
        df[f'batch_{bs//1000}k_mean_{c}'] = df.groupby(
            [f'{c}'])['signal_undrifted'].transform(np.mean)
        df[f'batch_{bs//1000}k_std_{c}'] = df.groupby(
            [f'{c}'])['signal_undrifted'].transform(np.std)
        df[f'batch_{bs//1000}k_median_{c}'] = df.groupby(
            [f'{c}'])['signal_undrifted'].transform(np.median)
        df[f'batch_{bs//1000}k_diff_max_{c}'] = df.groupby([
            f'{c}'
        ])['signal_undrifted'].transform(lambda x: np.max(np.diff(x)))

        df[f'batch_{bs//1000}k_diff_min_{c}'] = df.groupby([
            f'{c}'
        ])['signal_undrifted'].transform(lambda x: np.min(np.diff(x)))
        df[f'batch_{bs//1000}k_range_{c}'] = np.abs(
            df[f'batch_{bs//1000}k_max_{c}'] -
            df[f'batch_{bs//1000}k_min_{c}'])
        df[f'batch_{bs//1000}k_maxtomin_{c}'] = np.abs(
            (df[f'batch_{bs//1000}k_max_{c}'] + 1e-10) /
            (df[f'batch_{bs//1000}k_min_{c}'] + 1e-10))

        df[f'batch_{bs//1000}k_shift_1_{c}'] = df.groupby(
            [f'{c}']).shift(1)['signal_undrifted']
        df[f'batch_{bs//1000}k_shift_-1_{c}'] = df.groupby(
            [f'{c}']).shift(-1)['signal_undrifted']
        df[f'batch_{bs//1000}k_shift_2_{c}'] = df.groupby(
            [f'{c}']).shift(2)['signal_undrifted']
        df[f'batch_{bs//1000}k_shift_-2_{c}'] = df.groupby(
            [f'{c}']).shift(-2)['signal_undrifted']
        for window in tqdm(windows):
            df[f'batch_{bs//1000}k_rolling_max_{c}_{window}'] = df[
                'signal_undrifted'].rolling(window=window).max()
            df[f'batch_{bs//1000}k_rolling_min_{c}_{window}'] = df[
                'signal_undrifted'].rolling(window=window).min()
            df[f'batch_{bs//1000}k_rolling_maxtomin_{c}_{window}'] = (
                df[f'batch_{bs//1000}k_rolling_max_{c}_{window}'] + 1e-10) / (
                    df[f'batch_{bs//1000}k_rolling_min_{c}_{window}'] + 1e-10)
            df[f'batch_{bs//1000}k_rolling_range_{c}_{window}'] = df[
                f'batch_{bs//1000}k_rolling_max_{c}_{window}'] - df[
                    f'batch_{bs//1000}k_rolling_min_{c}_{window}']

            df[f'batch_{bs//1000}k_rolling_mean_{c}_{window}'] = df[
                'signal_undrifted'].rolling(window=window).mean()
            df[f'batch_{bs//1000}k_rolling_std_{c}_{window}'] = df[
                'signal_undrifted'].rolling(window=window).std()

    feats = [
        c for c in df.columns if c not in [
            'time', 'open_channels', 'batch', 'batch_index', 'batch_slices',
            'batch_slices2'
        ]
    ]
    for c in feats:
        df[c + '_msignal'] = df[c] - df['signal_undrifted']
    feats = [
        c for c in df.columns if c not in [
            'time', 'open_channels', 'batch', 'batch_index', 'batch_slices',
            'batch_slices2'
        ]
    ]

    return df, feats


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize F1 (Macro) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        self.coef_ = 0

    def _f1_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf],
                     labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -f1_score(y, X_p, average='macro')

    def fit(self, X, y):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        print("Optimizing rounder...")
        self.coef_ = sp.optimize.minimize(loss_partial,
                                          initial_coef,
                                          method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        print("Rounding...")
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf],
                      labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

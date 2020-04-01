from utils import MacroF1MetricRegression
from utils import reduce_mem_usage
from utils import train_lgb, feature_eng
import pandas as pd
import numpy as np
train = pd.read_csv('liverpool-ion-switching/train.csv',
                    dtype={
                        'time': np.float32,
                        'signal': np.float32,
                        'open_channels': np.int32
                    })
test = pd.read_csv('liverpool-ion-switching/test.csv',
                   dtype={
                       'time': np.float32,
                       'signal': np.float32,
                   })
sub = pd.read_csv('liverpool-ion-switching/sample_submission.csv')
a = 500000
b = 600000
train['signal_undrifted'] = train.signal
train.loc[train.index[a:b],
          'signal_undrifted'] = train.signal[a:b].values - 3 * (
              train.time.values[a:b] - 50) / 10.


def f(x, low, high, mid):
    return -((-low + high) / 625) * (x - mid)**2 + high - low


# CLEAN TRAIN BATCH 7
batch = 7
a = 500000 * (batch - 1)
b = 500000 * batch
train.loc[train.index[a:b], 'signal_undrifted'] = train.signal.values[a:b] - f(
    train.time[a:b].values, -1.817, 3.186, 325)
# CLEAN TRAIN BATCH 8
batch = 8
a = 500000 * (batch - 1)
b = 500000 * batch
train.loc[train.index[a:b], 'signal_undrifted'] = train.signal.values[a:b] - f(
    train.time[a:b].values, -0.094, 4.936, 375)
# CLEAN TRAIN BATCH 9
batch = 9
a = 500000 * (batch - 1)
b = 500000 * batch
train.loc[train.index[a:b], 'signal_undrifted'] = train.signal.values[a:b] - f(
    train.time[a:b].values, 1.715, 6.689, 425)
# CLEAN TRAIN BATCH 10
batch = 10
a = 500000 * (batch - 1)
b = 500000 * batch
train.loc[train.index[a:b], 'signal_undrifted'] = train.signal.values[a:b] - f(
    train.time[a:b].values, 3.361, 8.45, 475)

test['signal_undrifted'] = test.signal

# REMOVE BATCH 1 DRIFT
start = 500
a = 0
b = 100000
test.loc[test.index[a:b], 'signal_undrifted'] = test.signal.values[a:b] - \
    3*(test.time.values[a:b]-start)/10.
start = 510
a = 100000
b = 200000
test.loc[test.index[a:b], 'signal_undrifted'] = test.signal.values[a:b] - \
    3*(test.time.values[a:b]-start)/10.
start = 540
a = 400000
b = 500000
test.loc[test.index[a:b], 'signal_undrifted'] = test.signal.values[a:b] - \
    3*(test.time.values[a:b]-start)/10.

# REMOVE BATCH 2 DRIFT
start = 560
a = 600000
b = 700000
test.loc[test.index[a:b], 'signal_undrifted'] = test.signal.values[a:b] - \
    3*(test.time.values[a:b]-start)/10.
start = 570
a = 700000
b = 800000
test.loc[test.index[a:b], 'signal_undrifted'] = test.signal.values[a:b] - \
    3*(test.time.values[a:b]-start)/10.
start = 580
a = 800000
b = 900000
test.loc[test.index[a:b], 'signal_undrifted'] = test.signal.values[a:b] - \
    3*(test.time.values[a:b]-start)/10.

# REMOVE BATCH 3 DRIFT


def f2(x):
    return -(0.00788)*(x-625)**2+2.345 + 2.58


a = 1000000
b = 1500000
test.loc[test.index[a:b], 'signal_undrifted'] = test.signal.values[a:b] - \
    f2(test.time[a:b].values)


train, features = feature_eng(
    train, bs=25_000, bs_slice=2500, windows=[10, 50])
train = reduce_mem_usage(train)
test, _ = feature_eng(test, bs=25_000, bs_slice=2500, windows=[10, 50])
test = reduce_mem_usage(test)
params_lgb = {
    'learning_rate': 0.015,  # 0.098
    'max_depth': 7,
    'num_leaves': 200,  # 2**8 + 1,
    'metric': 'rmse',
    'random_state': 42,
    'n_jobs': -1,
    'sample_fraction': 0.33
}
feval = {'regression': MacroF1MetricRegression
         }
X = train[features]
X_test = test[features]
y = train['open_channels']
oof_df = train[['time', 'open_channels']].copy()
oof_df_lgb, feat_importance_df_lgb, sub_lgb2, oof_f2, oof_rmse2 = train_lgb(
    params=params_lgb,
    X=X,
    y=y,
    X_test=X_test,
    sub=sub,
    early_stopping_rounds=150,
    num_boost_round=2000,  # 3000
    oof_df=oof_df,
    features=features,
    feval=feval,
    verbose=100)

sub_lgb2['open_channels'] = sub_lgb2.iloc[:, 2:].median(axis=1).astype(int)

sub_lgb2[['time', 'open_channels']].to_csv(
    'submission_lgb_with_drift_rolling.csv',
    index=False,
    float_format='%.4f')

sub_lgb2.to_csv(
    'out_of_fold_test_set_lgb_pred.csv',
    index=False,
    float_format='%.4f')

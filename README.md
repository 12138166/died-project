# died-project
something pass away……


This project builds an end-to-end pipeline for the Kaggle "DRW - Crypto Market Prediction" competition.
It uses gradient boosting models (XGBoost, LightGBM), multiple training-window slices (full history vs. recent data), time-decay sample weights, and both intra-model and cross-model ensembling.
The goal was to explore how different history lengths and weighting schemes affect predictive performance in a crypto market setting.


if enhance

kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    ...
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
    ...
shuffle=False 虽然不会打乱顺序，但 KFold 的行为是：
把数据按顺序切成 3 段：
fold1：valid = 第一段，train = 第二段 + 第三段
fold2：valid = 第二段，train = 第一段 + 第三段
fold3：valid = 第三段，train = 第一段 + 第二段

=> 也就是说，在 fold1 里，在预测“早期样本”（valid=前 1/3）的时候，训练集合里包含了后面 2/3 的未来数据。在真正“时间上的未来不可见”的任务里，这就是一种时间泄露（temporal leakage）。

so,
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
for fold, (train_idx, valid_idx) in enumerate(tscv.split(train_df), start=1):
    ...
TimeSeriesSplit 的逻辑是：train 永远是“之前的所有数据”，valid 是“后面那段”, 不会用“未来”去预测“过去”。



fit部分中，
model.fit(
    X_train, y_train, sample_weight=sw,
    eval_set=[(X_valid, y_valid)]
)

没有设 eval_metric，也没有设 early_stopping_rounds，就只能把 n_estimators 全跑完，eval_set 实际上只用于打印（xgboost 默认 metric），也不能自动找到“最优迭代轮数”（LGBM 也类似，多加一个 early_stopping_rounds 会非常有用）
改成：
model = Estimator(**params)
model.fit(
    X_train, y_train,
    sample_weight=sw,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",          # 或者根据任务设
    early_stopping_rounds=100,
    verbose=False
)


写法问题：
mask = valid_idx >= cutoff
if mask.any():
    idxs = valid_idx[mask]
    oof_preds[name][slice_name][idxs] = model.predict(
        train_df.iloc[idxs][Config.FEATURES])
if cutoff > 0 and (~mask).any():
    oof_preds[name][slice_name][valid_idx[~mask]] = (
        oof_preds[name]["full_data"][valid_idx[~mask]])



total_score = sum(slice_scores.values())
slice_weights = {sn: sc/total_score for sn, sc in slice_scores.items()}
如果所有 slice 的 Pearson 都是正的，就完全没问题。但如果有一个 slice Pearson 是负的 / 非常小，这种“直接除以总和”的办法就会有点危险（出现负权重）。







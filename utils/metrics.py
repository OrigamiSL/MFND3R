import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    mae = np.abs(pred - true)
    return np.mean(mae)


def MSE(pred, true):
    mse = (pred - true) ** 2
    return np.mean(mse)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true):
    return 200 * np.mean(np.abs((pred - true) / (np.abs(true) + np.abs(pred))))


def MASE(pred, true, input_x, frequency):
    denom = np.mean(np.abs(input_x[:, frequency:, :] - input_x[:, :input_x.shape[1] - frequency, :]), axis=1) + 1e-6
    mase = np.mean(np.mean(np.abs(pred - true), axis=1) / denom)
    return mase


def OWA(pred, true, pred_naive, input_x, frequency):
    smape_init = SMAPE(pred, true)
    smape_naive = SMAPE(pred_naive, true)
    mase_init = MASE(pred, true, input_x, frequency)
    mase_naive = MASE(pred_naive, true, input_x, frequency)
    owa = 0.5 * (smape_init / smape_naive + mase_init / mase_naive)
    return owa


def metric(pred, true, pred_naive=None, input_x=None, frequency=1, data='ETTh1'):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    if pred_naive is None:
        if 'PEMS' in data:
            mape = MAPE(pred, true)
            rmse = RMSE(pred, true)
            return mae, mape, rmse
        return mae, mse
    smape = SMAPE(pred, true)
    owa = OWA(pred, true, pred_naive, input_x, frequency)
    return mae, mse, smape, owa


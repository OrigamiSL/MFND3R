import argparse
import os
import torch
import numpy as np
import time

from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', type=str, required=True, default='MFND3R',
                    help='forecasting model of experiment, options: [MFND3R]')
parser.add_argument('--mode', type=str, required=True, default='direct',
                    help='forecasting format')
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S]; M:multivariate predict multivariate, '
                         'S: univariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or M task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--input_len', type=int, default=96, help='input length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction length')

parser.add_argument('--enc_in', type=int, default=7, help='input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=28, help='hidden dims of model')
parser.add_argument('--representation', type=int, default=320,
                    help='representation dims in the end of the intra-reconstruction phase')
parser.add_argument('--ODA_layers', type=int, default=3, help='nums of ODA layers')
parser.add_argument('--VRCA_layers', type=int, default=1, help='nums of VRCA layers')

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--save_loss', action='store_true', help='whether saving results and checkpoints', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--train', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)
parser.add_argument('--use_RevIN', action='store_true',
                    help='whether to use RevIN'
                    , default=False)
parser.add_argument('--test_inverse_transform', action='store_true',
                    help='', default=False)
parser.add_argument('--freq', type=str, default='Daily', help='')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'M': [7, 7], 'S': [1, 1]},
    'ECL': {'data': 'ECL.csv', 'M': [321, 321], 'S': [1, 1]},
    'weather': {'data': 'weather.csv', 'M': [21, 21], 'S': [1, 1]},
    'Solar': {'data': 'solar_AL.txt', 'M': [137, 137], 'S': [1, 1]},
    'Ship1': {'data': 'Ship1.csv', 'M': [100, 100], 'S': [1, 1]},
    'Ship2': {'data': 'Ship2.csv', 'M': [100, 100], 'S': [1, 1]},
    'PEMS08': {'data': 'PEMS08.npz', 'M': [170, 170], 'S': [1, 1]},
    'M4_yearly': {'data': 'Yearly-train.csv', 'M': [1, 1], 'S': [1, 1]},
    'M4_quarterly': {'data': 'Quarterly-train.csv', 'M': [1, 1], 'S': [1, 1]},
    'M4_monthly': {'data': 'Monthly-train.csv', 'M': [1, 1], 'S': [1, 1]},
    'M4_weekly': {'data': 'Weekly-train.csv', 'M': [1, 1], 'S': [1, 1]},
    'M4_daily': {'data': 'Daily-train.csv', 'M': [1, 1], 'S': [1, 1]},
    'M4_hourly': {'data': 'Hourly-train.csv', 'M': [1, 1], 'S': [1, 1]},
}
type_map = {'Yearly': 1, 'Quarterly': 4, 'Monthly': 12, 'Weekly': 1, 'Daily': 1, 'Hourly': 24}
args.frequency = type_map[args.freq]

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.enc_in, args.c_out = data_info[args.features]

args.target = args.target.replace('/r', '').replace('/t', '').replace('/n', '')

lr = args.learning_rate
print('Args in experiment:')
print(args)

mse_total = []
mae_total = []

smape_total = []
owa_total = []

rmse_total = []
mape_total = []

Exp = Exp_Model
for ii in range(args.itr):
    # setting record of experiments
    exp = Exp(args)  # set experiments
    if args.train:
        setting = '{}/{}_{}_ft{}_ll{}_pl{}_{}'.format(args.model, args.model, args.data,
                                                      args.features, args.input_len,
                                                      args.pred_len, ii)
        print('>>>>>>>start training| pred_len:{}, settings: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.
              format(args.pred_len, setting))
        try:
            exp = Exp(args)  # set experiments
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from forecasting early')

        print('>>>>>>>testing| pred_len:{}: {}<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments
        if 'M4' in args.data:
            smape, owa = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
            smape_total.append(smape)
            owa_total.append(owa)
        elif 'PEMS' in args.data:
            mae, mape, rmse = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
            mae_total.append(mae)
            mape_total.append(mape)
            rmse_total.append(rmse)
        else:
            mse, mae = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
            mse_total.append(mse)
            mae_total.append(mae)
        torch.cuda.empty_cache()
        args.learning_rate = lr
    else:
        setting = '{}/{}_{}_ft{}_ll{}_pl{}_{}'.format(args.model, args.model, args.data,
                                                      args.features, args.input_len,
                                                      args.pred_len, ii)
        print('>>>>>>>testing| pred_len:{} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments

        if 'M4' in args.data:
            smape, owa = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
            smape_total.append(smape)
            owa_total.append(owa)
        elif 'PEMS' in args.data:
            mae, mape, rmse = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
            mae_total.append(mae)
            mape_total.append(mape)
            rmse_total.append(rmse)
        else:
            mse, mae = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
            mse_total.append(mse)
            mae_total.append(mae)
        torch.cuda.empty_cache()
        args.learning_rate = lr

if 'M4' in args.data:
    path1 = './result_M4.csv'
    if not os.path.exists(path1):
        with open(path1, "a") as f:
            write_csv = ['Time', 'Model', 'Data', 'seq_len', 'pred_len', 'ODA_layers', 'VRCA_nums',
                         'Mean SMAPE', 'Mean OWA', 'Std SMAPE', 'Std OWA']
            np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
            f.flush()
            f.close()
elif 'PEMS' in args.data:
    path1 = './result_PEMS.csv'
    if not os.path.exists(path1):
        with open(path1, "a") as f:
            write_csv = ['Time', 'Model', 'Data', 'seq_len', 'pred_len', 'ODA_layers', 'VRCA_nums',
                         'Mean MAE', 'Mean MAPE', 'Mean RMSE', 'Std MAE', 'Std MAPE', 'Std RMSE']
            np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
            f.flush()
            f.close()
else:
    path1 = './result.csv'
    if not os.path.exists(path1):
        with open(path1, "a") as f:
            write_csv = ['Time', 'Model', 'Data', 'seq_len', 'pred_len', 'ODA_layers', 'VRCA_nums', 'Mean MSE',
                         'Mean MAE', 'Std MSE', 'Std MAE']
            np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
            f.flush()
            f.close()

if 'M4' in args.data:
    smape = np.asarray(smape_total)
    owa = np.asarray(owa_total)
    avg_smape = np.mean(smape)
    std_smape = np.std(smape)
    avg_owa = np.mean(owa)
    std_owa = np.std(owa)
    print('|Mean|smape:,{}, owa:,{}|Std|smape:,{}, owa:,{}'.format(avg_smape, avg_owa, std_smape, std_owa))
    path = './result.log'
    with open(path, "a") as f:
        f.write('|{}_{}|pred_len{}: nums of ODA:{}, nums of VRCA:{}, dropout:{}'.format(
            args.data, args.features, args.pred_len, args.ODA_layers, args.VRCA_layers, args.dropout) + '\n')
        f.write('|Mean|smape:,{}, owa:,{}|Std|smape:,{}, owa:,{}'.
                format(avg_smape, avg_owa, std_smape, std_owa) + '\n')
        f.flush()
        f.close()
    with open(path1, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write(',{},{},{},{},{},{},{},{},{},{}'.
                format(args.model, args.data, args.input_len, args.pred_len, args.ODA_layers,
                       args.VRCA_layers, avg_smape, avg_owa, std_smape, std_owa) + '\n')
        f.flush()
        f.close()
elif 'PEMS' in args.data:
    mae = np.asarray(mae_total)
    mape = np.asarray(mape_total)
    rmse = np.asarray(rmse_total)
    avg_mae = np.mean(mae)
    std_mae = np.std(mae)
    avg_mape = np.mean(mape)
    std_mape = np.std(mape)
    avg_rmse = np.mean(rmse)
    std_rmse = np.std(rmse)
    print('|Mean|mae:,{}, mape:,{}, rmse:,{}|Std|mae:,{}, mape:,{}, rmse:,{}'.format(avg_mae, avg_mape, avg_rmse,
                                                                                     std_mae, std_mape, std_rmse))
    path = './result.log'
    with open(path, "a") as f:
        f.write('|{}_{}|pred_len{}: nums of ODA:{}, nums of VRCA:{}'.format(
            args.data, args.features, args.pred_len, args.ODA_layers, args.VRCA_layers) + '\n')
        f.write('|Mean|mae:,{}, mape:,{}, rmse:,{}|Std|mae:,{}, mape:,{}, rmse:,{}'.
                format(avg_mae, avg_mape, avg_rmse, std_mae, std_mape, std_rmse) + '\n')
        f.flush()
        f.close()
    with open(path1, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write(',{},{},{},{},{},{},{},{},{},{},{},{}'.
                format(args.model, args.data, args.input_len, args.pred_len, args.ODA_layers,
                       args.VRCA_layers, avg_mae, avg_mape, avg_rmse, std_mae, std_mape, std_rmse) + '\n')
        f.flush()
        f.close()
else:
    mse = np.asarray(mse_total)
    mae = np.asarray(mae_total)
    avg_mse = np.mean(mse)
    std_mse = np.std(mse)
    avg_mae = np.mean(mae)
    std_mae = np.std(mae)
    print('|Mean|mse:,{}, mae:,{}|Std|mse:,{}, mae:,{}'.format(avg_mse, avg_mae, std_mse, std_mae))
    path = './result.log'
    with open(path, "a") as f:
        f.write('|{}_{}|pred_len{}: nums of ODA:{}, nums of VRCA:{},'.format(
            args.data, args.features, args.pred_len, args.ODA_layers, args.VRCA_layers) + '\n')
        f.write('|Mean|mse:,{}, mae:,{}|Std|mse:,{}, mae:,{}'.
                format(avg_mse, avg_mae, std_mse, std_mae) + '\n')
        f.flush()
        f.close()
    with open(path1, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write(',{},{},{},{},{},{},{},{},{},{}'.
                format(args.model, args.data, args.input_len, args.pred_len, args.ODA_layers,
                       args.VRCA_layers, avg_mse, avg_mae, std_mse, std_mae) + '\n')
        f.flush()
        f.close()

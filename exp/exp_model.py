from data.data_loader import Dataset_ETT_hour, Dataset_ETT_min, Dataset_Custom, Dataset_Solar, Dataset_m4, Dataset_PEMS
from exp.exp_basic import Exp_Basic
from MFND3R.MFND3R import MFND3R

from utils.tools import EarlyStopping, adjust_learning_rate, loss_process
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'MFND3R': MFND3R,
        }
        pred_len = self.args.pred_len if self.args.mode == 'direct' else 1
        model = model_dict[self.args.model](
            self.args.enc_in,
            self.args.c_out,
            self.args.input_len,
            pred_len,
            self.args.ODA_layers,
            self.args.VRCA_layers,
            self.args.d_model,
            self.args.representation,
            self.args.dropout,
            self.args.alpha,
            self.args.use_RevIN,
        ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, predict=False):
        args = self.args
        data_set = None
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm2': Dataset_ETT_min,
            'weather': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Ship1': Dataset_Custom,
            'Ship2': Dataset_Custom,
            'Solar': Dataset_Solar,
            'PEMS08': Dataset_PEMS,
            'M4_yearly': Dataset_m4,
            'M4_quarterly': Dataset_m4,
            'M4_monthly': Dataset_m4,
            'M4_weekly': Dataset_m4,
            'M4_daily': Dataset_m4,
            'M4_hourly': Dataset_m4,
        }
        Data = data_dict[self.args.data]

        size = [args.input_len, args.pred_len]
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=size,
            features=args.features,
            target=args.target,
            freq=args.freq,
            predict=predict
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = None
        return criterion

    def vali(self, vali_data=None, vali_loader=None, criterion=None):
        self.model.eval()
        total_loss = []
        pred_loss = []
        with torch.no_grad():
            for i, (batch_x) in enumerate(vali_loader):
                if batch_x.ndim == 2:
                    input_x = batch_x.unsqueeze(-1)
                elif batch_x.ndim == 3:
                    input_x = batch_x
                else:
                    print('error!')
                    exit(-1)
                recon, r_true, pred, true = self._process_one_batch(input_x, predict=True)
                loss, loss_pred = loss_process(pred, true, recon, r_true, criterion=criterion, flag=1,
                                               loss_type=self.args.loss)
                total_loss.append(loss)
                pred_loss.append(loss_pred)
            total_loss = np.average(total_loss)
            pred_loss = np.average(pred_loss)
        self.model.train()
        return total_loss, pred_loss

    def train(self, setting=None):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        train_steps = len(train_loader)

        criterion = self._select_criterion()
        lr = self.args.learning_rate

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x) in enumerate(train_loader):
                if batch_x.ndim == 2:
                    input_x = batch_x.unsqueeze(-1)
                elif batch_x.ndim == 3:
                    input_x = batch_x
                else:
                    print('error!')
                    exit(-1)
                model_optim.zero_grad()
                iter_count += 1
                recon, r_true, pred, true = self._process_one_batch(input_x, predict=False)

                loss = loss_process(pred, true, recon, r_true, criterion=criterion, flag=0,
                                    loss_type=self.args.loss)
                loss.backward(loss)
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                            torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss, vali_pred_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_pred_loss = self.vali(test_data, test_loader, criterion)

            print("Pred_len: {0}| Epoch: {1}, Steps: {2} | Total: Vali Loss: {3:.7f} Test Loss: {4:.7f}| "
                  "Pred: Vali Loss:{5:.7f} Test Loss:{6:.7f}".format(self.args.pred_len, epoch + 1,
                                                                     train_steps, vali_loss, test_loss, vali_pred_loss,
                                                                     test_pred_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, (epoch + 1), self.args)

        self.args.learning_rate = lr

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=False, write_loss=True, save_loss=True):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        test_data, test_loader = self._get_data(flag='test', predict=True)
        time_now = time.time()
        if save_loss:
            preds = []
            trues = []
            if 'M4' in self.args.data:
                naive_data = []
                input_data = []
                for i, (batch_x, naive_x) in enumerate(test_loader):
                    batch_x = batch_x.unsqueeze(-1)
                    _, _, pred, true = self._process_one_batch(batch_x, predict=True)
                    naive_x = naive_x.unsqueeze(-1).detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)
                    naive_data.append(naive_x[:, :self.args.pred_len])
                    input_data.append(batch_x.detach().cpu().numpy())
            else:
                for i, (batch_x) in enumerate(test_loader):
                    _, _, pred, true = self._process_one_batch(batch_x, predict=True)
                    pred = pred.detach().cpu().numpy()
                    true = batch_x[:, -self.args.pred_len:].detach().cpu().numpy()
                    if self.args.test_inverse_transform:
                        pred = test_data.inverse_transform(pred)
                        true = test_data.inverse_transform(true)
                    preds.append(pred)
                    trues.append(true)

            print("inference time: {}".format(time.time() - time_now))
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)
            if 'M4' in self.args.data:
                input_data = np.array(input_data)
                naive_data = np.array(naive_data)
                input_data = input_data.reshape(-1, input_data.shape[-2], input_data.shape[-1])
                naive_data = naive_data.reshape(-1, naive_data.shape[-2], naive_data.shape[-1])
                mae, mse, smape, owa = metric(preds, trues, naive_data, input_data, self.args.frequency)
                print('|{}_{}|pred_len{}|mse:{}, mae:{}, smape:{}, owa:{}'.
                      format(self.args.data, self.args.features, self.args.pred_len, mse, mae, smape, owa) + '\n')
            else:
                if 'PEMS' in self.args.data:
                    mae, mape, rmse = metric(preds, trues, data='PEMS')
                    print('|{}_{}|pred_len{}|mae:{}, mape:{}, rmse:{}'.
                          format(self.args.data, self.args.features, self.args.pred_len, mae, mape, rmse) + '\n')
                else:
                    mae, mse = metric(preds, trues)
                    print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                          format(self.args.data, self.args.features, self.args.pred_len, mse, mae) + '\n')
            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if 'PEMS' in self.args.data:
                np.save(folder_path + f'metrics.npy', np.array([mae, mape, rmse]))
            elif 'M4' in self.args.data:
                np.save(folder_path + f'metrics.npy', np.array([mae, mse, smape, owa]))
            else:
                np.save(folder_path + f'metrics.npy', np.array([mae, mse]))
            np.save(folder_path + f'pred.npy', preds)
            np.save(folder_path + f'true.npy', trues)

        else:
            mses = []
            maes = []
            mapes = []
            if 'M4' in self.args.data:
                smapes = []
                owas = []

                for i, (batch_x, naive_x) in enumerate(test_loader):
                    batch_x = batch_x.unsqueeze(-1)
                    _, _, pred, true = self._process_one_batch(batch_x, predict=True)
                    naive_x = naive_x.unsqueeze(-1).detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    input_data = batch_x.detach().cpu().numpy()
                    mae, mse, smape, owa = metric(pred, true, naive_x, input_data, self.args.frequency)
                    mses.append(mse)
                    maes.append(mae)
                    smapes.append(smape)
                    owas.append(owa)

                print("inference time: {}".format(time.time() - time_now))
                mse = np.mean(mses)
                mae = np.mean(maes)
                smape = np.mean(smapes)
                owa = np.mean(owas)
                print('|{}_{}|pred_len{}|smape:{}, owa:{}'.
                      format(self.args.data, self.args.features, self.args.pred_len, smape, owa) + '\n')
            else:
                for i, (batch_x) in enumerate(test_loader):
                    _, _, pred, true = self._process_one_batch(batch_x, predict=True)
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    if self.args.test_inverse_transform:
                        pred = test_data.inverse_transform(pred)
                        true = test_data.inverse_transform(true)
                    mae = np.mean(abs(pred - true))
                    mse = np.mean((pred - true) ** 2)
                    maes.append(mae)
                    mses.append(mse)
                    if 'PEMS' in self.args.data:
                        mape = np.mean(np.abs((pred - true) / true))
                        mapes.append(mape)

                print("inference time: {}".format(time.time() - time_now))
                mae = np.mean(maes)
                mse = np.mean(mses)
                if 'PEMS' in self.args.data:
                    mape = np.mean(mapes)
                    rmse = np.sqrt(mse)
                    print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                          format(self.args.data, self.args.features, self.args.pred_len, mae, mape, rmse) + '\n')
                else:
                    print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                          format(self.args.data, self.args.features, self.args.pred_len, mse, mae) + '\n')

        if write_loss:
            path = './result.log'
            with open(path, "a") as f:
                f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                if 'M4' in self.args.data:
                    f.write('|{}_{}|pred_len{}|mse:,{}, mae:,{}, smape:,{}, owa:,{}'.
                            format(self.args.data, self.args.features, self.args.pred_len, mse, mae, smape, owa) + '\n')
                elif 'PEMS' in self.args.data:
                    f.write('|{}_{}|pred_len{}|mae:,{}, mape:,{}, rmse:,{}'.
                            format(self.args.data, self.args.features, self.args.pred_len, mae, mape, rmse) + '\n')
                else:
                    f.write('|{}_{}|pred_len{}|mse:,{}, mae:,{}'.
                            format(self.args.data, self.args.features, self.args.pred_len, mse, mae) + '\n')
                f.flush()
                f.close()
        else:
            pass

        if not save_loss:
            dir_path = os.path.join(self.args.checkpoints, setting)
            check_path = dir_path + '/' + 'checkpoint.pth'
            if os.path.exists(check_path):
                os.remove(check_path)
                os.removedirs(dir_path)
        if 'M4' in self.args.data:
            return smape, owa
        elif 'PEMS' in self.args.data:
            return mae, mape, rmse
        else:
            return mse, mae

    def _process_one_batch(self, batch_x, predict=False):
        batch_x = batch_x.float().to(self.device)
        input_seq = batch_x[:, :self.args.input_len, :]
        batch_y = batch_x[:, -self.args.pred_len:, :]
        pred_data = torch.zeros_like(batch_y)

        if self.args.mode == 'iterative':
            if predict:
                for i in range(self.args.pred_len):
                    if i:
                        _, pred, _ = self.model(input_seq)
                    else:
                        recon, pred, input_x = self.model(input_seq)
                    pred_data[:, i:i + 1] = pred
                    input_seq = torch.cat([input_seq[:, 1:], pred], dim=1)
            else:
                recon, pred, input_x = self.model(input_seq)
                pred_data = pred
                batch_y = batch_y[:, :1]
        else:
            recon, pred_data, input_x = self.model(input_seq)

        return recon, input_x, pred_data, batch_y

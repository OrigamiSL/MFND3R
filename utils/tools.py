import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args, fine_tuning=False):
    if args.lradj == 'type1':
        args.learning_rate = args.learning_rate * 0.5
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if not fine_tuning:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = -np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


def loss_process(pred, true, recons=None, r_true=None, criterion=None,
                 flag=0, loss_type='mse'):
    input_len = r_true.shape[1]
    pred_len = true.shape[1]
    if loss_type in ['mse', 'mae']:
        if flag == 0:
            loss = criterion(pred, true) * pred_len / (input_len + pred_len)
            loss += criterion(recons, r_true) * input_len / (input_len + pred_len)
            return loss
        elif flag == 1:
            mse = criterion(pred, true)
            loss = mse * pred_len / (input_len + pred_len)
            loss += criterion(recons, r_true) * input_len / (input_len + pred_len)
            return loss.detach().cpu().numpy(), mse.detach().cpu().numpy()
        else:
            loss2 = criterion(pred, true)
            return loss2.detach().cpu().numpy()
    elif loss_type == 'smape':
        smape = SMAPE_loss(pred, true)
        loss = SMAPE_loss(pred, true) * pred_len / (input_len + pred_len) + SMAPE_loss(recons, r_true) * input_len / (input_len + pred_len)
        if flag == 0:
            return loss
        elif flag == 1:
            return loss.detach().cpu().numpy(), smape.detach().cpu().numpy()
        else:
            return smape.detach().cpu().numpy()
    else:
        print('error!')
        exit(-1)


def SMAPE_loss(pred, true, mask=None):
    divide = torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true))
    divide[divide != divide] = .0
    divide[divide == np.inf] = .0
    if mask is not None:
        divide = divide * mask
    smape = 200 * torch.mean(divide)
    return smape

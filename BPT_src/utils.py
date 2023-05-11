import os
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, fbeta_score
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from pretrain.pre_utils import compute_power
from collections import OrderedDict
from scipy.signal import hilbert


def get_emb(x, power, encoder_t, encoder_ch):
    bat_size, ch_num, seq_len, seg_len = x.shape

    # time encoder capture long-term dependency
    time_z = encoder_t(mask=None, data=x, power=power, need_mask=False)  # time_z.shape: new_bat * ch_num, seq_len, d_model

    _, _, d_model = time_z.shape
    time_z = time_z.reshape(bat_size, ch_num, seq_len, d_model)  # time_z.shape: new_bat, ch_num, seq_len, d_model
    time_z = torch.transpose(time_z, 1, 2)                       # time_z.shape: new_bat, seq_len, ch_num, d_model
    time_z = time_z.reshape(bat_size*seq_len, ch_num, d_model)   # time_z.shape: new_bat*seq_len, ch_num, d_model

    emb, _ = encoder_ch(time_z)  # emb.shape: new_bat*seq_len, ch_num, d_model

    emb = emb.reshape(bat_size, seq_len, ch_num, d_model)
    emb = torch.transpose(emb, 1, 2)

    return emb


def _load_data(_dir, pat_file, cal_power, need_y):
    _data = np.load(os.path.join(_dir, f'{pat_file}/data.npy'))
    _y = np.load(os.path.join(_dir, f'{pat_file}/label.npy')) if need_y else None
    if cal_power:
        _power = compute_power(_data, 256)
    else:
        _power = np.load(os.path.join(_dir, f'{pat_file}/power.npy'))

    return _data, _power, _y


def load_data(pats_name, cal_power=False, need_y=True):
    data, power, y = [], [], []
    for idx, pat_name in enumerate(tqdm(pats_name, disable=True)):
        _dir = ...  # data directory
        pat_file = ...  # the selected file from pat_name
        _data, _power, _y = _load_data(_dir, pat_file, cal_power=cal_power, need_y=need_y)
        data.append(_data); power.append(_power); y.append(_y)

    return data, power, y


def split_data_ts(x, power, y, src_ratio, val_ratio, test_ratio, need_y):
    pat_num = len(x)
    test_x, test_power, test_y = [], [], []
    for i in range(pat_num):
        _x, _power = x[i], power[i]
        _y = y[i] if need_y else None
        sample_num = _x.shape[1]
        perm = torch.randperm(sample_num)
        _x = _x[:, perm]
        _power = _power[:, perm]
        _y = _y[:, perm] if need_y else None
        src_sample_end = int(sample_num*src_ratio)
        val_sample_end = src_sample_end + int(sample_num*val_ratio)
        test_sample_end = val_sample_end + int(sample_num*test_ratio)

        test_x.append(_x[:, val_sample_end:test_sample_end])
        test_power.append(_power[:, val_sample_end:test_sample_end])
        test_y.append(_y[:, val_sample_end:test_sample_end] if need_y else None)

    return [test_x, test_power, test_y]


def split_data_trvlts(x, power, y, src_ratio, val_ratio, test_ratio, need_y):
    pat_num = len(x)
    src_x, src_power, src_y = [], [], [],
    val_x, val_power, val_y = [], [], [],
    test_x, test_power, test_y = [], [], []
    for i in range(pat_num):
        _x, _power = x[i], power[i]
        _y = y[i] if need_y else None
        sample_num = _x.shape[1]
        perm = torch.randperm(sample_num)
        _x = _x[:, perm]
        _power = _power[:, perm]
        _y = _y[:, perm] if need_y else None

        src_val_sample_end = int(sample_num*(src_ratio+val_ratio))
        # test data
        test_sample_end = src_val_sample_end + int(sample_num*test_ratio)
        test_x.append(_x[:, src_val_sample_end:test_sample_end])
        test_power.append(_power[:, src_val_sample_end:test_sample_end])
        test_y.append(_y[:, src_val_sample_end:test_sample_end] if need_y else None)

        _src_val_x     = _x[:, :src_val_sample_end]
        _src_val_power = _power[:, :src_val_sample_end]
        _src_val_y     = _y[:, :src_val_sample_end] if need_y else None

        src_val_perm = torch.randperm(src_val_sample_end)
        _src_val_x     = _src_val_x     [:, src_val_perm]
        _src_val_power = _src_val_power [:, src_val_perm]
        _src_val_y     = _src_val_y     [:, src_val_perm] if need_y  else None

        src_sample_num = int(max(src_val_sample_end * src_ratio, 1))
        val_sample_num = int(max(src_val_sample_end * val_ratio, 1))

        src_x.append(    _src_val_x    [:, :src_sample_num])
        src_power.append(_src_val_power[:, :src_sample_num])
        src_y.append(    _src_val_y    [:, :src_sample_num] if need_y else None)

        val_x.append(    _src_val_x    [:, -val_sample_num:])
        val_power.append(_src_val_power[:, -val_sample_num:])
        val_y.append(    _src_val_y    [:, -val_sample_num:] if need_y else None)

    return [src_x, src_power, src_y], \
           [val_x, val_power, val_y], \
           [test_x, test_power, test_y]


class Metrics:
    def __init__(self, pred, true, use_prob=False):
        pred, true = np.array(pred), np.array(true)
        if use_prob:
            threshold = 0.5
            pred = (pred >= threshold)
        if np.sum(true) == 0 and np.sum(pred) == 0:
            self.special_good = True
        else:
            self.special_good = False

        self.tn = np.sum((pred == 0) & (true == 0))
        self.tp = np.sum((pred == 1) & (true == 1))
        self.fn = np.sum((pred == 0) & (true == 1))
        self.fp = np.sum((pred == 1) & (true == 0))
        self.acc = accuracy_score(true, pred)
        self.prec, self.rec, *_ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
        self.f_half = fbeta_score(true, pred, average='binary', beta=0.5)
        self.f_one  = fbeta_score(true, pred, average='binary', beta=1)
        self.f_doub = fbeta_score(true, pred, average='binary', beta=2)

    def get_confusion(self):
        return f"TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn} " if not self.special_good else "special_good"

    def get_metrics(self, one_line=False):
        if one_line:
            out = 'Acc:%.4f Prec:%.4f Rec:%.4f F1:%.4f F2:%.4f' \
                  % (self.acc, self.prec, self.rec, self.f_one, self.f_doub)
        else:
            out = ''
            out += '-' * 15 + 'Metrics' + '-' * 15 + '\n'
            out += 'Accuracy:  ' + str(self.acc) + '\n'
            out += 'Precision: ' + str(self.prec) + '\n'
            out += 'Recall:    ' + str(self.rec) + '\n'
            out += 'F1:        ' + str(self.f_one) + '\n'
            out += 'F2:        ' + str(self.f_doub) + '\n'
        return out if not self.special_good else "special_good"

    @staticmethod
    def fbeta(precision, recall, beta):
        return (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)


class UnsupervisedMetrics:
    def __init__(self, pred, true):
        mae_cal   = MeanAbsoluteError()
        self.mae  = mae_cal(pred, true)
        mape_cal  = MeanAbsolutePercentageError()
        self.mape = mape_cal(pred, true)
        mse_cal   = MeanSquaredError()
        self.mse  = mse_cal(pred, true)
        self.rmse = torch.sqrt(self.mse)

    def get_metrics(self, one_line=False):
        if one_line:
            out = 'MAE:%.4f MAPE:%.4f MSE:%.4f RMSE:%.4f' \
                  % (self.mae, self.mape, self.mse, self.rmse)
        else:
            out = ''
            out += '-' * 15 + 'Metrics' + '-' * 15 + '\n'
            out += 'MAE:  ' + str(self.mae) + '\n'
            out += 'MAPE: ' + str(self.mape) + '\n'
            out += 'MSE:  ' + str(self.mse) + '\n'
            out += 'RMSE: ' + str(self.rmse)
        return out


class PhFreqMetrics:
    def __init__(self, pred_ph, true_ph, pred_freq, true_freq):
        # ph
        delta = true_ph - pred_ph
        l = pred_ph.shape[0]

        zeros = torch.zeros_like(delta)
        t = torch.transpose(
            torch.concat([zeros, delta]).reshape(2, l),
            0, 1
        )
        expo = torch.view_as_complex(t.contiguous())

        res = torch.exp(expo)
        res = torch.sum(res) / l
        self.ph_plv = torch.norm(res)

        # freq
        mae_cal = MeanAbsoluteError()
        self.freq_mae = mae_cal(pred_freq, true_freq)

    def get_metrics(self, one_line=False):
        if one_line:
            out = 'phase_PLV:%.4f freq_MAE:%.8f ' \
                  % (self.ph_plv, self.freq_mae)
        else:
            out = ''
            out += '-' * 15 + 'Metrics' + '-' * 15 + '\n'
            out += 'phase_PLV: ' + str(self.ph_plv) + '\n'
            out += 'freq_MAE:  ' + str(self.freq_mae)
        return out


def unwrap_ddp(state_dict: OrderedDict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value

    return new_state_dict


def get_metric(metr, metr_name):
    if metr_name == 'acc':
        return metr.acc
    elif metr_name == 'f0.5':
        return metr.f_half
    elif metr_name == 'f1':
        return metr.f_one
    elif metr_name == 'f2':
        return metr.f_doub
    elif metr_name == 'prec':
        return metr.prec
    elif metr_name == 'rec':
        return metr.rec
    else:
        raise NotImplementedError


def get_seg_property(args, fut_seg):
    w = torch.fft.fft(fut_seg.cpu(), dim=-1)
    f = torch.fft.fftfreq(fut_seg.cpu().shape[-1])

    freq = [ f[ int(torch.argmax(w[b,:].abs())) ]
             for b in range(fut_seg.shape[0])
            ]

    analytic_signal = hilbert(fut_seg.cpu())
    instantaneous_phase = np.angle(analytic_signal)

    phase = torch.tensor(instantaneous_phase)
    freq  = torch.tensor(freq).reshape(-1, 1)
    return phase.to(args.device), \
           freq .to(args.device)
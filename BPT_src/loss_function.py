import os
import pickle

import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from tqdm import tqdm

root_sampled_data_path = "/data/zdz/EEG/div_all_sampled_data"
root_brain_region_path = '/data/eeggroup/new_data2'



# def NCELoss(basic_cfg, his_x, seg_x, fut_x,):
#     def func(z, W, c):
#         return torch.exp(z*W*c)
#
#     z = encoder(seg_x[:, :, :])
#     c = ar_lstm(z)
#     for j in range(basic_cfg.f['his_len']):
#         z_j = encoder(his_x[:, j, :, :])
#         denominator += func(z_j, Wk, c)
#     for k in range(basic_cfg.d['fut_len']):
#         z_k = encoder(fut_x[:, k, :, :])
#         numerator = func(z_k, Wk, c)



def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def loss_weight_cal(config, patients):
    fre = np.zeros(2*(90+1))
    num_points = 0
    for patient in patients:
        pat_y = pickle.load(open(os.path.join(root_sampled_data_path, f'y_{patient}.pkl'), 'rb'))
        if config.src_patients[0] == 't01_14ki':
            pat_y = pat_y[:-1]

        subject = config.pat2subject[patient]
        suffix = subject + '/brain_dict.pkl'
        dir = os.path.join(root_brain_region_path, suffix)
        with open(dir, 'rb') as f:
            brain_dict = pickle.load(f)

        for ch_idx, ch_y in enumerate(pat_y):
            brain_region = get_brain_region(brain_dict, ch_idx)
            fre[2*brain_region] += np.sum(ch_y == 0)
            fre[2*brain_region+1] += np.sum(ch_y == 1)

            num_points += len(ch_y)

    fre = torch.tensor(fre / num_points, dtype=torch.float32)

    alpha = torch.tensor(0.01)
    use_log = False
    weight = torch.log(1 / (fre+alpha)) if use_log else 1 / (fre + alpha)

    return weight.to(config.device)





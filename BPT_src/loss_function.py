import os
import pickle

import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from tqdm import tqdm

root_sampled_data_path = "/path/to/sampled/data/"
root_brain_region_path = "/path/to/brain/region/"




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
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)     # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1)) 
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft) 

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





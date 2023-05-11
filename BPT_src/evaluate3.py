import torch
from torch.utils.data import DataLoader
from dataset import BoardDataset
from utils import UnsupervisedMetrics, get_emb


def evaluate3(args, val_data_list, modules, patients, linear,):
    encoder_t, encoder_ch = modules
    val_x, val_power, _ = val_data_list

    mse_loss = torch.nn.MSELoss(reduction='mean')

    eval_loss = 0
    tot_pred, tot_true = [], []

    with torch.no_grad():
        encoder_t.eval()
        encoder_ch.eval()
        linear.eval()

        for p_idx, patient in enumerate(patients):

            dataset = BoardDataset(data=val_x[p_idx],
                                   power=val_power[p_idx],
                                   y_label=None,
                                   need_y=False)

            data_iter = DataLoader(dataset, shuffle=False, batch_size=args.infer_batch_size, drop_last=False)

            for bat_idx, (x, power) in enumerate(data_iter):

                x, power = x.to(args.device), power.to(args.device)

                ori_x = x.clone()

                mask = torch.rand_like(x).to(args.device)
                mask[mask <= args.main3_mask_rate] = 0  # mask=0 masked
                mask[mask > args.main3_mask_rate] = 1   # mask=1 remained
                x = x.masked_fill(mask == 0, 0)

                emb = get_emb(x, power, encoder_t, encoder_ch)

                rec_x = linear(emb)

                bat_loss = mse_loss(ori_x[mask == 0], rec_x[mask == 0])

                eval_loss += bat_loss.cpu().item()

                tot_pred.append(rec_x.cpu().view(-1))
                tot_true.append(ori_x.cpu().view(-1))

    tot_pred, tot_true = torch.concat(tot_pred), torch.concat(tot_true)
    uns_metr = UnsupervisedMetrics(tot_pred, tot_true)

    return eval_loss, uns_metr

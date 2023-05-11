import torch
from torch.utils.data import DataLoader
from dataset import BoardDataset
from utils import Metrics, get_emb


def evaluate1(args, val_data_list, modules, mlp, patients):
    encoder_t, encoder_ch = modules
    val_x, val_power, val_y = val_data_list

    tot_pred, tot_label, tot_prob = [], [], []
    eval_loss = 0
    with torch.no_grad():
        encoder_t.eval()
        encoder_ch.eval()
        mlp.eval()

        for p_idx, patient in enumerate(patients):
            dataset = BoardDataset(data=val_x[p_idx],
                                   power=val_power[p_idx],
                                   y_label=val_y[p_idx],
                                   need_y=True)

            data_iter = DataLoader(dataset, shuffle=False, batch_size=args.infer_batch_size, drop_last=False)

            loss_fn = torch.nn.CrossEntropyLoss()
            for bat_idx, (x, power, label) in enumerate(data_iter):
                x, power, label = x.to(args.device), power.to(args.device), label.to(args.device).long()
                bat_size, ch_num, seq_len, seg_len = x.shape

                emb = get_emb(x, power, encoder_t, encoder_ch)
                emb = emb.reshape(bat_size * ch_num * seq_len, -1)

                logit = mlp(emb)
                pred = torch.argmax(logit, dim=-1)
                label = label.reshape(-1)
                bat_loss = loss_fn(logit, label)
                eval_loss += bat_loss.cpu().item()

                tot_label.append(label.cpu())
                tot_pred.append(pred.cpu())
                tot_prob.append(logit.cpu())

        tot_pred, tot_label, tot_prob = torch.concat(tot_pred), torch.concat(tot_label), torch.concat(tot_prob, dim=0)
        eval_metr = Metrics(tot_pred, tot_label)

    return eval_loss, eval_metr

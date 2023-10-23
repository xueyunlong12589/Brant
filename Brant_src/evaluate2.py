import torch
from torch.utils.data import DataLoader
from dataset import BoardDataset
from utils import UnsupervisedMetrics, PhFreqMetrics, get_emb, get_seg_property


def evaluate2(args, val_data_list, modules, patients, his_len, fut_len, pred_head, ):
    encoder_t, encoder_ch = modules
    val_x, val_power, _ = val_data_list

    mse_loss = torch.nn.MSELoss(reduction='mean')

    if args.main2_task == 'ph_freq':
        eval_ph_loss, eval_freq_loss = 0, 0
        tot_pred_ph,   tot_true_ph   = [], []
        tot_pred_freq, tot_true_freq = [], []
    else:
        eval_fore_loss = 0
        tot_pred, tot_true = [], []

    with torch.no_grad():
        encoder_t.eval()
        encoder_ch.eval()

        if args.main2_task == 'ph_freq':
            pred_head_freq, pred_head_ph = pred_head
            pred_head_freq.eval()
            pred_head_ph.eval()
        else:
            pred_head.eval()

        for p_idx, patient in enumerate(patients):

            val_x[p_idx] = val_x[p_idx][:, :, :his_len + fut_len]
            val_power[p_idx] = val_power[p_idx][:, :, :his_len + fut_len]
            dataset = BoardDataset(data=val_x[p_idx],
                                   power=val_power[p_idx],
                                   y_label=None,
                                   need_y=False)

            data_iter = DataLoader(dataset, shuffle=False, batch_size=args.infer_batch_size, drop_last=False)

            for bat_idx, (x, power) in enumerate(data_iter):

                x, power = x.to(args.device), power.to(args.device)
                bat_size, ch_num, seq_len, seg_len = x.shape

                his_x = x[:, :, :his_len]
                his_pow = power[:, :, :his_len]

                fut_x = x[:, :, -fut_len:]
                emb = get_emb(his_x, his_pow, encoder_t, encoder_ch)
                emb = emb.reshape(bat_size * ch_num, his_len, args.d_model)
                emb = torch.mean(emb, dim=-2)

                if args.main2_task == 'ph_freq':
                    pred_freq = pred_head_freq(emb)
                    pred_ph   = pred_head_ph(emb)

                    ph, freq = get_seg_property(args, fut_x.reshape(bat_size*ch_num, -1) )
                    mean, std = torch.mean(ph,   dim=-1, keepdim=True), torch.std(ph,   dim=-1, keepdim=True)
                    ph   = (ph   - mean) / std

                    bat_ph_loss = mse_loss(pred_ph, ph)
                    bat_freq_loss = mse_loss(pred_freq, freq)

                    eval_ph_loss += bat_ph_loss.cpu().item()
                    eval_freq_loss += bat_freq_loss.cpu().item()

                    tot_pred_ph  .append(pred_ph.cpu().view(-1))
                    tot_true_ph  .append(ph     .cpu().view(-1))
                    tot_pred_freq.append(pred_freq.cpu().view(-1))
                    tot_true_freq.append(freq     .cpu().view(-1))
                else:
                    forecast = pred_head(emb)

                    fut_x = fut_x.reshape(bat_size * ch_num, -1)
                    if args.main2_task == 'long':
                        fut_x = fut_x[:, ::args.main2_long_term_dr]
                    else:
                        fut_x = fut_x[:, ::args.main2_short_term_dr]
                    bat_fore_loss = mse_loss(forecast, fut_x)

                    eval_fore_loss += bat_fore_loss.cpu().item()

                    tot_pred.append(forecast.cpu().view(-1))
                    tot_true.append(fut_x.cpu().view(-1))

            del dataset, data_iter

    if args.main2_task == 'ph_freq':
        ret = (eval_ph_loss, eval_freq_loss)
        tot_pred_ph, tot_true_ph, tot_pred_freq, tot_true_freq = torch.concat(tot_pred_ph), torch.concat(tot_true_ph), torch.concat(tot_pred_freq), torch.concat(tot_true_freq)
        pf_metr = PhFreqMetrics(tot_pred_ph, tot_true_ph, tot_pred_freq, tot_true_freq)
        ret += (pf_metr, )
    else:
        ret = (eval_fore_loss, )
        tot_pred, tot_true = torch.concat(tot_pred), torch.concat(tot_true)
        uns_metr = UnsupervisedMetrics(tot_pred, tot_true)
        ret += (uns_metr, )

    return ret

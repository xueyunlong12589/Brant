import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BoardDataset
from evaluate2 import evaluate2
from utils import get_seg_property
from utils import load_data, get_emb, split_data_trvlts


def train2(args, modules, pred_head):
    encoder_t, encoder_ch = modules

    mse_loss = nn.MSELoss(reduction='mean')

    if args.main2_mode == 'finetune':
        enc_lr = args.ft_lr
    else:
        enc_lr = 0
    head_lr = args.lr

    if args.main2_task == 'ph_freq':
        pred_head_freq, pred_head_ph = pred_head
        optimizer = torch.optim.Adam(
            [{'params': list(encoder_t.parameters()), 'lr': enc_lr},
             {'params': list(encoder_ch.parameters()), 'lr': enc_lr},
             {'params': list(pred_head_freq.parameters()), 'lr': head_lr},
             {'params': list(pred_head_ph  .parameters()), 'lr': head_lr},],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        optimizer = torch.optim.Adam(
            [{'params': list(encoder_t.parameters()),   'lr': enc_lr},
             {'params': list(encoder_ch.parameters()),   'lr': enc_lr},
             {'params': list(pred_head.parameters()),    'lr': head_lr}, ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)

    patients = ...  # the list of patients

    all_x, all_power, all_y = load_data(patients, need_y=False,)

    src_data_list, val_data_list, test_data_list = split_data_trvlts(all_x, all_power, all_y,
                                                                     src_ratio= args.src_ratio,
                                                                     val_ratio= args.val_ratio,
                                                                     test_ratio=args.test_ratio,
                                                                     need_y=False)

    optimal = torch.inf
    best_epo_idx = -1
    src_x, src_power, _ = src_data_list
    his_len = args.seq_len
    if args.main2_task == 'long':
        fut_len = args.main2_long_term_pred_len
    elif args.main2_task == 'short':
        fut_len = args.main2_short_term_pred_len
    elif args.main2_task == 'ph_freq':
        fut_len = args.main2_ph_freq_pred_len

    for epoch in range(args.num_epochs):
        encoder_t.train()
        encoder_ch.train()

        if args.main2_task == 'ph_freq':
            pred_head_freq.train()
            pred_head_ph.train()
        else:
            pred_head.train()

        if args.main2_task == 'ph_freq':
            epo_ph_loss, epo_freq_loss = 0, 0
        else:
            epo_fore_loss = 0

        for p_idx, patient in enumerate(patients):

            src_x[p_idx] = src_x[p_idx][:, :, :his_len + fut_len]
            src_power[p_idx] = src_power[p_idx][:, :, :his_len + fut_len]
            dataset = BoardDataset(data=src_x[p_idx],
                                   power=src_power[p_idx],
                                   y_label=None,
                                   need_y=False)

            data_iter = DataLoader(dataset, shuffle=False, batch_size=args.train_batch_size, drop_last=False)

            for bat_idx, (x, power) in enumerate(data_iter):

                x, power = x.to(args.device), power.to(args.device)
                bat_size, ch_num, seq_len, seg_len = x.shape

                his_x = x[:, :, :his_len]
                his_pow = power[:, :, :his_len]

                fut_x = x[:, :, -fut_len:]
                emb = get_emb(his_x, his_pow, encoder_t, encoder_ch)
                emb = emb.reshape(bat_size*ch_num, his_len, args.d_model)
                emb = torch.mean(emb, dim=-2)

                if args.main2_task == 'ph_freq':
                    pred_freq = pred_head_freq(emb)
                    pred_ph   = pred_head_ph(emb)

                    ph, freq = get_seg_property(args, fut_x.reshape(bat_size*ch_num, -1))
                    mean, std = torch.mean(ph,   dim=-1, keepdim=True), torch.std(ph,   dim=-1, keepdim=True)
                    ph = ((ph - mean) / std).float()

                    bat_ph_loss = mse_loss(pred_ph, ph)
                    bat_freq_loss = mse_loss(pred_freq, freq)

                    optimizer.zero_grad()
                    (bat_ph_loss+bat_freq_loss).backward()
                    optimizer.step()

                    epo_ph_loss   += bat_ph_loss.cpu().item()
                    epo_freq_loss += bat_freq_loss.cpu().item()
                else:
                    forecast = pred_head(emb)

                    fut_x = fut_x.reshape(bat_size*ch_num, -1)
                    if args.main2_task == 'long':
                        fut_x = fut_x[:, ::args.main2_long_term_dr]
                    else:
                        fut_x = fut_x[:, ::args.main2_short_term_dr]
                    bat_fore_loss = mse_loss(forecast, fut_x)

                    optimizer.zero_grad()
                    bat_fore_loss.backward()
                    optimizer.step()

                    epo_fore_loss += bat_fore_loss.cpu().item()

        scheduler.step()

        if epoch >= args.warm_up_epo:
            modules = [encoder_t, encoder_ch]

            eval_res = evaluate2(args, val_data_list, modules, patients, his_len, fut_len, pred_head, )
            if args.main2_task == 'ph_freq':
                val_ph_loss, val_freq_loss, val_metr = eval_res
                cur = val_ph_loss + val_freq_loss
            else:
                val_loss, val_metr = eval_res
                cur = val_loss
                
            if cur < optimal:
                optimal = cur
                best_epo_idx = epoch
                torch.save(encoder_t.state_dict(), f'./model_ckpt/encoder_t_2_{args.main2_task}.pt')
                torch.save(encoder_ch.state_dict(), f'./model_ckpt/encoder_ch_2_{args.main2_task}.pt')
                if args.main2_task == 'ph_freq':
                    torch.save(pred_head_ph  .state_dict(), f'./model_ckpt/pred_head_ph_{args.main2_task}.pt')
                    torch.save(pred_head_freq.state_dict(), f'./model_ckpt/pred_head_freq_{args.main2_task}.pt')
                else:
                    torch.save(pred_head.state_dict(), f'./model_ckpt/pred_head_{args.main2_task}.pt')
                print('performance on validation set increases')

    return best_epo_idx


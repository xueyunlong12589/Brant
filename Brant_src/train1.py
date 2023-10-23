from dataset import BoardDataset
from evaluate1 import evaluate1
import torch
from torch.utils.data import DataLoader

from utils import load_data, get_emb, split_data_trvlts, get_metric


def train1(args, modules, mlp):
    encoder_t, encoder_ch = modules
    loss_func_clsf = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [{'params': list(encoder_t.parameters()),   'lr': args.ft_lr},
         {'params': list(encoder_ch.parameters()),   'lr': args.ft_lr},
         {'params': list(mlp.parameters()),        'lr': args.lr}],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)

    # --------- read data ---------
    all_x, all_power, all_y = load_data(patients, need_y=True,)

    src_data_list, val_data_list, test_data_list = split_data_trvlts(all_x, all_power, all_y,
                                                                     src_ratio= args.src_ratio,
                                                                     val_ratio= args.val_ratio,
                                                                     test_ratio=args.test_ratio,
                                                                     need_y=True,
                                                                     )

    patients = ...  # the list of patients

    optimal = -torch.inf
    best_epo_idx = -1
    src_x, src_power, src_y = src_data_list
    for epoch in range(args.num_epochs):

        encoder_t.train()
        encoder_ch.train()
        mlp.train()

        epo_loss_clsf = 0
        for p_idx, patient in enumerate(patients):

            dataset = BoardDataset(data=src_x[p_idx],
                                   power=src_power[p_idx],
                                   y_label=src_y[p_idx],
                                   need_y=True)

            data_iter = DataLoader(dataset, shuffle=True, batch_size=args.train_batch_size, drop_last=False)

            for bat_idx, (x, power, label) in enumerate(data_iter):
                x, power, label = x.to(args.device), power.to(args.device), label.to(args.device).long()
                bat_size, ch_num, seq_len, seg_len = x.shape

                emb = get_emb(x, power, encoder_t, encoder_ch)
                emb = emb.reshape(bat_size * ch_num * seq_len, -1)

                logit = mlp(emb)
                label = label.reshape(-1)
                bat_loss = loss_func_clsf(logit, label)

                optimizer.zero_grad()
                bat_loss.backward()
                optimizer.step()

                epo_loss_clsf += bat_loss.cpu().item()

        if epoch >= args.warm_up_epo:
            modules = [encoder_t, encoder_ch]
            val_loss, val_metr = evaluate1(args, val_data_list, modules, mlp, patients)

            cur = get_metric(val_metr, metr_name=args.metric)
            if cur > optimal:
                optimal = cur
                best_epo_idx = epoch
                torch.save(encoder_t.state_dict(), './model_ckpt/encoder_t_1.pt')
                torch.save(encoder_ch.state_dict(), './model_ckpt/encoder_ch_1.pt')
                torch.save(mlp.state_dict(), './model_ckpt/clsf_mlp_1.pt')
                print('performance on validation set increases')

        scheduler.step()

    return best_epo_idx


import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BoardDataset
from evaluate3 import evaluate3
from utils import load_data, get_emb, split_data_trvlts


def train3(args, modules, linear):
    encoder_t, encoder_ch = modules

    mse_loss = nn.MSELoss()
    enc_lr = args.ft_lr
    lin_lr = args.lr

    optimizer = torch.optim.Adam(
        [{'params': list(encoder_t.parameters()),   'lr': enc_lr},
         {'params': list(encoder_ch.parameters()),  'lr': enc_lr},
         {'params': list(linear.parameters()),      'lr': lin_lr}, ],
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
                                                                     need_y=False,
                                                                     )

    optimal = torch.inf
    best_epo_idx = -1
    src_x, src_power, _ = src_data_list

    for epoch in range(args.num_epochs):
        encoder_t.train()
        encoder_ch.train()
        linear.train()

        epo_loss = 0
        for p_idx, patient in enumerate(patients):

            dataset = BoardDataset(data=src_x[p_idx],
                                   power=src_power[p_idx],
                                   y_label=None,
                                   need_y=False)

            data_iter = DataLoader(dataset, shuffle=False, batch_size=args.train_batch_size, drop_last=False)

            for bat_idx, (x, power) in enumerate(data_iter):
                x, power = x.to(args.device), power.to(args.device)
                ori_x = x.clone()

                mask = torch.rand_like(x).to(args.device)
                mask[mask <= args.main3_mask_rate] = 0    # mask=0 masked
                mask[mask >  args.main3_mask_rate] = 1    # mask=1 remained
                x = x.masked_fill(mask==0, 0)

                emb = get_emb(x, power, encoder_t, encoder_ch)

                rec_x = linear(emb)

                bat_loss = mse_loss(ori_x[mask==0], rec_x[mask==0])

                optimizer.zero_grad()
                bat_loss.backward()
                optimizer.step()

                epo_loss += bat_loss.cpu().item()

        scheduler.step()

        if epoch >= args.warm_up_epo:
            modules = [encoder_t, encoder_ch]

            val_loss, val_metr = evaluate3(args, val_data_list, modules, patients, linear)

            cur = val_loss
            if cur < optimal:
                optimal = cur
                best_epo_idx = epoch
                torch.save(encoder_t.state_dict(),  './model_ckpt/encoder_t_3.pt')
                torch.save(encoder_ch.state_dict(), './model_ckpt/encoder_ch_3.pt')
                torch.save(linear.state_dict(), './model_ckpt/linear_3.pt')
                print('performance on validation set increases')

    return best_epo_idx




from torch import nn

from train1 import train1
from evaluate1 import evaluate1
from train2 import train2
from evaluate2 import evaluate2
from train3 import train3
from evaluate3 import evaluate3
from pretrain.pre_model import TimeEncoder, ChannelEncoder
from utils import load_data, split_data_ts

import argparse
import time
import numpy as np
import torch
import random

from utils import unwrap_ddp

from model import MLP


def load_encoder(args):
    encoder_t = TimeEncoder(in_dim=args.seg_len,
                            d_model=args.d_model,
                            dim_feedforward=args.dim_feedforward,
                            seq_len=args.seq_len,
                            n_layer=args.time_ar_layer,
                            nhead=args.time_ar_head,
                            band_num=args.band_num,
                            project_mode=args.input_emb_mode,
                            learnable_mask=args.learnable_mask).to(args.device)
    encoder_ch = ChannelEncoder(out_dim=args.seg_len,
                                d_model=args.d_model,
                                dim_feedforward=args.dim_feedforward,
                                n_layer=args.ch_ar_layer,
                                nhead=args.ch_ar_head).to(args.device)

    # --------- pretrained model loading ---------
    if args.load_pretrained:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.gpu_id}
        t_state_dict = torch.load(torch.load(f'./pretrain/encoder_ckpt/time_encoder_{args.start_epo_idx}.pt', map_location=map_location))
        ch_state_dict = torch.load(torch.load(f'./pretrain/encoder_ckpt/channel_encoder_{args.start_epo_idx}.pt', map_location=map_location))
        if args.unwrap_ddp:
            t_state_dict = unwrap_ddp(t_state_dict)
            ch_state_dict = unwrap_ddp(ch_state_dict)

        encoder_t.load_state_dict(t_state_dict)
        encoder_ch.load_state_dict(ch_state_dict)
        print('----- Pretrained Models Loaded -----\n')

    if args.freeze_encoder:
        for param in encoder_t.parameters():
            param.requires_grad = False
        for param in encoder_ch.parameters():
            param.requires_grad = False
    return encoder_t, encoder_ch


def model_prepare():
    encoder_t, encoder_ch = load_encoder(args)

    module = (encoder_t, encoder_ch)
    emb_dim = args.d_model

    if args.main_task == 1:
        mlp = MLP(in_dim=emb_dim, out_dim=2).to(args.device)
        module += (mlp, )

    elif args.main_task == 2:
        in_dim = args.d_model
        if args.main2_task == 'ph_freq':
            out_dim = args.main2_ph_freq_pred_len * args.seg_len
            pred_head_freq = nn.Sequential(nn.Linear(in_dim, 1), nn.Dropout(p=0.2), ).to(args.device)
            pred_head_ph = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Dropout(p=0.2), ).to(args.device)
            module += (pred_head_freq, pred_head_ph, )
        elif args.main2_task == 'long':
            out_dim = args.main2_long_term_pred_len * args.seg_len //  args.main2_long_term_dr
            pred_head = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Dropout(p=0.2), ).to(args.device)
            module += (pred_head,)
        elif args.main2_task == 'short':
            out_dim = args.main2_short_term_pred_len * args.seg_len //  args.main2_short_term_dr
            pred_head = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Dropout(p=0.1), ).to(args.device)
            module += (pred_head,)

    elif args.main_task  == 3:
        linear = nn.Linear(in_features=emb_dim, out_features=args.seg_len).to(args.device)
        module += (linear,)

    return module


def train(module):
    # --- main1 ---
    if args.main_task == 1:
        encoder_t, encoder_ch, mlp = module
        best_epo_idx = train1(args, [encoder_t, encoder_ch], mlp)
        print(f'On classification task: best epoch index = {best_epo_idx}')

    # --- main2 ---
    elif args.main_task == 2:
        if args.main2_task == 'ph_freq':
            encoder_t, encoder_ch, pred_head_freq, pred_head_ph, = module
            pred_head = (pred_head_freq, pred_head_ph)
        else:
            encoder_t, encoder_ch, pred_head, = module
        best_epo_idx = train2(args, [encoder_t, encoder_ch], pred_head)
        print(f'On forecasting task: best epoch index = {best_epo_idx}')

    # --- main3 ---
    elif args.main_task == 3:
        encoder_t, encoder_ch, linear = module
        best_epo_idx = train3(args, [encoder_t, encoder_ch], linear)
        print(f'On _imputation task: best epoch index = {best_epo_idx}')

    else:
        raise NotImplementedError("undefined main task idx")


def test(patients, module):
    need_y = True if args.main_task == 1 else False
    all_x, all_power, all_y = load_data(patients, need_y=need_y, )

    test_data_list = split_data_ts(all_x, all_power, all_y,
                                   src_ratio=args.src_ratio,
                                   val_ratio=args.val_ratio,
                                   test_ratio=args.test_ratio,
                                   need_y=need_y,
                                   )
    
    # --- main1 ---
    if args.main_task == 1:
        encoder_t, encoder_ch, mlp, _ = module
        encoder_t.load_state_dict(torch.load('./model_ckpt/encoder_t_1.pt', map_location='cpu'))
        encoder_t.to(args.device)
        encoder_ch.load_state_dict(torch.load('./model_ckpt/encoder_ch_1.pt', map_location='cpu'))
        encoder_ch.to(args.device)
        mlp.load_state_dict(torch.load('./model_ckpt/clsf_mlp_1.pt', map_location='cpu'))
        mlp.to(args.device)
        modules = [encoder_t, encoder_ch]
        test_loss, test_metr = evaluate1(args, test_data_list, modules, mlp, patients)
        print(f"On test:", test_metr.get_metrics(one_line=False), test_metr.get_confusion())

    # --- main2 ---
    elif args.main_task == 2:
        his_len = args.seq_len
        if args.main2_task == 'long':
            fut_len = args.main2_long_term_pred_len
        elif args.main2_task == 'short':
            fut_len = args.main2_short_term_pred_len
        elif args.main2_task == 'ph_freq':
            fut_len = args.main2_ph_freq_pred_len
        # test
        if args.main2_task == 'ph_freq':
            encoder_t, encoder_ch, pred_head_freq, pred_head_ph, = module
        else:
            encoder_t, encoder_ch, pred_head, = module
        encoder_t.load_state_dict(torch.load(f'./model_ckpt/encoder_t_2_{args.main2_task}.pt', map_location='cpu'))
        encoder_t.to(args.device)
        encoder_ch.load_state_dict(torch.load(f'./model_ckpt/encoder_ch_2_{args.main2_task}.pt', map_location='cpu'))
        encoder_ch.to(args.device)

        if args.main2_task == 'ph_freq':
            pred_head_freq.load_state_dict(torch.load(f'./model_ckpt/pred_head_freq_{args.main2_task}.pt', map_location='cpu'))
            pred_head_freq.to(args.device)
            pred_head_ph.load_state_dict(torch.load(f'./model_ckpt/pred_head_ph_{args.main2_task}.pt', map_location='cpu'))
            pred_head_ph.to(args.device)
            pred_head = (pred_head_freq, pred_head_ph)
        else:
            pred_head.load_state_dict(torch.load(f'./model_ckpt/pred_head_{args.main2_task}.pt', map_location='cpu'))
            pred_head.to(args.device)
        modules = [encoder_t, encoder_ch]
        test_res = evaluate2(args, test_data_list, modules, patients, his_len, fut_len, pred_head, )
        if args.main2_task == 'ph_freq':
            test_ph_loss, test_freq_loss, test_metr = test_res
            print("On test: ph_loss = %.4f, freq_loss = %.4f " % (test_ph_loss, test_freq_loss),
                  test_metr.get_metrics(one_line=False))
        else:
            test_loss, test_metr = test_res
            print("On test: loss = %.4f " % test_loss, test_metr.get_metrics(one_line=False))

    # --- main3 ---
    elif args.main_task == 3:
        encoder_t, encoder_ch, linear = module
        encoder_t.load_state_dict(torch.load('./model_ckpt/encoder_t_3.pt', map_location='cpu'))
        encoder_t.to(args.device)
        encoder_ch.load_state_dict(torch.load('./model_ckpt/encoder_ch_3.pt', map_location='cpu'))
        encoder_ch.to(args.device)
        linear.load_state_dict(torch.load('./model_ckpt/linear_3.pt', map_location='cpu'))
        linear.to(args.device)
        modules = [encoder_t, encoder_ch]
        test_loss, test_metr = evaluate3(args, test_data_list, modules, patients, linear)
        print("On test: loss = %.4f " % test_loss, test_metr.get_metrics(one_line=True))

    else:
        raise NotImplementedError()


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    print('This progress began at: ' + time.asctime(time.localtime(time.time())))
    torch.set_default_tensor_type(torch.FloatTensor)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--unwrap_ddp", type=str2bool, default=True)
    parser.add_argument("--load_pretrained", type=str2bool, default=True)
    parser.add_argument("--load_epo_idx", type=int, default=29)
    parser.add_argument("--freeze_encoder", type=bool, default=False)
    parser.add_argument("--metric", type=str, default='f1')
    parser.add_argument("--main_task", type=int, default=1)  # main task 1, 2, 3, 4

    parser.add_argument("--src_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--main2_mode", type=str, default='finetune')  # 'finetune' or 'probe'
    parser.add_argument("--main2_task", type=str, default='ph_freq')   # 'long' or 'short' or 'ph_freq'
    parser.add_argument("--main2_long_term_pred_len", type=int, default=20)
    parser.add_argument("--main2_short_term_pred_len", type=int, default=2)
    parser.add_argument("--main2_long_term_dr", type=int, default=1)
    parser.add_argument("--main2_short_term_dr", type=int, default=1)
    parser.add_argument("--main2_ph_freq_pred_len", type=int, default=5)

    parser.add_argument("--main3_mask_rate", type=float, default=0.4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ft_lr", type=float, default=1e-7)

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--warm_up_epo", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--infer_batch_size", type=int, default=4)

    args = parser.parse_args()

    if args.main_task == 1:
        args.task_cate = 'seizure_detection'
    elif args.main_task == 2:
        args.task_cate = 'forecast'
    elif args.main_task == 3:
        args.task_cate = 'imputation'

    patients = ...  # the list of patients

    module = model_prepare()
    train(module)
    test(patients, module)


print('\n' * 3)
print('─' * 50)
print('This progress finished at: ' + time.asctime(time.localtime(time.time())))

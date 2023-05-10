import datetime
import os
import argparse
from torch import nn

import time
import numpy as np
import torch
import random
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pre_model import ChannelEncoder, TimeEncoder
from pre_train import run_pre_train
from pre_utils import master_print


def load_model(args):
    time_enc =  TimeEncoder(in_dim          =args.seg_len,
                            d_model         =args.d_model,
                            dim_feedforward =args.dim_feedforward,
                            seq_len         =args.seq_len,
                            n_layer         =args.time_ar_layer,
                            nhead           =args.time_ar_head,
                            band_num        =args.band_num,
                            project_mode    =args.input_emb_mode,
                            learnable_mask  =args.learnable_mask)
    ch_enc = ChannelEncoder(out_dim         =args.seg_len,
                            d_model         =args.d_model,
                            dim_feedforward =args.dim_feedforward,
                            n_layer         =args.ch_ar_layer,
                            nhead           =args.ch_ar_head)

    time_enc = time_enc.to(args.local_rank)
    ch_enc = ch_enc.to(args.local_rank)

    if args.dist_data_parallel:
        # convert BatchNorm to SyncBatchNorm
        time_enc = nn.SyncBatchNorm.convert_sync_batchnorm(time_enc)
        ch_enc = nn.SyncBatchNorm.convert_sync_batchnorm(ch_enc)

        time_enc = DDP(time_enc, device_ids=[args.local_rank], output_device=args.local_rank, )
        ch_enc = DDP(ch_enc, device_ids=[args.local_rank], output_device=args.local_rank, )

    time_param = sum(p.numel() for p in time_enc.parameters())
    ch_param = sum(p.numel() for p in ch_enc.parameters())

    master_print(f'Model Param \n'
                f'feature dimension: {args.d_model} \n'
                f'time encoder has {args.time_ar_layer} layers with {time_param / 1e6}M parameters  \n'
                f'channel encoder has {args.ch_ar_layer} layers with {ch_param / 1e6}M parameters\n'
                f'total parameter number: {(time_param + ch_param) / 1e6}M')

    if args.start_epo_id >= 0:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        time_enc.load_state_dict(torch.load(f'./encoder_ckpt/time_encoder_{args.start_epo_idx}.pt', map_location=map_location))
        ch_enc.load_state_dict(torch.load(f'./encoder_ckpt/channel_encoder_{args.start_epo_idx}.pt', map_location=map_location))

    return time_enc, ch_enc


def get_optim(optim_name):
    if optim_name == 'adam':
        optimizer = optim.Adam(
            [{'params': list(time_enc.parameters()), 'lr': args.time_lr},
             {'params': list(ch_enc.parameters()), 'lr': args.ch_lr}],
            betas=(0.9, 0.99), eps=1e-8,
        )
    elif optim_name == 'adamw':
        optimizer = optim.AdamW(
            [{'params': list(time_enc.parameters()), 'lr': args.time_lr},
             {'params': list(ch_enc.parameters()), 'lr': args.ch_lr}],
            betas=(0.9, 0.99), eps=1e-6,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer, scheduler_name):
    if scheduler_name == 'mul_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif scheduler_name == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=3e-6, max_lr=1e-5, step_size_up=8000, cycle_momentum=False)
    else:
        raise NotImplementedError

    return scheduler


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=18000))


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

    num_threads = '32'
    torch.set_num_threads(int(num_threads))
    os.environ['OMP_NUM_THREADS'] = num_threads
    os.environ['OPENBLAS_NUM_THREADS'] = num_threads
    os.environ['MKL_NUM_THREADS'] = num_threads
    os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = num_threads

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dev_ids', type=str, default="0,1,2,3,4,5,6,7")

    parser.add_argument("--seg_len", type=int, default=1500)
    parser.add_argument("--band_num", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=1280)
    parser.add_argument("--dim_feedforward", type=int, default=3072)

    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--time_ar_layer", type=int, default=12)
    parser.add_argument("--time_ar_head", type=int, default=16)
    parser.add_argument("--input_emb_mode", type=str, default='linear')
    parser.add_argument("--ch_ar_layer", type=int, default=5)
    parser.add_argument("--ch_ar_head", type=int, default=16)
    parser.add_argument("--learnable_mask", type=str2bool, default=False)

    parser.add_argument("--dist_data_parallel", type=str2bool, default=False)
    parser.add_argument("--amp", type=str2bool, default=True,
                        help='whether to use automatic mixed precision')
    parser.add_argument("--start_epo_idx", type=int, default=-1,
                        help='load the model from last train, -1 means a new model')

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--mask_ratio", type=float, default=0.4)

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--scheduler", type=str, default='cyclic')
    parser.add_argument("--accu_step", type=int, default=4)
    parser.add_argument("--time_lr", type=float, default=3e-6)
    parser.add_argument("--ch_lr",   type=float, default=3e-6)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.dev_ids

    if args.dist_data_parallel:
        dist.init_process_group(backend="nccl")

    # get model, optimizer and scheduler
    time_enc, ch_enc = load_model(args)
    optimizer = get_optim(args.optimizer)
    scheduler = get_scheduler(optimizer, scheduler_name=args.scheduler)

    # pre-train
    run_pre_train(args, time_enc, ch_enc, optimizer, scheduler)

    master_print('\n' * 3)
    master_print('─' * 50)
    master_print('This progress finished at: ' + time.asctime(time.localtime(time.time())))

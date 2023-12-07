import numpy as np
import pandas as pd
import os
import scipy.io as sio

mayo_pat_ids = [0, 18, 21, 1, 9, 19, 2, 5, 16, 3, 4, 23, 6, 7, 8, 14, 17, 20]
fnusa_pat_ids = [1, 5, 2, 9, 3, 4, 12, 6, 7, 8, 10, 11, 13]
data_path = '/path/to/data'


def get_df(data_path, pat_ids):
    seg_path = os.path.join(data_path, 'segments.csv')
    df = pd.read_csv(seg_path)

    rm0_idx = df['category_id'] != 0
    idx = np.any([df['patient_id'] == pat_id for pat_id in pat_ids], axis=0)
    idx = np.logical_and(idx, rm0_idx)

    df = df[idx].reset_index(drop=True)

    return df


def interp(data_path, df, up_win_size):
    length = len(df)
    for seg_idx in range(length):
        seg_meta = df.iloc[seg_idx]
        seg_id = seg_meta['segment_id']

        # y = seg_meta['category_id']
        x = sio.loadmat(os.path.join(data_path, f'{seg_id}.mat'))['data'].reshape(-1)
        l = x.size
        x_axis = np.linspace(1, l, l)
        up_x_axis = np.linspace(1, l, up_win_size)
        up_x = np.interp(up_x_axis, x_axis, x)
        np.save(os.path.join(data_path, f'{seg_id}_{up_win_size}.npy'), up_x)


def agg_data(data_path, df, up_win_size):
    x = []
    y = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        seg_meta = df.iloc[i]
        seg_id = seg_meta['segment_id']

        y[i] = seg_meta['category_id'] - 1
        x.append(np.expand_dims(np.load(os.path.join(data_path, f'{seg_id}_{up_win_size}.npy')), axis=0))

    x = np.float32(np.concatenate(x, axis=0))
    np.save(os.path.join(data_path, f'x_{up_win_size}.npy'), x)
    np.save(os.path.join(data_path, f'y_{up_win_size}.npy'), y)


if __name__ == '__main__':
    mayo_df = get_df(os.path.join(data_path, 'MAYO'), mayo_pat_ids)
    fnusa_df = get_df(os.path.join(data_path, 'FNUSA'), fnusa_pat_ids)
    interp(os.path.join(data_path, 'MAYO/DATASET_MAYO'), mayo_df, 15*1500)
    interp(os.path.join(data_path, 'FNUSA/DATASET_FNUSA'), fnusa_df, 15 * 1500)
    agg_data(os.path.join(data_path, 'MAYO/DATASET_MAYO'), mayo_df, 15*1500)
    agg_data(os.path.join(data_path, 'FNUSA/DATASET_FNUSA'), fnusa_df, 15*1500)

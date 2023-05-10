import torch
import numpy as np
from scipy.signal import hilbert


def get_seg_property(args, fut_seg):
    w = torch.fft.fft(fut_seg.cpu(), dim=-1)
    f = torch.fft.fftfreq(fut_seg.cpu().shape[-1])

    freq = [ f[ int(torch.argmax(w[b,:].abs())) ]
             for b in range(fut_seg.shape[0])
            ]

    analytic_signal = hilbert(fut_seg.cpu())
    instantaneous_phase = np.angle(analytic_signal)

    phase = torch.tensor(instantaneous_phase)
    freq  = torch.tensor(freq).reshape(-1, 1)
    return phase.to(args.device), \
           freq .to(args.device)

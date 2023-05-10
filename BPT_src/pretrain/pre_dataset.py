from torch.utils.data import Dataset


class PreDataset(Dataset):
    def __init__(self, data, power):
        self.power = power
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        res = (self.data[idx],)
        if self.power is not None:
            res += (self.power[idx], )
        return res

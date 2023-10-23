from torch.utils.data import Dataset


class BoardDataset(Dataset):
    def __init__(self, data, power, y_label, need_y=True):
        self.need_y = need_y

        _, self.board_num, _, _ = data.shape

        self.data = data
        self.power = power
        if need_y:
            self.y_label = y_label

    def __len__(self):
        return self.board_num

    def __getitem__(self, board_idx):
        item = (self.data[:, board_idx, :, :],
                self.power[:, board_idx, :, :])

        if self.need_y:
            item += (self.y_label[:, board_idx, :],)

        return item


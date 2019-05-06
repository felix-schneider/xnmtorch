from torch.utils.data import Dataset

from xnmtorch.persistence import Serializable


class ParallelDataset(Dataset, Serializable):
    def __init__(self, src_data: Dataset, tgt_data: Dataset):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __getitem__(self, index):
        return {"src": self.src_data[index], "tgt": self.tgt_data[index]}

    def __len__(self):
        return len(self.src_data)



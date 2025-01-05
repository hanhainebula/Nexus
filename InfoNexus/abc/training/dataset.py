from torch.utils.data import Dataset

from .arguments import AbsDataArguments


class AbsDataset(Dataset):
    def __init__(self, args: AbsDataArguments):
        self.args = args

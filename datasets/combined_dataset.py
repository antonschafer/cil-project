from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets, is_test):
        """
        Create a combined dataset from a list of datasets.

        Args:
            datasets: A list of datasets. Each dataset consists of x, y (input, label)
                the labels are assumed to be the same across datasets and datasets
                consist of the same underlying data
            is_test: A boolean indicating whether this is a test dataset (i.e. no labels available)
        """
        super().__init__()
        self.datasets = datasets
        self.is_test = is_test

        if not all(len(d) == len(datasets[0]) for d in datasets):
            raise ValueError("Datasets must have the same length")

    def __getitem__(self, index):
        x = [d[index][0] for d in self.datasets]
        if self.is_test:
            # no labels available
            return x
        else:
            y = self.datasets[0][index][1]
            return x, y

    def __len__(self):
        return len(self.data)

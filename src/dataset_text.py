import torch
import numpy as np
from proc.dataset_config import data_args, get_dataset
from torch.utils.data import Dataset, DataLoader, random_split


class TextDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.inputs = None
        self.masks = None
        self.labels = None
        self.train_size = 0
        self.valid_size = 0
        self.process()
        super().__init__()

    def process(self):

        print('Load input...')
        all_inputs, all_masks, all_labels = self.get_input(self.dataset_name)
        self.inputs = torch.tensor(all_inputs, dtype=torch.long)
        self.masks = torch.tensor(all_masks, dtype=torch.bool)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

    @staticmethod
    def get_input(dataset, mode='all'):
        this_inputs = np.load('./data/temp/{}.inputs.{}.npy'.format(dataset, mode))
        this_masks = np.load('./data/temp/{}.masks.{}.npy'.format(dataset, mode))
        this_labels = np.load('./data/temp/{}.labels.{}.npy'.format(dataset, mode))
        return this_inputs, this_masks, this_labels

    def __getitem__(self, index):
        return self.inputs[index], self.masks[index], self.labels[index]

    def __len__(self):
        return self.inputs.shape[0]


def get_data_loader(dataset, batch_size):
    # load data
    tx_dataset = TextDataset(dataset)

    train_size = data_args[dataset]['train_size']
    valid_size = data_args[dataset]['valid_size']
    test_size = data_args[dataset]['test_size']

    # split
    train_data, valid_data, test_data = random_split(
        tx_dataset, [train_size-valid_size, valid_size, test_size])

    # return batch
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    Dataset = get_dataset()
    BatchSize = 16
    TrainLoader, ValidLoader, TestLoader = get_data_loader(Dataset, BatchSize)
    print(len(TrainLoader), len(ValidLoader), len(TestLoader))
    for batch_idx, (batch_input, batch_mask, batch_label) in enumerate(ValidLoader):
        print('*'*80)
        print(batch_idx)
        print(batch_input)
        print(batch_mask)
        print(batch_label)
        print(len(batch_label))

from torch.utils import data
import os
import pickle


class TextDataset(data.Dataset):

    def __init__(self, path):
        self.file_name = os.listdir(path)

    def __getitem__(self, index):
        # Rewrite the following code to get you own data.
        train_set, labels = pickle.load(open(self.file_name[index], 'rb'))

        # Should return the training matrix, labels, and the number of the sentences in one document.
        return train_set, labels, len(train_set)

    def __len__(self):
        return len(self.train_set)



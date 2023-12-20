from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class QQPDataSet(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = { key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_and_prepare_data(tokenizer, split='train', limit=None):
    dataset = load_dataset('glue', 'qqp', split=split)

    if limit:
        dataset = dataset.select(range(limit))

    labels = dataset['label']
    encodings = tokenizer(dataset['question1'], dataset['question2'], padding=True, truncation=True, return_tensors="pt")
    return QQPDataSet(encodings, labels)

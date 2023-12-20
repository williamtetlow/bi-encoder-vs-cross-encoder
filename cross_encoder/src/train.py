from dataset import load_and_prepare_data, QQPDataSet
from model import CrossEncoder
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
import os

def main():
    # Parameters 
    model_name = 'bert-base-uncased'
    num_labels = 2
    batch_size = 32
    epochs = 3
    learning_rate = 0.0001
    limit_dataset_size = 100  # [OPTIONAL] Limit the dataset for initial tests
    model_output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models', 'qpp/')

    # Load and prepare the dataset
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = load_and_prepare_data(tokenizer, split='train', limit=limit_dataset_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CrossEncoder(model_name, num_labels)
    model.model.to(model.model.device)

    model.train(train_dataloader, epochs)

    model.save_model(model_output_path)


if __name__ == "__main__":
    main()


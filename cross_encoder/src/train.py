from dataset import load_and_prepare_data, QQPDataSet
from model import CrossEncoder
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
import os
import argparse

def main(limit_dataset_size):
    # Parameters 
    model_name = 'bert-base-uncased'
    num_labels = 2
    batch_size = 32
    epochs = 3
    learning_rate = 0.0001
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
    parser = argparse.ArgumentParser(description='Script to train QQP model')
    parser.add_argument('--limit_dataset_size', type=int, default=None,
                        help='Limit the size of the dataset (optional)')
    args = parser.parse_args()
    
    main(args.limit_dataset_size)


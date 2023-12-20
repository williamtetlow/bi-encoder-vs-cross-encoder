# https://chat.openai.com/c/430aab69-f0c4-40d6-a326-c4041f05562d
from transformers import BertTokenizer, BertForSequenceClassification
import torch

from transformers import BertTokenizer, BertForSequenceClassification
import torch

class CrossEncoder:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def train(self, train_dataloader, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for batch in train_dataloader:
                inputs = {k: v.to(self.model.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.model.device)

                self.optimizer.zero_grad()

                outputs = self.model(**inputs)

                loss = outputs.loss if outputs.loss is not None else torch.nn.functional.cross_entropy(outputs.logits, labels)
                loss.backward()

                self.optimizer.step()
            print(f'Epoch {epoch+1}/{epochs} completed.')

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, save_path):
        self.model = BertForSequenceClassification.from_pretrained(save_path)
        self.tokenizer = BertTokenizer.from_pretrained(save_path)

    def predict(self, text1, text2):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text1, text2, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            return logits.argmax().item()


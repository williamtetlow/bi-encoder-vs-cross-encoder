from transformers import BertTokenizer, BertModel
import torch

class BiEncoder:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name) 
        self.model = BertModel.from_pretrained(model_name)

    def predict(self, input1, input2):
        t1 = tokenizer(input1, return_tensors="pt")
        t2 = tokenizer(input2, return_tensors="pt")

        with torch.no_grad():
            o1 = model(**t1)
            o2 = model(**t2)

        # we remove batch dimension with .squeeze
        emb1 = o1.pooler_output.squeeze()
        emb2 = o2.pooler_output.squeeze()

        cos_sim = torch.cosine_similarity(emb1, emb2, dim=0).item() 

        return cos_sim


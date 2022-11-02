import numpy as np
import gensim
import torch
import transformers as ppb
# Kibado: https://github.com/pdrm83/sent2vec/blob/master/sent2vec/vectorizer.py


class Vectorizer:
    def __init__(self):
        self.vectors = []

    def bert(self, sentences, pretrained_weights='neuralmind/bert-base-portuguese-cased'):
        model_class = ppb.AutoModel
        tokenizer_class = ppb.AutoTokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        tokenized = list(map(lambda x: tokenizer.encode(x, add_special_tokens=True), sentences))

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor)

        with torch.no_grad():
            last_hidden_states = model(input_ids)

        vectors = last_hidden_states[0][:, 0, :].numpy()
        self.vectors = vectors

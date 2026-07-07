import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config

# 1. Hamara purana Tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def fit_on_texts(self, texts):
        for text in texts:
            for word in str(text).lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text, max_len):
        tokens = [1] 
        for word in str(text).lower().split():
            tokens.append(self.word2idx.get(word, 3)) 
        tokens.append(2) 
        
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        return torch.tensor(tokens[:max_len])

    def decode(self, ids):
        words = []
        for i in ids:
            if i.item() in [0, 1, 2]:
                continue
            words.append(self.idx2word.get(i.item(), "<UNK>"))
        return " ".join(words)

# 2. Text Dataset Class
class TextDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_encoded = self.tokenizer.encode(self.src_texts[idx], self.max_len)
        tgt_encoded = self.tokenizer.encode(self.tgt_texts[idx], self.max_len)
        return src_encoded, tgt_encoded

def get_train_loader(csv_path="data.csv"):
    print("Wait sometime , CSV file is loading...")
    df = pd.read_csv(csv_path)
    
    df = df.dropna()
    
    src_sentences = df['english'].astype(str).tolist()
    tgt_sentences = df['hindi'].astype(str).tolist()
    
    print(f"Total {len(src_sentences)} sentences founded.")
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(src_sentences + tgt_sentences)
    
    config.SRC_VOCAB_SIZE = tokenizer.vocab_size
    config.TGT_VOCAB_SIZE = tokenizer.vocab_size
    print(f"New Vocab Size: {tokenizer.vocab_size} words")
    
    dataset = TextDataset(src_sentences, tgt_sentences, tokenizer, config.MAX_SEQ_LENGTH)
    
    return DataLoader(dataset, batch_size=32, shuffle=True), tokenizer
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

class WordDataset(Dataset):

    def __init__(self, data, block_size):
        self.tokenized_words = tokenizer(data)['input_ids']
        print('tokenized words shape',len(self.tokenized_words))
        unique = sorted(list(set(self.tokenized_words)))
        data_size, vocab_size = len(self.tokenized_words), len(unique)
        print('data has %d words, %d unique.' % (data_size, vocab_size))
        
        self.block_size = block_size
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.tokenized_words) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.tokenized_words[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y

class TensorWordDataset(Dataset):

    def __init__(self, path, block_size):
        self.tokenized_words = torch.load(path)
        print('tokenized words shape',len(self.tokenized_words))
        # unique = sorted(list(set(self.tokenized_words)))
        # data_size, vocab_size = len(self.tokenized_words), len(unique)
        # print('data has %d words, %d unique.' % (data_size, vocab_size))
        
        self.block_size = block_size
        # self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.tokenized_words) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.tokenized_words[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]

        return x, y

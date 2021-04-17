from typing import Generator

from torch.utils.data import Dataset, DataLoader
import torch

import attention.example.prepdata as prep

MAX_TOKENS = 50


class SentenceDataset(Dataset):

    def __init__(self):
        self.sentences = prep.compute_yelp()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        sentence = self.sentences[idx]
        return sentence


def compute_batch_generator(
    batch_size: int = 32,
    num_epochs: int = 1_000_000
) -> Generator:
    dataset = SentenceDataset()
    tokenizer = prep.compute_tokenizer()
    tokenizer.enable_padding()
    tokenizer.enable_truncation(MAX_TOKENS)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(num_epochs):
        for batch in loader:
            encoded = [s.ids for s in tokenizer.encode_batch(batch)]
            yield torch.LongTensor(encoded)

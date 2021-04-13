import json
import os
from os.path import join
from typing import Generator, List

from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

MAX_WORDS = 30
MAX_TOKENS = 50
DATA_PATH = "/home/hahdawg/projects/attention/attention/example/data"
OUTPUT_PATH_RAW = join(DATA_PATH, "sentences.txt")
BERT_PATH = join(DATA_PATH, "bert-base-uncased-vocab.txt")


def get_bert() -> None:
    cmd = "wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    cwd = os.getcwd()
    os.chdir(DATA_PATH)
    os.system(cmd)
    os.chdir(cwd)


def load_tokenizer() -> BertWordPieceTokenizer:
    return BertWordPieceTokenizer(BERT_PATH, lowercase=True)


def write_sentences_to_disk() -> None:
    src_path = join(DATA_PATH, "yelp_academic_dataset_review.json")
    with open(src_path, "r") as src, open(OUTPUT_PATH_RAW, "w") as dest:
        for line in src:
            review = json.loads(line)["text"].lower().strip()
            sentences = review.split(".")
            for sentence in sentences:
                sentence = sentence.strip().replace("\n", "")
                if len(sentence) and len(sentence.split(" ")):
                    dest.write(sentence + "." + "\n")


def load_sentences() -> List[str]:
    sentences = []
    with open(OUTPUT_PATH_RAW, "r") as f:
        for line in f:
            split = line.split(" ")
            num_words = len(split)
            if num_words < MAX_WORDS:
                sentences.append(line)
    return sentences


class SentenceDataset(Dataset):

    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        sentence = self.sentences[idx]
        return sentence


def batch_generator(
    batch_size: int,
    sentences: List[str],
    num_epochs: int
) -> Generator:
    sentences = sentences or load_sentences()
    dataset = SentenceDataset(sentences)
    tokenizer = load_tokenizer()
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


def load_batch_generator(batch_size: int, num_epochs: int = 1_000_000) -> Generator:
    sentences = load_sentences()
    return batch_generator(
        batch_size=batch_size,
        sentences=sentences,
        num_epochs=num_epochs
    )

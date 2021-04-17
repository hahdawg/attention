import json
import os
from os.path import join
import pickle
import subprocess
from typing import List

from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm

# ############################################################################################
#  NOTE: Download JSON data from https://www.yelp.com/dataset/download and put it in DATA_DIR.
# ############################################################################################
MAX_WORDS = 30
DATA_DIR = "/home/hahdawg/projects/attention/attention/example/data"


def get_bert_bpe() -> None:
    """
    Get bert BPE.
    """
    cmd = "wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    cwd = os.getcwd()
    os.chdir(DATA_DIR)
    os.system(cmd)
    os.chdir(cwd)


def get_yelp():
    """
    Unzips yelp. Can't download directly without signing in.
    """
    zip_path = join(DATA_DIR, "yelp_dataset.tgz")
    if not os.path.isfile(zip_path):
        yelp_url = "https://www.yelp.com/dataset/download"
        msg = f"Must download yelp dataset from {yelp_url} and put it in {DATA_DIR}"
        raise ValueError(msg)
    os.system(f"tar -xzvf {zip_path}")


def compute_tokenizer() -> BertWordPieceTokenizer:
    bert_path = join(DATA_DIR, "bert-base-uncased-vocab.txt")
    return BertWordPieceTokenizer(bert_path, lowercase=True)


def compute_yelp(use_cached=True) -> List[str]:
    """
    Get sentences from yelp review dataset. Takes a few minutes, so
    we store the output on the intitial run and read it from
    disk after that.
    """
    output_path = join(DATA_DIR, "sentences.p")
    load_from_disk = use_cached and os.path.isfile(output_path)
    if load_from_disk:
        with open(output_path, "rb") as f:
            return pickle.load(f)

    output = []
    src_path = join(DATA_DIR, "yelp_academic_dataset_review.json")
    num_row = int(subprocess.check_output(f"wc -l {src_path}", shell=True).split()[0])
    with open(src_path, "r") as src:
        for line in tqdm(src, total=num_row):
            review = json.loads(line)["text"].lower().strip()
            review = review.replace("!", ".")
            sentences = review.split(".")
            for sentence in sentences:
                sentence = sentence.strip().replace("\n", "")
                num_words = len(sentence.split(" "))
                if 0 < num_words <= MAX_WORDS:
                    output.append(sentence)

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    return output


def main() -> None:
    """
    Initialize all data.
    """
    get_bert_bpe()
    get_yelp()
    compute_yelp(use_cached=False)


if __name__ == "__main__":
    main()

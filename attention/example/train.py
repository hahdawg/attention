from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import attention.example.data as aed
import attention.model as am


def write_sentences(
    model: am.LanguageModel,
    tokenizer: aed.BertWordPieceTokenizer,
    batch_size: int,
    max_tokens: int,
    num_tokens_to_consider: int,
    device: str
) -> List[str]:
    """
    Given a model, create batch_size sentences with length max_tokens.
    """
    start_token = 101
    tokenizer = aed.load_tokenizer()
    softmax = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        x = start_token*torch.ones(batch_size).reshape(-1, 1)
        for _ in range(max_tokens):
            x = x.long().to(device)
            logits = model(x)[:, -1, :]
            probs = softmax(logits).cpu().numpy()
            cutoff = np.sort(probs, axis=1)[:, -num_tokens_to_consider].reshape(-1, 1)
            probs[probs < cutoff] = 0
            probs /= probs.sum(axis=1).reshape(-1, 1)
            preds = [
                np.random.choice(np.arange(probs.shape[1]), p=probs[i])
                for i in range(batch_size)
            ]
            preds = torch.LongTensor(preds).to(device).reshape(-1, 1)
            x = torch.cat((x, preds), dim=1)
        decoded = tokenizer.decode_batch(x.cpu().numpy())

    sentences = [
        s[:s.find(".") + 1]
        if "." in s
        else s
        for s in decoded
    ]
    return sentences


def main(
    batch_size_tr: int = 32,
    lr: float = 1e-3,
    logging_interval: int = 1000,
    max_steps: int = 100_000_000,
    embedding_size: int = 256,
    num_heads: int = 3,
    num_layers: int = 3,
    dim_feedforward: int = 1024,
    dropout_input: int = 0.1,
    dropout_hidden: int = 0.1,
    batch_size_val: int = 5,
    max_tokens_val: int = 30,
    num_tokens_to_consider: int = 15,
):
    """
    Train a model. At each logging interval, generate some sample sentences.
    """
    batch_generator = aed.load_batch_generator(batch_size=batch_size_tr, num_epochs=max_steps)
    tokenizer = aed.load_tokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = tokenizer.get_vocab_size()
    print(f"Number of tokens: {vocab_size}")
    model = am.LanguageModel(
        embedding_size=embedding_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout_input=dropout_input,
        dropout_hidden=dropout_hidden
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    pad_token = tokenizer.token_to_id("[PAD]")

    loss_fcn = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(lr=lr, params=model.parameters())

    running_loss_tr = deque(maxlen=logging_interval)
    running_acc_tr = deque(maxlen=logging_interval)
    baseline_loss = -np.log(1/vocab_size)
    print(f"Starting training. Baseline loss from random guesser: {baseline_loss: 0.4f}")
    for step, batch in enumerate(batch_generator):
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:].reshape(-1,)

        model.train()
        optimizer.zero_grad()
        pred = model(x).reshape(-1, vocab_size)
        pred_label = pred.argmax(axis=-1)
        loss = loss_fcn(pred.reshape(-1, vocab_size), y)
        acc = (pred_label.reshape(-1,) == y).float().mean()
        not_pad_idx = y != pad_token
        loss = loss[not_pad_idx].mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss_tr.append(loss.detach().item())
            running_acc_tr.append(acc.detach().item())

        if step % logging_interval == 0:
            model.eval()
            with torch.no_grad():
                loss_tr_mean = np.mean(running_loss_tr)
                acc_tr_mean = np.mean(running_acc_tr)
                print(100*"-")
                print(f"[{step}]  loss-tr: {loss_tr_mean: 0.5f}  acc-tr: {acc_tr_mean: 0.5f}")
                print(100*"-")
                sample_sentences = write_sentences(
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=batch_size_val,
                    max_tokens=max_tokens_val,
                    num_tokens_to_consider=num_tokens_to_consider,
                    device=device
                )
                for i, sentence in enumerate(sample_sentences):
                    print(f"  [sample {i}]: {sentence}")

        if step >= max_steps:
            break

    return model


if __name__ == "__main__":
    main()

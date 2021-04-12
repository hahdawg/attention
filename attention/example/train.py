from collections import deque

import numpy as np
from numpy import mean
import torch
import torch.nn as nn
import torch.optim as optim

import attention.example.data as aed
import attention.model as am


def write_sentences(model, batch_size, maxlen, device):
    start_token = 101
    tokenizer = aed.load_tokenizer()
    softmax = nn.Softmax(dim=1)
    x = start_token*torch.ones(batch_size).reshape(-1, 1)
    model.eval()
    N = 10
    with torch.no_grad():
        for _ in range(maxlen):
            x = x.long().to(device)
            logits = model(x)[:, -1, :]
            probs = softmax(logits).cpu().numpy()
            cutoff = np.sort(probs, axis=1)[:, -N].reshape(-1, 1)
            probs[probs < cutoff] = 0
            probs /= probs.sum(axis=1).reshape(-1, 1)
            preds = [
                np.random.choice(np.arange(probs.shape[1]), p=probs[i])
                for i in range(batch_size)
            ]
            preds = torch.LongTensor(preds).to(device).reshape(-1, 1)
            x = torch.cat((x, preds), dim=1)

    x = tokenizer.decode_batch(x.cpu().numpy())
    return x


def main(
    bg,
    lr=1e-3,
    logging_interval=500,
    max_steps=100_000_000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = aed.load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Number of tokens: {vocab_size}")
    model = am.LanguageModel(
        embedding_size=128,
        vocab_size=vocab_size,
        num_heads=4,
        num_layers=3,
        dim_feedforward=1024,
        dropout_input=0.1,
        dropout_hidden=0.1
    ).to(device)
    pad_token = tokenizer.token_to_id("[PAD]")

    loss_fcn = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(lr=lr, params=model.parameters())

    running_loss_tr = deque(maxlen=logging_interval)
    running_acc_tr = deque(maxlen=logging_interval)
    for step, batch in enumerate(bg):
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

        running_loss_tr.append(loss.item())
        running_acc_tr.append(acc.item())
        model.eval()
        if step % logging_interval == 0:
            with torch.no_grad():
                loss_tr_mean = mean(running_loss_tr)
                acc_tr_mean = mean(running_acc_tr)
                print(f"[{step}]  loss-tr: {loss_tr_mean}  acc-tr: {acc_tr_mean}")
                sample_sentences = write_sentences(model, batch_size=5, maxlen=40, device=device)
                for i, sentence in enumerate(sample_sentences):
                    print(f"\t[sample {i}]: {sentence}")

        if step >= max_steps:
            break

    return model

from collections import deque

from numpy import mean
import torch
import torch.nn as nn
import torch.optim as optim

import attention.model as am


def main(
    bg,
    tokenizer,
    lr=1e-3,
    logging_interval=500,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = tokenizer.get_vocab_size()
    model = am.LanguageModel(
        embedding_size=128,
        vocab_size=vocab_size,
        num_heads=4,
        num_layers=3,
        dim_feedforward=1024
    ).to(device)
    pad_token = tokenizer.token_to_id("[PAD]")

    # loss_fcn = nn.NLLLoss(reduction="none")
    loss_fcn = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(lr=lr, params=model.parameters())

    running_loss_tr = deque(maxlen=logging_interval)
    for step, batch in enumerate(bg):
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fcn(pred.reshape(-1, vocab_size), y.reshape(-1,))
        # acc = (pred.argmax(axis=2) == y).float().mean()
        not_pad_idx = y.reshape(-1,) != pad_token
        loss = loss[not_pad_idx].mean()
        loss.backward()
        optimizer.step()

        if False:
            print(100*"=")
            for p in model.parameters():
                print(p.grad)
            print(100*"=")
            if step > 2:
                break

        running_loss_tr.append(loss.item())

        model.eval()
        if step % logging_interval == 0:
            with torch.no_grad():
                loss_tr_mean = mean(running_loss_tr)
                print(f"{step}:  loss-tr: {loss_tr_mean}")

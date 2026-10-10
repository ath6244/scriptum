"""Seeding, batching, and the train / early-stopping loop shared by both neural models."""

import copy
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 64
MIN_WIDTH = 5  # TextCNN's widest filter is 5 tokens, so every batch is padded to at least 5


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ask cuDNN for deterministic algorithms so the same seed gives the same run.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate(batch):
    """batch is a list of (token_ids, label). Pad the ids with 0 to the longest row in the batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    ids = torch.zeros(len(sequences), max(lengths.max().item(), MIN_WIDTH), dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq)
    return ids, lengths, torch.tensor(labels)


def make_loader(sequences, labels, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)  # makes the shuffle order depend only on the seed
    return DataLoader(list(zip(sequences, labels)), batch_size=BATCH_SIZE, shuffle=shuffle,
                      collate_fn=collate, generator=generator)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()  # turns dropout off
    predictions, truth = [], []
    for ids, lengths, labels in loader:
        logits = model(ids.to(device), lengths.to(device))
        predictions.append(logits.argmax(dim=1).cpu())
        truth.append(labels)
    return torch.cat(predictions).numpy(), torch.cat(truth).numpy()


def train_model(model, train_loader, val_loader, device, lr=1e-3, weight_decay=1e-4,
                max_epochs=20, patience=3, clip=1.0):
    """Train with early stopping on val macro-F1. Returns (best_epoch, history); model ends at its best epoch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_f1, best_epoch, best_state, epochs_without_gain = -1.0, 0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for ids, lengths, labels in train_loader:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # cap the gradient size for stable LSTM training
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n_seen += len(labels)

        val_pred, val_true = predict(model, val_loader, device)
        val_f1 = f1_score(val_true, val_pred, average="macro")
        val_acc = accuracy_score(val_true, val_pred)
        history.append({"epoch": epoch, "train_loss": total_loss / n_seen, "val_acc": val_acc, "val_macro_f1": val_f1})
        print(f"    epoch {epoch:2d}  train loss {total_loss / n_seen:.4f}  val acc {val_acc:.3f}  val F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, epochs_without_gain = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())  # snapshot of the weights at the best epoch
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                break

    model.load_state_dict(best_state)  # restore the best checkpoint before testing
    return best_epoch, history

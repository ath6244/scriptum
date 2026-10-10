"""The two required neural classifiers. Both take (ids, lengths) and return one score per class.

ids:     LongTensor (batch, seq_len), padded with 0 (<pad>) on the right
lengths: LongTensor (batch,), the real number of tokens in each row
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCNN(nn.Module):
    """Kim (2014): convolutions of several widths over word embeddings, max-pooled over time."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, filter_sizes=(3, 4, 5), n_filters=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)  # <pad> stays a zero vector
        # One Conv1d per filter size; a filter of size 3 looks at 3 consecutive tokens (a "3-gram").
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, size) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes), n_classes)

    def forward(self, ids, lengths):
        x = self.embedding(ids)   # (batch, seq_len, emb_dim)
        x = x.transpose(1, 2)     # (batch, emb_dim, seq_len): Conv1d wants channels second
        # For each filter: ReLU, then keep its strongest response anywhere in the text.
        pooled = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]  # each (batch, n_filters)
        features = torch.cat(pooled, dim=1)  # (batch, 300)
        return self.fc(self.dropout(features))


class BiLSTM(nn.Module):
    """2-layer bidirectional LSTM; the per-token outputs are averaged over the real (non-pad) tokens."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # dropout here is applied between the two LSTM layers
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=n_layers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)  # forward + backward states

    def forward(self, ids, lengths):
        x = self.embedding(ids)  # (batch, seq_len, emb_dim)
        # Packing tells the LSTM each row's real length, so the backward direction starts at the
        # last real token instead of reading the padding first.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))  # (batch, seq_len, 256)

        # Mean over real tokens only: mask is 1 for real tokens, 0 for padding.
        positions = torch.arange(ids.size(1), device=ids.device)
        mask = (positions[None, :] < lengths[:, None]).unsqueeze(2).float()  # (batch, seq_len, 1)
        mean = (out * mask).sum(dim=1) / lengths[:, None].float()            # (batch, 256)
        return self.fc(self.dropout(mean))


def build_model(name, vocab_size, n_classes):
    if name == "textcnn":
        return TextCNN(vocab_size, n_classes)
    if name == "bilstm":
        return BiLSTM(vocab_size, n_classes)
    raise ValueError(name)

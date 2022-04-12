import torch
import numpy as np
from fastai.data.load import _FakeLoader, _loaders

def collate_fn(data):
    images, reports_ids, reports_masks, seq_lengths, onehot_lbls = zip(*data)
    if len(images)>=2: 
        images = torch.stack(images, 0)
        onehot_lbls = torch.stack(onehot_lbls, 0)
    else: 
        images = images[0]
        onehot_lbls = onehot_lbls[0]
        images = images.unsqueeze(0)
        onehot_lbls = onehot_lbls.unsqueeze(0)
    images = images.permute(0, 3, 1, 2)
    images = images.float() 
    max_seq_length = max(seq_lengths)

    targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
    targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)
    
    for i, report_ids in enumerate(reports_ids):
        targets[i, :len(report_ids)] = report_ids

    for i, report_masks in enumerate(reports_masks):
        targets_masks[i, :len(report_masks)] = report_masks
    return ((images, torch.LongTensor(targets)), (torch.FloatTensor(targets_masks), onehot_lbls))

def fa_convert(t):
    "A replacement for PyTorch `default_convert` which maintains types and handles `Sequence`s"
    return (default_convert(t) if isinstance(t, _collate_types)
            else type(t)([fa_convert(s) for s in t]) if isinstance(t, Sequence)
            else default_convert(t))
def create_batch(b): return (collate_fn,fa_convert)[False](b)
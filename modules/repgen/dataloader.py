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

class MultiViewRepGenDL():
    def __init__(self, device, *args):
        "Stores away `tab_dl` and `vis_dl`, and overrides `shuffle_fn`"
        self.device = device
        for dl in args: dl.shuffle_fn = self.shuffle_fn
        self.dls = args
        self.count = 0
        self.fake_l = _FakeLoader(self, False, 0, 0, False) #TypeError: __init__() missing 1 required positional argument: 'persistent_workers'
    
    def __len__(self): return len(self.dls[0])
        
    def shuffle_fn(self, idxs):
        "Generates a new `rng` based upon which `DataLoader` is called"
        if self.count == 0: # if we haven't generated an rng yet
            self.rng = self.dls[0].rng.sample(idxs, len(idxs))
            self.count += 1
            return self.rng
        else: return self.rng
        
    def to(self, device): self.device = device
        
    def __iter__(self):
        "Iterate over your `DataLoader`"
        for multidl in self.dls: multidl.fake_l.num_workers=0 #Dataloader worker killed
        z = zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in self.dls])
        for b in z:
            if self.device is not None: b = to_device(b, self.device)
            batch = []
            for multidl in range(len(self.dls)): batch.append(self.dls[multidl].after_batch(b[multidl][0]))
            try: # In case the data is unlabelled
                batch.append(b[1][1])
                yield tuple(batch)
            except:
                yield tuple(batch)
    
    def one_batch(self):
        "Grab a batch from the `DataLoader`"
        with self.fake_l.no_multiproc(): res = first(self)
        if hasattr(self, 'it'): delattr(self, 'it')
        return res
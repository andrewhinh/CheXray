import torch
import numpy as np
from fastai.data.load import _FakeLoader, _loaders

#from modules.sum.dataloader import SumDL  
from fastai.data.load import _FakeLoader, _loaders

class SumDL():
    def __init__(self, device, *args):
        "Stores away `tab_dl` and `vis_dl`, and overrides `shuffle_fn`"
        self.device = device
        for dl in args: dl.shuffle_fn = self.shuffle_fn
        self.dls = args
        self.count = 0
        for dl in self.dls: dl._DataLoader__idxs = sorted(self.dls[0].get_idxs())
        self.fake_l = _FakeLoader(self, False, 0, 0, True) #TypeError: __init__() missing 1 required positional argument: 'persistent_workers'
    
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
        z = zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in self.dls])
        for b in z:
            if self.device is not None: b = to_device(b, self.device)
            batch = []
            for multidl in range(len(self.dls)-1): batch.append(self.dls[multidl].after_batch(b[multidl][0]))
            tab = []
            for i in self.dls[len(self.dls)-1]: tab.append(i)
            batch.extend([tab[0][0], tab[0][1]])
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
    
    def show_batch(self):
        "Show a batch from multiple `DataLoaders`"
        for dl in self.dls: dl.show_batch(max_n=1)
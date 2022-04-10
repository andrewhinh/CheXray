import torch
import torch.nn as nn
import numpy as np

import modules.sum.logits as log
from fastai.vision.all import *

class cast_to_tensor(Module):
    def forward(self, x): return cast(x, Tensor)
        
class SumModel(nn.Module):
    def __init__(self, vis_model, txtcls_model, tab_model, num_classes=14):
        super(SumModel, self).__init__()
        self.models = nn.ModuleList([vis_model, txtcls_model, tab_model])
        self.classes = num_classes

        self.vis_handle = self.models[0][-1][-1].register_forward_hook(log.get_vis_logits)
        self.txtcls_handle = list(self.models[1][-1].children())[-1][-1][-1].register_forward_hook(log.get_txtcls_logits)
        self.tab_handle = self.models[2].layers[-1][0].register_forward_hook(log.get_tab_logits)

        self.handles = [self.vis_handle,
                        self.txtcls_handle,
                        self.tab_handle]
    
        self.mixed_cls = nn.Linear(self.models[0][-1][-1].in_features+list(self.models[1][-1].children())[-1][-1][-1].in_features+self.models[2].layers[-1][0].in_features, 
                                   self.classes)
        
    def remove_my_hooks(self):
        for handle in self.handles: handle.remove()
        return None
        
    def forward(self, x_vis, x_txtcls, x_cat, x_cont): 
        out_vis = self.models[0](x_vis)
        out_txtcls = self.models[1](x_txtcls)    
        out_tab = self.models[2](x_cat, x_cont)    
        
        # Logits
        vis_log = log.glb_vis_logits[0]   # Only grabbling weights, not bias
        txtcls_log = log.glb_txtcls_logits[0]
        tab_log = log.glb_tab_logits[0]

        if len(list(vis_log.shape))<3: 
            vis_log = torch.unsqueeze(vis_log, 0)
            txtcls_log = torch.unsqueeze(txtcls_log, 0)
            tab_log = torch.unsqueeze(tab_log, 0)

        mixed = torch.cat((vis_log, txtcls_log, tab_log), dim=2)

        # Mixed Classifier
        out_mix = self.mixed_cls(torch.squeeze(mixed, 0))
        return (out_vis, out_txtcls[0], out_tab, out_mix)
import torch
import torch.nn as nn
import numpy as np

import modules.sum.logits as log
from fastai.vision.all import *

class SumModel(nn.Module):
    def __init__(self, vis_model, txtcls_model, tab_model, num_classes=14):
        super(SumModel, self).__init__()
        self.models = nn.ModuleList([vis_model]*14+[txtcls_model, tab_model])
        self.mixed_cls = nn.Linear(512*14+200+100, num_classes)

        self.ap_handle = self.models[0][-1][-1].register_forward_hook(log.get_ap_logits)
        self.ap_axial_handle = self.models[1][-1][-1].register_forward_hook(log.get_ap_axial_logits)
        self.ap_lld_handle = self.models[2][-1][-1].register_forward_hook(log.get_ap_lld_logits)
        self.ap_rld_handle = self.models[3][-1][-1].register_forward_hook(log.get_ap_rld_logits)
        self.pa_handle = self.models[4][-1][-1].register_forward_hook(log.get_pa_logits)
        self.pa_lld_handle = self.models[5][-1][-1].register_forward_hook(log.get_pa_lld_logits)
        self.pa_rld_handle = self.models[6][-1][-1].register_forward_hook(log.get_pa_rld_logits)
        self.lateral_handle = self.models[7][-1][-1].register_forward_hook(log.get_lateral_logits)
        self.ll_handle = self.models[8][-1][-1].register_forward_hook(log.get_ll_logits)
        self.lao_handle = self.models[9][-1][-1].register_forward_hook(log.get_lao_logits)
        self.rao_handle = self.models[10][-1][-1].register_forward_hook(log.get_rao_logits)
        self.swimmers_handle = self.models[11][-1][-1].register_forward_hook(log.get_swimmers_logits)
        self.xtable_lateral_handle = self.models[12][-1][-1].register_forward_hook(log.get_xtable_lateral_logits)
        self.lpo_handle = self.models[13][-1][-1].register_forward_hook(log.get_lpo_logits)
        self.txtcls_handle = list(self.models[14][-1].children())[-1][-1][-1].register_forward_hook(log.get_txtcls_logits)
        self.tab_handle = self.models[15].layers[2][0].register_forward_hook(log.get_tab_logits)

        self.handles = [self.ap_handle,
                        self.ap_axial_handle,
                        self.ap_lld_handle,
                        self.ap_rld_handle,
                        self.pa_handle,
                        self.pa_lld_handle,
                        self.pa_rld_handle,
                        self.lateral_handle,
                        self.ll_handle,
                        self.lao_handle,
                        self.rao_handle,
                        self.swimmers_handle,
                        self.xtable_lateral_handle,
                        self.lpo_handle,
                        self.txtcls_handle,
                        self.tab_handle]
    
    def remove_my_hooks(self):
        for handle in self.handles: handle.remove()
        return None
        
    def forward(self, x_ap, x_ap_ax, x_ap_lld, x_ap_rld, x_pa, x_pa_lld, x_pa_rld, x_lat, x_ll, x_lao, x_rao, x_swim, x_xtab, x_lpo, x_txtcls, x_cat, x_cont): 
        out_ap = self.models[0](x_ap)
        out_ap_ax = self.models[1](x_ap_ax)
        out_ap_lld = self.models[2](x_ap_lld)
        out_ap_rld = self.models[3](x_ap_rld)
        out_pa = self.models[4](x_pa)
        out_pa_lld = self.models[5](x_pa_lld)
        out_pa_rld = self.models[6](x_pa_rld)
        out_lat = self.models[7](x_lat)
        out_ll = self.models[8](x_ll)
        out_lao = self.models[9](x_lao)
        out_rao = self.models[10](x_rao)
        out_swim = self.models[11](x_swim)
        out_xtab = self.models[12](x_xtab)
        out_lpo = self.models[13](x_lpo)
        out_txtcls = self.models[14](x_txtcls)    
        out_tab = self.models[15](x_cat, x_cont)    
        
        # Logits
        ap_log = log.glb_ap_logits[0]   # Only grabbling weights, not bias
        ap_ax_log = log.glb_ap_axial_logits[0]
        ap_lld_log = log.glb_ap_lld_logits[0]
        ap_rld_log = log.glb_ap_rld_logits[0]
        pa_log = log.glb_pa_logits[0]
        pa_lld_log = log.glb_pa_lld_logits[0]
        pa_rld_log = log.glb_pa_rld_logits[0]
        lat_log = log.glb_lateral_logits[0]
        ll_log = log.glb_ll_logits[0]
        lao_log = log.glb_lao_logits[0]
        rao_log = log.glb_rao_logits[0]
        swim_log = log.glb_swimmers_logits[0]
        xtab_log = log.glb_xtable_lateral_logits[0]
        lpo_log = log.glb_lpo_logits[0]
        txtcls_log = log.glb_txtcls_logits[0]
        tab_log = log.glb_tab_logits[0]
        
        if len(list(ap_log.shape))<3: 
            ap_log = torch.unsqueeze(ap_log, 0)
            ap_ax_log = torch.unsqueeze(ap_ax_log, 0)
            ap_lld_log = torch.unsqueeze(ap_lld_log, 0)
            ap_rld_log = torch.unsqueeze(ap_rld_log, 0)
            pa_log = torch.unsqueeze(pa_log, 0)
            pa_lld_log = torch.unsqueeze(pa_lld_log, 0)
            pa_rld_log = torch.unsqueeze(pa_rld_log, 0)
            lat_log = torch.unsqueeze(lat_log, 0)
            ll_log = torch.unsqueeze(ll_log, 0)
            lao_log = torch.unsqueeze(lao_log, 0)
            rao_log = torch.unsqueeze(rao_log, 0)
            swim_log = torch.unsqueeze(swim_log, 0)
            xtab_log = torch.unsqueeze(xtab_log, 0)
            lpo_log = torch.unsqueeze(lpo_log, 0)
            txtcls_log = torch.unsqueeze(txtcls_log, 0)
            tab_log = torch.unsqueeze(tab_log, 0)
        
        mixed = torch.cat((ap_log, ap_ax_log, ap_lld_log, ap_rld_log, pa_log, pa_lld_log, pa_rld_log, lat_log, ll_log, lao_log, rao_log, swim_log, xtab_log, lpo_log, txtcls_log, tab_log), dim=2)
        
        # Mixed Classifier
        out_mix = self.mixed_cls(mixed)
        return (out_ap, out_ap_ax, out_ap_lld, out_ap_rld, out_pa, out_pa_lld, out_pa_rld, out_lat, out_ll, out_lao, out_rao, out_swim, out_xtab, out_lpo, out_txtcls[0], out_tab, out_mix[0])
import torch
import torch.nn as nn
import numpy as np

from modules.repgen.visual_extractor import VisualExtractor
from modules.repgen.encoder_decoder import EncoderDecoder
import modules.repgen.logits as log
from fastai.vision.all import *

class R2GenModel(nn.Module):
    def __init__(self, 
                visual_extractor, 
                pretrained, 
                num_layers, 
                d_model, 
                d_ff, 
                num_heads, 
                dropout, 
                rm_num_slots, 
                rm_num_heads, 
                rm_d_model,
                vocab,
                input_encoding_size,
                drop_prob_lm, 
                max_seq_length, 
                att_feat_size, 
                use_bn,
                beam_size, 
                group_size, 
                sample_n, 
                sample_method, 
                temperature,
                output_logsoftmax,
                decoding_constraint,
                block_trigrams,
                diversity_lambda,
                suppress_UNK, 
                length_penalty,
                mode):
        super(R2GenModel, self).__init__()
        self.visual_extractor = VisualExtractor(visual_extractor, pretrained)
        self.encoder_decoder = EncoderDecoder(num_layers, 
                                            d_model, 
                                            d_ff, 
                                            num_heads, 
                                            dropout, 
                                            rm_num_slots, 
                                            rm_num_heads, 
                                            rm_d_model,
                                            vocab,
                                            input_encoding_size,
                                            drop_prob_lm, 
                                            max_seq_length, 
                                            att_feat_size, 
                                            use_bn,
                                            beam_size, 
                                            group_size, 
                                            sample_n, 
                                            sample_method, 
                                            temperature,
                                            output_logsoftmax,
                                            decoding_constraint,
                                            block_trigrams,
                                            diversity_lambda,
                                            suppress_UNK, 
                                            length_penalty)
        self.forward = self.forward_mimic_cxr
        self.mode = mode
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    def forward_mimic_cxr(self, imgrep):
        images, reports_ids = imgrep
        att_feats, fc_feats = self.visual_extractor(images)
        if self.mode == 'forward': output = self.encoder_decoder(fc_feats, att_feats, reports_ids, mode='forward')
        elif self.mode == 'sample': output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        return (output, reports_ids)

class MultiViewR2GenModel(nn.Module):
    def __init__(self, model, len_vocab, len_views, num_last_in=512): #num_last_in = model's last layer's num_in_features
        super(MultiViewR2GenModel, self).__init__()
        self.models = nn.ModuleList([model]*14)
        self.mixed_cls = nn.Linear(num_last_in*len_views, len_vocab)

        self.ap_handle = list(list(self.models[0].children())[1].children())[-1].register_forward_hook(log.get_ap_logits)
        self.ap_axial_handle = list(list(self.models[1].children())[1].children())[-1].register_forward_hook(log.get_ap_axial_logits)
        self.ap_lld_handle = list(list(self.models[2].children())[1].children())[-1].register_forward_hook(log.get_ap_lld_logits)
        self.ap_rld_handle = list(list(self.models[3].children())[1].children())[-1].register_forward_hook(log.get_ap_rld_logits)
        self.pa_handle = list(list(self.models[4].children())[1].children())[-1].register_forward_hook(log.get_pa_logits)
        self.pa_lld_handle = list(list(self.models[5].children())[1].children())[-1].register_forward_hook(log.get_pa_lld_logits)
        self.pa_rld_handle = list(list(self.models[6].children())[1].children())[-1].register_forward_hook(log.get_pa_rld_logits)
        self.lateral_handle = list(list(self.models[7].children())[1].children())[-1].register_forward_hook(log.get_lateral_logits)
        self.ll_handle = list(list(self.models[8].children())[1].children())[-1].register_forward_hook(log.get_ll_logits)
        self.lao_handle = list(list(self.models[9].children())[1].children())[-1].register_forward_hook(log.get_lao_logits)
        self.rao_handle = list(list(self.models[10].children())[1].children())[-1].register_forward_hook(log.get_rao_logits)
        self.swimmers_handle = list(list(self.models[11].children())[1].children())[-1].register_forward_hook(log.get_swimmers_logits)
        self.xtable_lateral_handle = list(list(self.models[12].children())[1].children())[-1].register_forward_hook(log.get_xtable_lateral_logits)
        self.lpo_handle = list(list(self.models[13].children())[1].children())[-1].register_forward_hook(log.get_lpo_logits)
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
                        self.lpo_handle]
    
    def remove_my_hooks(self):
        for handle in self.handles: handle.remove()
        return None
        
    def forward(self, x_ap, x_ap_ax, x_ap_lld, x_ap_rld, x_pa, x_pa_lld, x_pa_rld, x_lat, x_ll, x_lao, x_rao, x_swim, x_xtab, x_lpo): 
        out_ap, rep = self.models[0](x_ap)
        out_ap_ax, _ = self.models[1](x_ap_ax)
        out_ap_lld, _ = self.models[2](x_ap_lld)
        out_ap_rld, _ = self.models[3](x_ap_rld)
        out_pa, _ = self.models[4](x_pa)
        out_pa_lld, _ = self.models[5](x_pa_lld)
        out_pa_rld, _ = self.models[6](x_pa_rld)
        out_lat, _ = self.models[7](x_lat)
        out_ll, _ = self.models[8](x_ll)
        out_lao, _ = self.models[9](x_lao)
        out_rao, _ = self.models[10](x_rao)
        out_swim, _ = self.models[11](x_swim)
        out_xtab, _ = self.models[12](x_xtab)
        out_lpo, _ = self.models[13](x_lpo)
        
        # Logits
        ap_log = log.glb_ap_logits[0]   # Only grabbling weights, not bias'
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
            
        mixed = torch.cat((ap_log, ap_ax_log, ap_lld_log, ap_rld_log, pa_log, pa_lld_log, pa_rld_log, lat_log, ll_log, lao_log, rao_log, swim_log, xtab_log, lpo_log), dim=2)
        
        # Mixed Classifier
        out_mix = self.mixed_cls(mixed)
        return (out_ap, out_ap_ax, out_ap_lld, out_ap_rld, out_pa, out_pa_lld, out_pa_rld, out_lat, out_ll, out_lao, out_rao, out_swim, out_xtab, out_lpo, out_mix, rep)
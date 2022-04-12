import torch
import torch.nn as nn
import numpy as np

from modules.repgen.visual_extractor import VisualExtractor
from modules.repgen.encoder_decoder import EncoderDecoder

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
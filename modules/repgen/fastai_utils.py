import torch
from fastai.torch_core import params, to_float
from fastai.callback.core import Callback

def rep_gen(model): return [params(model.visual_extractor), params(model.encoder_decoder)]
class SelectPred(Callback):
    def after_pred(self): 
        if self.pred[0].dtype==torch.float16: self.learn.pred[0] = to_float(self.pred[0]) #Since model returns tuple and only needs predictions
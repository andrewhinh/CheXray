import torch
import torch.nn as nn

def loss_func(pred, targ): return nn.BCEWithLogitsLoss()(torch.tensor(pred), torch.tensor(targ))

class SumGradientBlending(nn.Module):
    def __init__(self, loss_scale=1.0, set_grad=False, *args):
        "Expects weights for each model, the combined model, and an overall scale"
        super(SumGradientBlending, self).__init__()
        self.weights = args
        self.scale = loss_scale
        self.set_grad = set_grad
        
    def forward(self, xb, yb):
        outs = list(xb)
        targ = yb
        "Gathers `self.loss` for each model, weighs, then sums"
        loss=0
        for idx in range(len(self.weights)): loss+=loss_func(outs[idx], targ) * self.scale * self.weights[idx]
        if self.set_grad: loss.requires_grad = True
        return loss
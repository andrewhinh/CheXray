import torch
import torch.nn as nn

class LanguageModelCriterion(nn.Module):
    def __init__(self): super(LanguageModelCriterion, self).__init__()
    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output
def compute_loss(predrep, masklbl, reduction=None):
    preds, reps = predrep
    masks, _ = masklbl
    criterion = LanguageModelCriterion()
    loss = criterion(preds, reps[:, 1:], masks[:, 1:]).mean()
    return loss

class myGradientBlending(nn.Module):
    def __init__(self, loss_scale=1.0, *args):
        "Expects weights for each model, the combined model, and an overall scale"
        super(myGradientBlending, self).__init__()
        self.weights = args
        self.scale = loss_scale
        
    def forward(self, xb, yb):
        outs = list(xb)
        masklbl = yb
        "Gathers `self.loss` for each model, weighs, then sums"
        loss=0
        for idx in range(len(self.weights)): loss+=compute_loss([outs[idx], outs[-1]], masklbl) * self.scale * self.weights[idx]
        return loss
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
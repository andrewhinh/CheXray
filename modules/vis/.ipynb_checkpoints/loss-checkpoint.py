import torch 
import torch.nn as nn
def MultiLabelSmoothingLoss(pred, target, epsilon=0.05, reduction=None):
    temp = target.detach().clone()
    temp.requires_grad = True
    for i, x in enumerate(temp): temp[i] = torch.where(x==1.0, 1.0-epsilon, epsilon)
    smoothed_criterion = nn.BCEWithLogitsLoss()(pred, temp)
    return smoothed_criterion
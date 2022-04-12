from fastai.metrics import APScoreMulti

#Metrics
def single(pred, targ, idx=None, thresh=None, beta=None, weights=None):
    if weights:
        all_inp=0
        for weight in range(len(weights)): all_inp += pred[weight] * weights[weight]
        pred = all_inp/len(weights)
    else:
        if idx is not None: pred = pred[idx]
        else: pred = pred[-1]

    return APScoreMulti(average='weighted')(pred, targ)

def ap_weighted(pred, targ, weights): return single(pred, targ, None, None, None, weights)
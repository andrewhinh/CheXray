from fastai.vision.all import *

#Metrics
def single(pred, targ, metric, idx=None, thresh=None, beta=None, weights=None):
    if weights:
        all_inp=0
        for weight in range(len(weights)): all_inp += pred[weight] * weights[weight]
        pred = all_inp/len(weights)
    else:
        if idx is not None: pred = pred[idx]
        else: pred = pred[-1]

    if metric=='ap': return APScoreMulti(average='weighted')(pred, targ)
    elif metric=='roc': return RocAucMulti(average='weighted')(pred, targ)
    elif metric=="acc": return accuracy_multi(pred, targ, thresh)
    else:
        pred = pred.sigmoid()
        pred[pred>=thresh]=1
        pred[pred<thresh]=0
        if metric=='prec': return PrecisionMulti(thresh=thresh, average='weighted')(pred, targ)
        elif metric=='rec': return RecallMulti(thresh=thresh, average='weighted')(pred, targ)
        elif metric=='fbeta': return FBetaMulti(beta=beta, thresh=thresh, average='weighted')(pred, targ)

def ap_vis(pred, targ): return single(pred, targ, "ap", 0)
def ap_txtcls(pred, targ): return single(pred, targ, "ap", 1)
def ap_tab(pred, targ): return single(pred, targ, "ap", 2)
def ap_multi(pred, targ): return single(pred, targ, "ap")
def ap_weighted(pred, targ, weights): return single(pred, targ, "ap", None, None, None, weights)

def roc_vis(pred, targ): return single(pred, targ, "roc", 0)
def roc_txtcls(pred, targ): return single(pred, targ, "roc", 1)
def roc_tab(pred, targ): return single(pred, targ, "roc", 2)
def roc_multi(pred, targ): return single(pred, targ, "roc")
def roc_weighted(pred, targ, weights): return single(pred, targ, "roc", None, None, None, weights)
    
thresh=0.5
def acc_vis(pred, targ, thresh=thresh): return single(pred, targ, "acc", 0, thresh)
def acc_txtcls(pred, targ, thresh=thresh): return single(pred, targ, "acc", 1, thresh)
def acc_tab(pred, targ, thresh=thresh): return single(pred, targ, "acc", 2, thresh)
def acc_multi(pred, targ, thresh=thresh): return single(pred, targ, "acc", None, thresh)
def acc_weighted(pred, targ, weights, thresh=thresh): return single(pred, targ, "acc", None, thresh, None, weights)

def prec_vis(pred, targ, thresh=thresh): return single(pred, targ, "prec", 0, thresh)
def prec_txtcls(pred, targ, thresh=thresh): return single(pred, targ, "prec", 1, thresh)
def prec_tab(pred, targ, thresh=thresh): return single(pred, targ, "prec", 2, thresh)
def prec_multi(pred, targ, thresh=thresh): return single(pred, targ, "prec", None, thresh)
def prec_weighted(pred, targ, weights, thresh=thresh): return single(pred, targ, "prec", None, thresh, None, weights)

def rec_vis(pred, targ, thresh=thresh): return single(pred, targ, "rec", 0, thresh)
def rec_txtcls(pred, targ, thresh=thresh): return single(pred, targ, "rec", 1, thresh)
def rec_tab(pred, targ, thresh=thresh): return single(pred, targ, "rec", 2, thresh)
def rec_multi(pred, targ, thresh=thresh): return single(pred, targ, "rec", None, thresh)
def rec_weighted(pred, targ, weights, thresh=thresh): return single(pred, targ, "rec", None, thresh, None, weights)

beta=1
def fbeta_vis(pred, targ, thresh=thresh, beta=beta): return single(pred, targ, "fbeta", 0, thresh, beta)
def fbeta_txtcls(pred, targ, thresh=thresh, beta=beta): return single(pred, targ, "fbeta", 1, thresh, beta)
def fbeta_tab(pred, targ, thresh=thresh, beta=beta): return single(pred, targ, "fbeta", 2, thresh, beta)
def fbeta_multi(pred, targ, thresh=thresh, beta=beta): return single(pred, targ, "fbeta", None, thresh, beta)
def fbeta_weighted(pred, targ, weights, thresh=thresh, beta=beta): return single(pred, targ, "fbeta", None, thresh, beta, weights)
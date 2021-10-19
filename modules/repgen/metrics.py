import pickle
from fastai.vision.all import *
from modules.repgen.pycocoevalcap.bleu.bleu import Bleu
from modules.repgen.pycocoevalcap.meteor import Meteor
from modules.repgen.pycocoevalcap.rouge import Rouge

class_learn = load_learner(Path('./models/txtcls.pkl'))      
with open(Path('./modules/repgen/vocab.pkl'), 'rb') as f: vocab = pickle.load(f)

def compute_scores(predrep, masklbl, scorer, thresh=None):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
    :param predrep: tuple of Tensors of size (bs, rep_len, vocab_size) and (bs, rep_len)
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    preds, reps = predrep
    _, lbls = masklbl
    val_res=[] #List of size (bs, rep_len)
    val_gts=[] #List of size (bs, rep_len)
    for report in list(preds): #Tensor of size (rep_len, vocab_size)
        words=[] #For every word in report (size rep_len)
        for word in report: words.append((word == word.max()).nonzero(as_tuple=True)[0].item()) #For each word in report (size vocab_size), append word with max log_softmax
        wordstostr = [] #For every word in report (size rep_len)
        for word in words: wordstostr.append(vocab[word]) #For each word in report (size vocab_size) append index for vocab
        val_res.append(" ".join(wordstostr))
    for report in list(reps): #Tensor of size (rep_len)
        wordstostr = [] #For every word in report (size vocab_size)
        for word in report: wordstostr.append(vocab[word]) #For each word in report (size vocab_size) append index for vocab
        val_gts.append(" ".join(wordstostr)) 
    res, gts = {i: [re] for i, re in enumerate(val_res)}, {i: [gt] for i, gt in enumerate(val_gts)}
    
    def nlg(scorer, idx=None):
        try: score, _ = scorer.compute_score(gts, res, verbose=0)
        except TypeError: score, _ = scorer.compute_score(gts, res)
        if type(score)==list:
            metrics=[]
            for sc in score: metrics.append(sc) 
            return metrics[idx]
        else: return score
    if scorer=='bleu1': return nlg(Bleu(4), 0)
    if scorer=='bleu2': return nlg(Bleu(4), 1)
    if scorer=='bleu3': return nlg(Bleu(4), 2)
    if scorer=='bleu4': return nlg(Bleu(4), 3)
    if scorer=='meteor': return nlg(Meteor())
    if scorer=='rouge': return nlg(Rouge())
    
    def ce_metric(scorer):
        predlbls = []
        for report in val_res: 
            _, _, act = class_learn.predict(report)
            act = torch.where(act>=thresh, 1, 0)
            predlbls.append(act)
        for predlbl in predlbls: predlbl.unsqueeze(0)
        predlbls = torch.stack(predlbls)
        return scorer(predlbls, lbls)
    if scorer=='precision': return ce_metric(PrecisionMulti(thresh=thresh, average='weighted'))
    if scorer=='recall': return ce_metric(RecallMulti(thresh=thresh, average='weighted'))
    if scorer=='f1': return ce_metric(F1ScoreMulti(thresh=thresh, average='weighted'))

def bleu1(predrep, masklbl): return compute_scores(predrep, masklbl, 'bleu1')
def bleu2(predrep, masklbl): return compute_scores(predrep, masklbl, 'bleu2')
def bleu3(predrep, masklbl): return compute_scores(predrep, masklbl, 'bleu3')
def bleu4(predrep, masklbl): return compute_scores(predrep, masklbl, 'bleu4')
def meteor(predrep, masklbl): return compute_scores(predrep, masklbl, 'meteor')
def rouge(predrep, masklbl): return compute_scores(predrep, masklbl, 'rouge')
def precision(predrep, masklbl, thresh=0.5): return compute_scores(predrep, masklbl, 'precision', thresh)
def recall(predrep, masklbl, thresh=0.5): return compute_scores(predrep, masklbl, 'recall', thresh)
def f1(predrep, masklbl, thresh=0.5): return compute_scores(predrep, masklbl, 'f1', thresh)

max_seq_len=100
# Acc Metrics
def single(predrep, masklbl, metric, view_idx=None, thresh=None, weights=None):
    if weights:
        all_inp=0
        if list(predrep[0].shape)[-1]!=list(predrep[len(weights)-1].shape)[-1]:
            new_report = []
            for report in predrep[len(weights)-1]:
                new_idxs = []
                for poss_words in report:
                    idxs = torch.argmax(poss_words, dim=0).item()
                    new_idxs.append(idxs)
                new_report.append(new_idxs)
            if len(new_report[0])<max_seq_len: new_report[0].extend([0]*(max_seq_len-len(new_report[0])))
            temp = list(predrep[:len(weights)-1])
            for tensor in range(len(temp)): temp[tensor] = temp[tensor].cpu()
            temp.append(torch.tensor(new_report))
            predrep = torch.stack(temp)
        for weight in range(len(weights)): all_inp += predrep[weight] * weights[weight]
        pred = all_inp/len(weights)
    else:
        if view_idx is not None: pred = predrep[view_idx]
        else: pred = predrep[-2]
    rep = predrep[-1]
    if thresh is not None: return metric([pred, rep], masklbl, thresh)
    else: return metric([pred, rep], masklbl)
    
def bleu1_ap(predrep, masklbl): return single(predrep, masklbl, bleu1, 0)
def bleu1_ap_ax(predrep, masklbl): return single(predrep, masklbl, bleu1, 1)
def bleu1_ap_lld(predrep, masklbl): return single(predrep, masklbl, bleu1, 2)
def bleu1_ap_rld(predrep, masklbl): return single(predrep, masklbl, bleu1, 3)
def bleu1_pa(predrep, masklbl): return single(predrep, masklbl, bleu1, 4)
def bleu1_pa_lld(predrep, masklbl): return single(predrep, masklbl, bleu1, 5)
def bleu1_pa_rld(predrep, masklbl): return single(predrep, masklbl, bleu1, 6)
def bleu1_lat(predrep, masklbl): return single(predrep, masklbl, bleu1, 7)
def bleu1_ll(predrep, masklbl): return single(predrep, masklbl, bleu1, 8)
def bleu1_lao(predrep, masklbl): return single(predrep, masklbl, bleu1, 9)
def bleu1_rao(predrep, masklbl): return single(predrep, masklbl, bleu1, 10)
def bleu1_swim(predrep, masklbl): return single(predrep, masklbl, bleu1, 11)
def bleu1_xtab(predrep, masklbl): return single(predrep, masklbl, bleu1, 12)
def bleu1_lpo(predrep, masklbl): return single(predrep, masklbl, bleu1, 13)
def bleu1_multi(predrep, masklbl): return single(predrep, masklbl, bleu1)
def bleu1_weighted(predrep, masklbl, weights): return single(predrep, masklbl, bleu1, None, None, weights)

def bleu2_ap(predrep, masklbl): return single(predrep, masklbl, bleu2, 0)
def bleu2_ap_ax(predrep, masklbl): return single(predrep, masklbl, bleu2, 1)
def bleu2_ap_lld(predrep, masklbl): return single(predrep, masklbl, bleu2, 2)
def bleu2_ap_rld(predrep, masklbl): return single(predrep, masklbl, bleu2, 3)
def bleu2_pa(predrep, masklbl): return single(predrep, masklbl, bleu2, 4)
def bleu2_pa_lld(predrep, masklbl): return single(predrep, masklbl, bleu2, 5)
def bleu2_pa_rld(predrep, masklbl): return single(predrep, masklbl, bleu2, 6)
def bleu2_lat(predrep, masklbl): return single(predrep, masklbl, bleu2, 7)
def bleu2_ll(predrep, masklbl): return single(predrep, masklbl, bleu2, 8)
def bleu2_lao(predrep, masklbl): return single(predrep, masklbl, bleu2, 9)
def bleu2_rao(predrep, masklbl): return single(predrep, masklbl, bleu2, 10)
def bleu2_swim(predrep, masklbl): return single(predrep, masklbl, bleu2, 11)
def bleu2_xtab(predrep, masklbl): return single(predrep, masklbl, bleu2, 12)
def bleu2_lpo(predrep, masklbl): return single(predrep, masklbl, bleu2, 13)
def bleu2_multi(predrep, masklbl): return single(predrep, masklbl, bleu2)
def bleu2_weighted(predrep, masklbl, weights): return single(predrep, masklbl, bleu2, None, None, weights)

def bleu3_ap(predrep, masklbl): return single(predrep, masklbl, bleu3, 0)
def bleu3_ap_ax(predrep, masklbl): return single(predrep, masklbl, bleu3, 1)
def bleu3_ap_lld(predrep, masklbl): return single(predrep, masklbl, bleu3, 2)
def bleu3_ap_rld(predrep, masklbl): return single(predrep, masklbl, bleu3, 3)
def bleu3_pa(predrep, masklbl): return single(predrep, masklbl, bleu3, 4)
def bleu3_pa_lld(predrep, masklbl): return single(predrep, masklbl, bleu3, 5)
def bleu3_pa_rld(predrep, masklbl): return single(predrep, masklbl, bleu3, 6)
def bleu3_lat(predrep, masklbl): return single(predrep, masklbl, bleu3, 7)
def bleu3_ll(predrep, masklbl): return single(predrep, masklbl, bleu3, 8)
def bleu3_lao(predrep, masklbl): return single(predrep, masklbl, bleu3, 9)
def bleu3_rao(predrep, masklbl): return single(predrep, masklbl, bleu3, 10)
def bleu3_swim(predrep, masklbl): return single(predrep, masklbl, bleu3, 11)
def bleu3_xtab(predrep, masklbl): return single(predrep, masklbl, bleu3, 12)
def bleu3_lpo(predrep, masklbl): return single(predrep, masklbl, bleu3, 13)
def bleu3_multi(predrep, masklbl): return single(predrep, masklbl, bleu3)
def bleu3_weighted(predrep, masklbl, weights): return single(predrep, masklbl, bleu3, None, None, weights)

def bleu4_ap(predrep, masklbl): return single(predrep, masklbl, bleu4, 0)
def bleu4_ap_ax(predrep, masklbl): return single(predrep, masklbl, bleu4, 1)
def bleu4_ap_lld(predrep, masklbl): return single(predrep, masklbl, bleu4, 2)
def bleu4_ap_rld(predrep, masklbl): return single(predrep, masklbl, bleu4, 3)
def bleu4_pa(predrep, masklbl): return single(predrep, masklbl, bleu4, 4)
def bleu4_pa_lld(predrep, masklbl): return single(predrep, masklbl, bleu4, 5)
def bleu4_pa_rld(predrep, masklbl): return single(predrep, masklbl, bleu4, 6)
def bleu4_lat(predrep, masklbl): return single(predrep, masklbl, bleu4, 7)
def bleu4_ll(predrep, masklbl): return single(predrep, masklbl, bleu4, 8)
def bleu4_lao(predrep, masklbl): return single(predrep, masklbl, bleu4, 9)
def bleu4_rao(predrep, masklbl): return single(predrep, masklbl, bleu4, 10)
def bleu4_swim(predrep, masklbl): return single(predrep, masklbl, bleu4, 11)
def bleu4_xtab(predrep, masklbl): return single(predrep, masklbl, bleu4, 12)
def bleu4_lpo(predrep, masklbl): return single(predrep, masklbl, bleu4, 13)
def bleu4_multi(predrep, masklbl): return single(predrep, masklbl, bleu4)
def bleu4_weighted(predrep, masklbl, weights): return single(predrep, masklbl, bleu4, None, None, weights)

def meteor_ap(predrep, masklbl): return single(predrep, masklbl, meteor, 0)
def meteor_ap_ax(predrep, masklbl): return single(predrep, masklbl, meteor, 1)
def meteor_ap_lld(predrep, masklbl): return single(predrep, masklbl, meteor, 2)
def meteor_ap_rld(predrep, masklbl): return single(predrep, masklbl, meteor, 3)
def meteor_pa(predrep, masklbl): return single(predrep, masklbl, meteor, 4)
def meteor_pa_lld(predrep, masklbl): return single(predrep, masklbl, meteor, 5)
def meteor_pa_rld(predrep, masklbl): return single(predrep, masklbl, meteor, 6)
def meteor_lat(predrep, masklbl): return single(predrep, masklbl, meteor, 7)
def meteor_ll(predrep, masklbl): return single(predrep, masklbl, meteor, 8)
def meteor_lao(predrep, masklbl): return single(predrep, masklbl, meteor, 9)
def meteor_rao(predrep, masklbl): return single(predrep, masklbl, meteor, 10)
def meteor_swim(predrep, masklbl): return single(predrep, masklbl, meteor, 11)
def meteor_xtab(predrep, masklbl): return single(predrep, masklbl, meteor, 12)
def meteor_lpo(predrep, masklbl): return single(predrep, masklbl, meteor, 13)
def meteor_multi(predrep, masklbl): return single(predrep, masklbl, meteor)
def meteor_weighted(predrep, masklbl, weights): return single(predrep, masklbl, meteor, None, None, weights)

def rouge_ap(predrep, masklbl): return single(predrep, masklbl, rouge, 0)
def rouge_ap_ax(predrep, masklbl): return single(predrep, masklbl, rouge, 1)
def rouge_ap_lld(predrep, masklbl): return single(predrep, masklbl, rouge, 2)
def rouge_ap_rld(predrep, masklbl): return single(predrep, masklbl, rouge, 3)
def rouge_pa(predrep, masklbl): return single(predrep, masklbl, rouge, 4)
def rouge_pa_lld(predrep, masklbl): return single(predrep, masklbl, rouge, 5)
def rouge_pa_rld(predrep, masklbl): return single(predrep, masklbl, rouge, 6)
def rouge_lat(predrep, masklbl): return single(predrep, masklbl, rouge, 7)
def rouge_ll(predrep, masklbl): return single(predrep, masklbl, rouge, 8)
def rouge_lao(predrep, masklbl): return single(predrep, masklbl, rouge, 9)
def rouge_rao(predrep, masklbl): return single(predrep, masklbl, rouge, 10)
def rouge_swim(predrep, masklbl): return single(predrep, masklbl, rouge, 11)
def rouge_xtab(predrep, masklbl): return single(predrep, masklbl, rouge, 12)
def rouge_lpo(predrep, masklbl): return single(predrep, masklbl, rouge, 13)
def rouge_multi(predrep, masklbl): return single(predrep, masklbl, rouge)
def rouge_weighted(predrep, masklbl, weights): return single(predrep, masklbl, rouge, None, None, weights)

thresh=0.5
def precision_ap(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 0, thresh)
def precision_ap_ax(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 1, thresh)
def precision_ap_lld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 2, thresh)
def precision_ap_rld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 3, thresh)
def precision_pa(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 4, thresh)
def precision_pa_lld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 5, thresh)
def precision_pa_rld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 6, thresh)
def precision_lat(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 7, thresh)
def precision_ll(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 8, thresh)
def precision_lao(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 9, thresh)
def precision_rao(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 10, thresh)
def precision_swim(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 11, thresh)
def precision_xtab(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 12, thresh)
def precision_lpo(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, 13, thresh)
def precision_multi(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, precision, None, thresh)
def precision_weighted(predrep, masklbl, weights, thresh=thresh): return single(predrep, masklbl, precision, None, thresh, weights)

def recall_ap(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 0, thresh)
def recall_ap_ax(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 1, thresh)
def recall_ap_lld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 2, thresh)
def recall_ap_rld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 3, thresh)
def recall_pa(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 4, thresh)
def recall_pa_lld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 5, thresh)
def recall_pa_rld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 6, thresh)
def recall_lat(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 7, thresh)
def recall_ll(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 8, thresh)
def recall_lao(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 9, thresh)
def recall_rao(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 10, thresh)
def recall_swim(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 11, thresh)
def recall_xtab(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 12, thresh)
def recall_lpo(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, 13, thresh)
def recall_multi(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, recall, None, thresh)
def recall_weighted(predrep, masklbl, weights, thresh=thresh): return single(predrep, masklbl, recall, None, thresh, weights)

def f1_ap(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 0, thresh)
def f1_ap_ax(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 1, thresh)
def f1_ap_lld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 2, thresh)
def f1_ap_rld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 3, thresh)
def f1_pa(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 4, thresh)
def f1_pa_lld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 5, thresh)
def f1_pa_rld(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 6, thresh)
def f1_lat(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 7, thresh)
def f1_ll(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 8, thresh)
def f1_lao(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 9, thresh)
def f1_rao(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 10, thresh)
def f1_swim(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 11, thresh)
def f1_xtab(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 12, thresh)
def f1_lpo(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, 13, thresh)
def f1_multi(predrep, masklbl, thresh=thresh): return single(predrep, masklbl, f1, None, thresh)
def f1_weighted(predrep, masklbl, weights, thresh=thresh): return single(predrep, masklbl, f1, None, thresh, weights)
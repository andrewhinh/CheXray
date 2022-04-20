from pickle import load
from fastai.learner import load_learner
from pathlib import Path
from modules.repgen.pycocoevalcap.bleu.bleu import Bleu

class_learn = load_learner(Path('./models/txtcls.pkl'))      
with open(Path('./modules/repgen/vocab.pkl'), 'rb') as f: vocab = load(f)

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

def bleu4(predrep, masklbl): return compute_scores(predrep, masklbl, 'bleu4')
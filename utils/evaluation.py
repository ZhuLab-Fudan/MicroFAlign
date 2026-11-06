"""Weighted maximum F-measure and auxiliary-model evaluation."""
import torch
from utils.data import read_lines

ROOTS = {'GO:0008150', 'GO:0003674', 'GO:0005575'}


def fmax(term_scores, pro_anno, type_list, ic):
    best = 0.0, 1.0
    for cut in (c / 100 for c in range(101)):
        psum, pnum, rsum, rnum = 0.0, 0, 0.0, 0
        for seq in type_list:
            if seq in pro_anno and len(pro_anno[seq]) > 1:
                rnum += 1
                if seq in term_scores:
                    total, correct = 0, 0
                    for term in term_scores[seq]:
                        if term not in ROOTS and term_scores[seq][term] >= cut:
                            total += ic.get(term, 0)
                            if term in pro_anno[seq]:
                                correct += ic.get(term, 0)
                    true_weight = 0.0
                    for term in pro_anno[seq]:
                        if term not in ROOTS:
                            true_weight += ic.get(term, 0)
                    if true_weight == 0:
                        continue
                    rsum += correct / true_weight
                    if total > 0:
                        pnum += 1
                        psum += correct / total
        if pnum > 0:
            precision, recall = psum / pnum, rsum / rnum
            score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            best = max(best, (score, cut))
    return best


@torch.no_grad()
def evaluate(model, batches, device, labels):
    model.eval()
    scores = {}
    for batch in batches:
        probabilities = model(batch['input'].to(device)).sigmoid().cpu()
        for index, row in zip(batch['index'].tolist(), probabilities):
            raw = {go: float(p) for go, p in zip(labels, row) if p >= 0.01}
            propagated = batches.dataset.ontology.transfer_scores(raw)
            scores[batches.dataset.pids[index]] = dict(sorted(propagated.items(), key=lambda x: x[1], reverse=True)[:100])
    ic = dict((go, float(value)) for go, value in (line.split() for line in read_lines(batches.dataset.ic_path)))
    return fmax(scores, batches.dataset.truth, batches.dataset.pids, ic)

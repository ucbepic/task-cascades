import re

def exact_match(pred, gold):
    return pred == gold

def set_match(pred, gold):
    # Lowercase and split on ,
    pred_set = pred.lower().split(";")
    pred_set = [re.sub(r'[^a-zA-Z0-9]', '', p.strip()) for p in pred_set if p.strip()]
    pred_set = set(pred_set)
    gold_set = gold.lower().split(";")
    gold_set = [re.sub(r'[^a-zA-Z0-9]', '', g.strip()) for g in gold_set if g.strip()]
    gold_set = set(gold_set)
    return pred_set == gold_set
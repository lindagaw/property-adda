import numpy as np
from sklearn.metrics import accuracy_score

def error_rate(y_trues, y_preds, error_bar):
    hit = 0
    miss = 0

    for y_true, y_pred in zip(y_trues, y_preds):

        if abs(y_true - y_pred) < error_bar:
            hit += 1
        else:
            miss += 1

    return hit/(hit+miss)

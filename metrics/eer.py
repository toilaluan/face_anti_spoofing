from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
from sklearn.metrics import roc_curve
def eer(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer
if __name__ == '__main__':
    y = [1, 0, 1, 1]
    scores = [1/2, 1/2, 1/2, 1/2]
    eer_score = eer(y,scores)
    print(eer_score)
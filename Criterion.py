import numpy as np
from sklearn.metrics import precision_recall_curve
import os
class Criterion:
    def precision_recall(self,dst,epoch,ytrue,ypred):
        precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
        np.save(os.path.join(dst,"P%s"%epoch),precision)
        np.save(os.path.join(dst, "R%s" % epoch), recall)
        np.save(os.path.join(dst, "T%s" % epoch), thresholds)
import numpy as np
from sklearn import metrics


def calculate_metrics(labels, predictions, verbose: bool):
    """ 
    Calculate the confusion matrix, acc, aucroc, and aucprc.
        
    Returns
    -------

    """
    predictions = np.array(predictions)
    predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(labels, predictions.argmax(axis=1))
    cf = cf.astype(np.float32)
    if verbose:
        print("confusion matrix:")
        print(cf)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    auroc = metrics.roc_auc_score(labels, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(labels, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)

    return {"acc": acc,
            "auroc": auroc,
            "auprc": auprc,
            "cf": cf.tolist()}
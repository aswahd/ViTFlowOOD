import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sk


def get_auroc(y_true, scores, make_plot=True, add_to_title=None, name='auroc.png'):
    auroc = roc_auc_score(y_true, scores)

    to_replot_dict = dict()

    out_scores, in_scores = scores[y_true == 0], scores[y_true == 1]

    if make_plot:
        plt.figure(figsize=(5.5, 3), dpi=100)

        if add_to_title is not None:
            plt.title(add_to_title + " AUROC=" + str(float(auroc * 100))[:6] + "%", fontsize=14)
        else:
            plt.title(" AUROC=" + str(float(auroc * 100))[:6] + "%", fontsize=10)

    vals, bins = np.histogram(out_scores, bins=100)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    if make_plot:
        plt.plot(bin_centers, vals, linewidth=4, color="crimson", marker="", label="out test")
        plt.fill_between(bin_centers, vals, [0] * len(vals), color="crimson", alpha=0.3)

    to_replot_dict["out_bin_centers"] = bin_centers
    to_replot_dict["out_vals"] = vals

    vals, bins = np.histogram(in_scores, bins=100)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    if make_plot:
        plt.plot(bin_centers, vals, linewidth=4, color="navy", marker="", label="in test")
        plt.fill_between(bin_centers, vals, [0] * len(vals), color="navy", alpha=0.3)

    to_replot_dict["in_bin_centers"] = bin_centers
    to_replot_dict["in_vals"] = vals

    if make_plot:
        plt.xlabel("Score", fontsize=14)
        plt.ylabel("Count", fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim([0, None])

        plt.legend(fontsize=14)

        plt.tight_layout()
        plt.savefig(name)
        plt.close()

    return auroc, to_replot_dict


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[len(pos):] += 1
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr


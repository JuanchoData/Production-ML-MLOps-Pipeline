import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def find_best_threshold(
    y_true,
    y_prob,
    thresholds=None,
):
    """
    Find the probability threshold that maximizes
    F1 score for the positive class.

    Parameters
    ----------
    y_true
        True binary labels.

    y_prob
        Predicted probabilities for FAIL.

    thresholds
        Threshold values to evaluate.

    Returns
    -------
    results
        List containing metrics for each threshold.

    best_result
        Threshold with highest FAIL F1 score.
    """

    if thresholds is None:
        thresholds = np.arange(
            0.01,
            0.51,
            0.01,
        )

    results = []

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    best_result = max(
        results,
        key=lambda x: x["f1"],
    )

    return results, best_result
import os

import numpy as np
from datasets import load_metric


def compute_base_metrics(eval_pred):
    """
    compute accuracy, recall and f1 score based model prediction and label
    :param eval_pred: model prediction(logit) and label
    :return: accuracy, recall and f1 score
    """
    acc_metric_path = os.path.join("definition/metrics", "accuracy")
    recall_metric_path = os.path.join("definition/metrics", "recall")
    f1_metric_path = os.path.join("definition/metrics", "f1")
    acc_metric = load_metric(acc_metric_path)
    recall_metric = load_metric(recall_metric_path)
    f1_metric = load_metric(f1_metric_path)

    logit, labels = eval_pred
    predictions = np.argmax(logit, axis=-1)

    result = {}
    result.update(
        acc_metric.compute(predictions=predictions, references=labels))
    result.update(
        recall_metric.compute(predictions=predictions, references=labels))
    result.update(
        f1_metric.compute(predictions=predictions, references=labels))

    return result

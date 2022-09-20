from __future__ import print_function
import datasets

rouge_metric = datasets.load_metric('rouge')


def rouge(prediction, ground_truth):
    score = rouge_metric.compute(
        predictions=[prediction],
        references=[ground_truth],
        **{'use_agregator': False, 'use_stemmer': True, 'rouge_types': ['rougeL']}
    )
    return score['rougeL'][0].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

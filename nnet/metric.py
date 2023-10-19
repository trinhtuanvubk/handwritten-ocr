
from rapidfuzz.distance import Levenshtein
import numpy as np
import string


class RecMetric(object):
    def __init__(self,
                 main_indicator='acc',
                 using_cer=True,
                 is_filter=False,
                 ignore_space=True,
                 **kwargs):
        self.main_indicator = main_indicator
        self.using_cer = using_cer
        if self.using_cer is not None:
            from evaluate import load
            self.cer = load('cer')
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        preds = [i[0] for i in preds]
        labels = [j[0] for j in labels]
        for pred, target in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis

        if self.using_cer:
            cer_score = self.cer.compute(predictions=preds, references=labels)
        else:
            cer_score = None
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps),
            'cer_score': cer_score
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
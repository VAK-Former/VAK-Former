from mmseg.evaluation.metrics import IoUMetric
from mmengine.registry import METRICS
import numpy as np

@METRICS.register_module()
class IoUF1Metric(IoUMetric):

    def process(self, data_batch, data_samples):
        """Override to store raw predictions before IoUMetric reduces them."""
        if not hasattr(self, "raw_results"):
            self.raw_results = []

        for sample in data_samples:
            pred = sample['pred_sem_seg']['data']
            gt = sample['gt_sem_seg']['data']

            # Ensure both are tensors
            if isinstance(pred, tuple):
                pred = pred[0]
            if isinstance(gt, tuple):
                gt = gt[0]

            self.raw_results.append((pred.cpu().numpy(), gt.cpu().numpy()))

        # Still run normal IoU processing
        return super().process(data_batch, data_samples)


    def compute_metrics(self, results):
        """Compute IoU (super class) + F1 (custom)."""
        metrics = super().compute_metrics(results)

        num_classes = len(self.dataset_meta['classes'])
        ignore_index = self.ignore_index

        cm = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Use raw masks instead of IoUMetric’s area tuples
        for pred, target in self.raw_results:
            pred = pred.flatten()
            target = target.flatten()

            mask = target != ignore_index
            pred = pred[mask]
            target = target[mask]

            for t, p in zip(target, pred):
                if t < num_classes and p < num_classes:
                    cm[t, p] += 1

        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        metrics['Precision'] = precision.tolist()
        metrics['Recall'] = recall.tolist()
        metrics['F1'] = f1.tolist()
        metrics['mF1'] = float(f1.mean())

        return metrics

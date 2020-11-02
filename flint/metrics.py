import torch
from pytorch_lightning.metrics import Metric
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


class BalancedAccuracy(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False, process_group=None):
        super().__init__(compute_on_step, dist_sync_on_step, process_group)

        self.add_state('predicted', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('actual', default=torch.tensor([]), dist_reduce_fx='cat')

    def update(self, predicted, actual):
        predicted = torch.argmax(predicted, dim=1)
        assert predicted.shape == actual.shape

        self.predicted = torch.cat([self.predicted, predicted])
        self.actual = torch.cat([self.actual, actual])

    def compute(self):
        return balanced_accuracy_score(self.actual.cpu(), self.predicted.cpu())


class ROCAUC(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False, process_group=None):
        super().__init__(compute_on_step, dist_sync_on_step, process_group)

        self.add_state('predicted', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('actual', default=torch.tensor([]), dist_reduce_fx='cat')

    def update(self, predicted, actual):
        self.predicted = torch.cat([self.predicted, predicted[:, 1]])
        self.actual = torch.cat([self.actual, actual])

    def compute(self):
        return roc_auc_score(self.actual.cpu(), self.predicted.cpu())

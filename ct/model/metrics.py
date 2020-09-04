from keras import metrics
from abc import abstractmethod


class CustomMetric(metrics.Metric):
    def __init__(self,
                 name=None,
                 dtype=None,
                 **kwargs):
        """Custom, stateful metrics are not updated correctly during keras fit_generator.
        Instead, although the `update_state` function will be correctly callled, the changes
        will not be displayed.

        The `reset_states` still works as expected, and as such this can be used as a "last resort"
        to at least display any desired information. This information will however be displayed
        at the next epoch, and will not change for any batch during an epoch.

        There are bug reports filed on github:
        #19186: https://github.com/tensorflow/tensorflow/issues/19186
        #20529: https://github.com/tensorflow/tensorflow/issues/20529
        """
        super().__init__(name=name, dtype=None, **kwargs)
        self.custom_stateful_metric = True

    @abstractmethod
    def update_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def reset_states(self):
        pass


class AttentionReconstructionMetric(CustomMetric):
    def __init__(self,
                 ct,
                 name='ar_loss',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss = self.add_weight(name='_ar_loss',
                                    initializer='zeros')
        self.ct = ct

    def update_state(self,
                     *args,
                     **kwargs):
        if len(self.ct._loss_ar_batch) > 0:
            ar_loss = self.ct._loss_ar_batch[-1]
            self.loss.assign_add(ar_loss)

    def result(self):
        return self.loss

    def reset_states(self):
        self.loss.assign(0)

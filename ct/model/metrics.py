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
                 name='ar_loss',
                 display_after_each_epoch=True,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss = self.add_weight(name='_ar_loss',
                                    initializer='zeros')
        self.loss.assign(12)
        self.display_after_each_epoch = display_after_each_epoch

    def update_state(self,
                     *args,
                     ar_loss=0,
                     **kwargs):
        self.loss.assign_add(ar_loss)
        self._loss = ar_loss

    def result(self):
        return self.loss

    def reset_states(self):
        if not self.display_after_each_epoch:
            return self.loss.assign(0)
        self.loss.assign_add(12)

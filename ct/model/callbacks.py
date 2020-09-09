import os
import warnings
import numpy as np
from keras import callbacks
from keras.callbacks import Callback


class ClearCompressedMemory(Callback):
    """Clears the `Compressed Memory` after each batch.
    Effectively makes sure that the `Compressed Memory` only
    contains 0's."""

    def on_train_batch_end(self, batch, logs=None):
        self.model.compressed_memory *= 0


class WriteLogsToFile(Callback):
    """Writes the logs created during training containing losses and metrics
    to an additional file.
    """
    def __init__(self, filepath, overwrite_old_file=False):
        super().__init__()
        self.filepath = filepath
        self.overwrite_old_file = overwrite_old_file
        self.line_ending = '\n'

        if overwrite_old_file:
            if os.path.exists(filepath):
                os.remove(filepath)

        directory = os.path.split(filepath)[0]
        os.makedirs(directory, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        log_msg = '\t'.join(f'{k}={v}' for k, v in logs.items())
        msg = f'{epoch:4d}:   {log_msg}{self.line_ending}'
        with open(file=self.filepath, mode='a+') as file:
            file.write(msg)


class SaveModel(Callback):
    """Saves the model with the specified interval(s).
    """
    def __init__(self,
                 filepath=None,
                 on_epoch_end=True,
                 save_every_n_batches=None,
                 overwrite_old_file=True):
        super().__init__()
        self.filepath = filepath
        self.overwrite_old_file = overwrite_old_file

        if on_epoch_end:
            self.on_epoch_end = self._on_epoch_end

        if save_every_n_batches:
            warnings.filterwarnings('ignore',
                                    category=RuntimeWarning,
                                    module=callbacks.__name__)
            self.save_every_n_batches = save_every_n_batches
            self.on_batch_end = self._on_batch_end

        directory = os.path.split(filepath)[0]
        os.makedirs(directory, exist_ok=True)

    def _on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath, overwrite=self.overwrite_old_file)

    def _on_batch_end(self, batch, logs=None):
        if batch % self.save_every_n_batches == 0 and batch > 0:
            self.model.save(self.filepath, overwrite=self.overwrite_old_file)


class PrintAttentionReconstructionLoss(Callback):
    """Prints the `Attention Reconstruction Loss` explicitly after each epoch,
    as the stateful metrics appear to have some issues working with keras.Model.fit_generator().
    """
    def on_epoch_end(self, epoch, logs=None):
        loss = np.mean(self.model._loss_ar_batch)
        msg = f'----> Epoch {epoch+1}: ar_loss={loss:.4f}'
        print(msg)

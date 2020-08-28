import os
from keras.callbacks import Callback


class ClearCompressedMemory(Callback):
    def on_train_batch_end(self, batch, logs=None):
        self.model.compressed_memory *= 0


class WriteLogsToFile(Callback):
    def __init__(self, filepath, overwrite_old_file=False):
        super().__init__()
        self.filepath = filepath
        self.overwrite_old_file = overwrite_old_file
        self.line_ending = '\n'

        if overwrite_old_file:
            if os.path.exists(filepath):
                os.remove(filepath)

    def on_epoch_end(self, epoch, logs=None):
        log_msg = '\t'.join(f'{k}={v}' for k, v in logs.items())
        msg = f'{epoch:4d}:   {log_msg}{self.line_ending}'
        with open(file=self.filepath, mode='a+') as file:
            file.write(msg)


class SaveModel(Callback):
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
            self.save_every_n_batches = save_every_n_batches
            self.on_batch_end = self._on_batch_end

    def _on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath, overwrite=self.overwrite_old_file)

    def _on_batch_end(self, batch, logs=None):
        if batch % self.save_every_n_batches == 0:
            self.model.save(self.filepath, overwrite=self.overwrite_old_file)

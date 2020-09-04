# Compressive Transformer in Keras
Keras implementation of the Compressive Transformer (with Multihead Attention) by Rae et. al <br>
_[Work in progress.]_


As specified in https://arxiv.org/pdf/1911.05507.pdf. <br>
(And further exemplified in https://deepmind.com/blog/article/A_new_model_and_dataset_for_long-range_memory.)

![Compressive Transformer Memory Visualization](https://lh3.googleusercontent.com/sGztWG_IU_PM_GNEYDOG99Hli3avAX0KJrEWLlosc5ZnMPEqdgpxnD3Z7s-rtcj9DeHhVfY2eyErzoP9mvPaQafdg4J70kl5b4kD=w1440-rw-v1)


## Installation:
```
make install
```

## Usage
```bash
python ct.py train
```

Runtime configurations - for tokenization, model options, etc. - can be configured in `ct/config/default.py`. _omegaconf_ is used for configuration.

## Instructions & Examples
A simple documentation of the code, together with some additional examples can be found in `docs/build/index.html`.

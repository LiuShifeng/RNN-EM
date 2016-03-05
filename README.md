# RNN-EM
Recurrent Neural Network with External Memory in Keras

* Paper : http://research.microsoft.com/pubs/246720/rnn_em.pdf

This is an implementation of a special kind of RNN which uses a 3-D external memory component to learn long range patterns in sequences. This is in contrast to LSTMs and GRUs which use 2-D hidden states. Though slower than LSTMs and GRUs, RNN-EMs can yield better results with lesser number of parameters.

* **API**

RNN-EM implements the [Recurrent](http://keras.io/layers/recurrent/) api in Keras. RNN-EM requires 2 additional arguments:
**nb_slots**: `int`. Number of memory slots.
**memory_size**: `int`. Size of each memory slot.

* **Example**

```python
from keras.models import Sequential

model = Sequential()
model.add(RNNEM(input_dim=10, output_dim=10, nb_slots=5, memory_size=10))
model.compile(loss='mse', optimizer='sgd')

```

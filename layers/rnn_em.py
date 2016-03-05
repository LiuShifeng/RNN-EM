from keras.layers.core import *
from keras.activations import *
from keras import backend as K
from theano import tensor as T
import numpy as np


class RNNEM(Recurrent):

	def __init__(self, output_dim, nb_slots, memory_size, **kwargs):
		self.output_dim = output_dim
		self.nb_slots = nb_slots
		self.memory_size = memory_size
		super(RNNEM, self).__init__(**kwargs)

	@property
	def output_shape(self):
		shape = list(self.input_shape)
		shape[-1] = self.output_dim
		if not self.return_sequences:
			shape.pop(1)
		return tuple(shape)

	def reset_states(self):
		nb_samples = self.input_shape[0]
		M = K.variable(np.zeros((nb_samples, self.nb_slots, self.memory_size)))
		h = K.variable(np.zeros((nb_samples, self.memory_size)))
		w = K.variable(np.zeros((nb_samples, self.nb_slots)))
		self.states = [M, h, w]

	def get_initial_states(self, x):
		M = K.zeros_like(x[:, 0, 0])  # (nb_samples,)
		M = K.pack([M] * self.nb_slots)  # (nb_slots, nb_samples)
		M = K.pack([M] * self.memory_size)  # (memory_size, nb_slots, nb_samples)
		M = K.permute_dimensions(M, (2, 1, 0))  # (nb_samples, nb_slots, memory_size)
		h = K.zeros_like(x[:, 0, 0])  # (nb_samples,)
		h = K.pack([h] * self.memory_size)  # (memory_size, nb_samples)
		h = K.permute_dimensions(h, (1, 0))  # (nb_samples, memory_size)
		w = K.zeros_like(x[:, 0, 0])  # (nb_samples,)
		w = K.pack([w] * self.nb_slots)  # (nb_slots, nb_samples)
		w = K.permute_dimensions(w, (1, 0))  # (nb_samples, nb_slots)
		states = [M, h, w]
		return states

	def build(self):
		self.states = [None, None, None]
		input_dim = self.input_shape[-1]
		output_dim = self.output_shape[-1]
		nb_slots = self.nb_slots
		memory_size = self.memory_size
		self.W_k = Dense(input_dim=memory_size, output_dim=memory_size)
		self.W_b = Dense(input_dim=memory_size, output_dim=1)
		self.W_hg = Dense(input_dim=memory_size, output_dim=1)
		self.W_ih = Dense(input_dim=input_dim, output_dim=memory_size)
		self.W_c = Dense(input_dim=memory_size, output_dim=memory_size)
		self.W_ho = Dense(input_dim=memory_size, output_dim=output_dim)
		self.W_v = Dense(input_dim=memory_size, output_dim=memory_size)
		self.W_he = Dense(input_dim=memory_size, output_dim=nb_slots)
		layers = [self.W_k, self.W_b, self.W_hg, self.W_ih, self.W_c, self.W_ho, self.W_v, self.W_he]
		weights = []
		for l in layers:
			weights += l.trainable_weights
		self.trainable_weights = weights

	def step(self, x, states):
		M = states[0]  # (nb_samples, nb_slots, memory_size)
		h = states[1]  # (nb_samples, memory_size)
		w = states[2]  # (nb_samples, nb_slots)
		#------Memory read--------#
		k = self.W_k(h)  # (nb_samples, memory_size)
		w_hat = T.batched_tensordot(M, k, axes=[(2), (1)])  # (nb_samples, nb_slots)
		beta = K.sigmoid(self.W_b(h))  # (nb_samples, 1)
		beta = K.repeat(beta, self.nb_slots)  # (nb_samples, nb_slots, 1)
		beta = K.squeeze(beta, 2)  # (nb_samples, nb_slots)
		w_hat = softmax(w_hat * beta)  # (nb_samples, nb_slots)
		g = sigmoid(self.W_hg(h))  # (nb_samples, 1)
		g = K.repeat(g, self.nb_slots)  # (nb_samples, nb_slots, 1)
		g = K.squeeze(g, 2)  # (nb_samples, nb_slots)
		w = (1 - g) * w + g * w_hat  # (nb_samples, nb_slots)
		c = T.batched_tensordot(w, M, axes=[(1), (1)])
		h = tanh(self.W_ih(x) + self.W_c(c))
		y = self.W_ho(h)
		#---------Memory write---------#
		v = self.W_v(h)  # (nb_samples, memory_size)
		v = K.repeat(v, 1)
		#v = K.reshape(v, (-1, 1, self.memory_size))  # (nb_samples, 1, memory_size)
		e = sigmoid(self.W_he(h))  # (nb_samples, nb_slots)
		f = 1 - w * e  # (nb_samples, nb_slots)
		f = K.repeat(f, self.memory_size)  # (nb_samples, memory_size, nb_slots)
		f = K.permute_dimensions(f, (0, 2, 1))  # (nb_samples, nb_slots, memory_size)
		u = w  # (nb_samples, nb_slots)
		u = K.repeat(u, 1)
		#u = K.reshape(-1, 1, self.nb_slots)  # (nb_samples, 1, nb_slots)
		uv = T.batched_tensordot(u, v, axes=[(1), (1)])
		M = M * f + uv
		return y, [M, h, w]

	def get_config(self):
		config = {'output_dim': self.output_dim, 'nb_slots': self.nb_slots, 'memory_size': self.memory_size}
        base_config = super(RNNEM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

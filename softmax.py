import numpy as np
import itertools
from layer import Layer
from utils import (
    debug_calculations, debug_logic
)
from scipy.special import softmax


class SoftmaxLayer(Layer):
    def __init__(self):
        self.inp = None
        self.num_inputs, self.num_labels = None, None
        self.output = None
        self.passback = None
    

    def forward(self, inp):
        self.inp = inp
        self.num_inputs, self.num_labels = self.inp.shape
        self.output = softmax(np.apply_along_axis(lambda x: x - np.amax(x), 1, self.inp), axis=1)
        debug_logic("SoftmaxLayer forward: output is: ", self.output)
    

    def backward(self, passback):
        assert passback.shape == self.output.shape
        if len(passback.shape) <= 1:
            passback = passback.reshape(1, *passback.shape)
        current_passback = None
        for x in range(self.num_inputs):
            current_scores_for_x = self.output[x, :].reshape(-1,)
            passback_for_x = passback[x, :].reshape(1, -1)
            jacobian = np.zeros((self.num_labels, self.num_labels))
            for i, j in itertools.product(range(self.num_labels), range(self.num_labels)):
                delta_fn = 1 if i==j else 0
                jacobian[i, j] = current_scores_for_x[i] * (delta_fn - current_scores_for_x[j])
            backprop_passback_for_x = np.matmul(passback_for_x, jacobian)
            if current_passback is None:
                current_passback = backprop_passback_for_x
            else:
                current_passback = np.vstack((
                    current_passback,
                    backprop_passback_for_x
                ))
        self.passback = current_passback
        debug_logic("SoftmaxLayer backward: passback is: ", self.passback)

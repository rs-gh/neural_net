import numpy as np
from layer import Layer
from utils import (
    debug_calculations, debug_logic
)
from scipy.special import expit


class SigmoidLayer(Layer):
    def __init__(self):
        self.inp = None
        self.num_inputs, self.num_labels = None, None
        self.output = None
        self.passback = None
    

    def forward(self, inp):
        self.inp = inp
        self.num_inputs, self.num_labels = self.inp.shape
        self.output = expit(self.inp)
        debug_logic("SigmoidLayer forward: output is: ", self.output)
    

    def backward(self, passback):
        if len(passback.shape) <= 1:
            passback = passback.reshape(1, *passback.shape)
        assert passback.shape == self.output.shape
        jacobians_for_all_inputs = np.multiply(expit(self.inp), np.ones(self.inp.shape) - expit(self.inp))
        passback = passback.reshape((self.num_inputs, self.num_labels))
        self.passback = np.multiply(jacobians_for_all_inputs, passback)
        debug_logic("SigmoidLayer backward: passback is: ", self.passback)

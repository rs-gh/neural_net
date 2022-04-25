import numpy as np
from layer import Layer
from utils import (
    debug_calculations, debug_logic,
    zero_pad
)


class FlattenLayer(Layer):
    def __init__(self):
        self.inp = None
        self.num_inputs, self.num_input_channels, self.input_height, self.input_width = None, None, None, None
        self.output = None
        self.passback = None


    def forward(self, inp):
        self.inp = inp
        self.num_inputs, self.num_input_channels, self.input_height, self.input_width = self.inp.shape
        current_output = None
        for i in range(self.num_inputs):
            x = self.inp[i, :, :, :].flatten()
            if current_output is None:
                current_output = x
            else:
                current_output = np.vstack((
                    current_output,
                    x
                ))
        self.output = current_output
        debug_logic("FlattenLayer forward: output is: ", self.output)


    def backward(self, passback):
        if len(passback.shape) <= 1:
            passback = passback.reshape(1, *passback.shape)
        assert self.num_inputs == passback.shape[0], "FlattenLayer backward: num_inputs does not match num passbacks"
        current_passback = None
        for i in range(self.num_inputs):
            passback_for_x = passback[i, :].reshape((1, self.num_input_channels, self.input_height, self.input_width))
            if current_passback is None:
                current_passback = passback_for_x
            else:
                current_passback = np.vstack((
                    current_passback,
                    passback_for_x
                ))
        self.passback = current_passback
        debug_logic("FlattenLayer backward: self.passback is: ", self.passback)

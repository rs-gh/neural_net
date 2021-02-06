import numpy as np
import itertools
from layer import Layer
from utils import (
    debug_calculations, debug_logic,
    zero_pad,
    get_conv_output_reshaped
)


class MaxpoolLayer(Layer):
    def __init__(self, kernel_size, zero_padding=False):
        self.inp = None
        self.num_inputs, self.num_input_channels, self.input_height, self.input_width = None, None, None, None
        self.max_mask = None
        self.output = None
        self.num_output_channels = None
        self.output_height = None
        self.output_width = None
        self.passback = None
        self.kernel_height, self.kernel_width = kernel_size
        self.zero_padding = zero_padding
        
    
    @staticmethod
    def _maxpool(inp, kernel_size, zero_padding=False):
        orig_input_channels, orig_input_height, orig_input_width = inp.shape
        if zero_padding:
            inp = zero_pad(inp, kernel_size)
        channels, input_height, input_width = inp.shape
        kernel_height, kernel_width = kernel_size
        output_height = int(input_height/kernel_height)
        output_width = int(input_width/kernel_width)
        output = np.zeros((channels, output_height, output_width))
        max_mask = np.zeros((orig_input_channels, orig_input_height, orig_input_width))
        for ch, h, w in itertools.product(
            range(channels), range(output_height), range(output_width)
        ):
            values = inp[
                ch,
                h*kernel_height:(h*kernel_height)+kernel_height,
                w*kernel_width:(w*kernel_width)+kernel_width
            ]
            max_kernel_indices = np.unravel_index(np.argmax(values), (kernel_height, kernel_width))
            if h*kernel_height+max_kernel_indices[0] < orig_input_height and w*kernel_width+max_kernel_indices[1] < orig_input_width:
                max_mask[
                    ch, 
                    h*kernel_height+max_kernel_indices[0],
                    w*kernel_width+max_kernel_indices[1]
                ] = 1
            output[ch, h, w] = np.amax(values)
        return output, max_mask

    
    def forward(self, inp):
        self.inp = get_conv_output_reshaped(inp)
        self.num_inputs, self.num_input_channels, self.input_height, self.input_width = self.inp.shape
        self.num_output_channels = self.num_input_channels
        current_output = None
        current_max_mask = None
        for i in range(self.num_inputs):
            output, max_mask = MaxpoolLayer._maxpool(
                self.inp[i, :, :, :].reshape((self.num_input_channels, self.input_height, self.input_width)),
                (self.kernel_height, self.kernel_width),
                self.zero_padding
            )
            output = output.reshape((1, *output.shape))
            max_mask = max_mask.reshape((1, *max_mask.shape))
            if current_output is None:
                current_output = output
            else:
                current_output = np.concatenate((
                    current_output,
                    output
                ), axis=0)
            if current_max_mask is None:
                current_max_mask = max_mask
            else:
                current_max_mask = np.concatenate((
                    current_max_mask,
                    max_mask
                ), axis=0)
        self.output = current_output
        self.max_mask = current_max_mask
        assert self.num_inputs == self.output.shape[0]
        self.output_height = self.output.shape[2]
        self.output_width = self.output.shape[3]
        debug_logic("MaxpoolLayer forward: output is: ", self.output)

    
    def backward(self, passback):
        if len(passback.shape) <= 1:
            passback = passback.reshape(1, *passback.shape)
        current_passback = np.zeros((self.num_inputs, self.num_input_channels, self.input_height, self.input_width))
        assert self.output.shape == passback.shape, "MaxpoolLayer backward: layer output shape does not match passbacks shape"
        for i, ch, h, w in itertools.product(
            range(self.num_inputs), range(self.num_input_channels), range(self.output_height), range(self.output_width)
        ):
            mask = self.max_mask[i, ch,
                h*self.kernel_height:(h+1)*self.kernel_height,
                w*self.kernel_width:(w+1)*self.kernel_width
            ]
            mask_indices = np.unravel_index(np.argmax(mask), (self.kernel_height, self.kernel_width))
            if h*self.kernel_height+mask_indices[0] < self.input_height and w*self.kernel_width+mask_indices[1] < self.input_width:
                current_passback[i, ch,
                    h*self.kernel_height+mask_indices[0],
                    w*self.kernel_width+mask_indices[1]
                ] = passback[i, ch, h, w]
        self.passback = current_passback
        debug_logic("MaxpoolLayer backward: self.passback is: ", self.passback)
    
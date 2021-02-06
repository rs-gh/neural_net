import numpy as np
import itertools
from layer import Layer
from utils import (
    debug_calculations, debug_logic,
    zero_pad, 
    get_conv_input_reshaped,
    get_conv_output_reshaped,
    get_kernel_reshaped
)


class ConvolutionalLayer(Layer):
    '''
        input_dimensions = num_inputs x num_input_channels x input_height x input_width
        output_dimensions = num_inputs x num_outputs (num_kernels) x num_output_channels x output_height x output_width
    '''
    def __init__(self, kernel_size, stride=1, zero_padding=False, num_output_channels=1, num_outputs=1, kernel_init_method="xavier", learning_rate=0.01):
        self.inp = None
        self.num_inputs, self.num_input_channels, self.input_height, self.input_width = None, None, None, None
        self.output = None
        self.output_height = None
        self.output_width = None
        self.kernels_grads = None
        self.biases_grads = None
        self.passback = None
        self.kernels, self.biases = None, None
        self.kernel_height, self.kernel_width = kernel_size
        self.stride = stride
        self.zero_padding = zero_padding
        self.num_output_channels = num_output_channels
        self.num_outputs = num_outputs
        self.kernel_init_method = kernel_init_method
        self.learning_rate = learning_rate

    
    def _init_kernels_and_biases(self):
        kernels = []
        biases = []
        for n in range(self.num_outputs):
            if self.kernel_init_method == "xavier":
                kernels.append(
                    np.random.multivariate_normal(
                        mean=[0]*self.kernel_width,
                        cov=[[0 if x!=i else 1/(self.kernel_height * self.kernel_width) for x in range(self.kernel_width)] for i in range(self.kernel_width)],
                        size=(self.num_input_channels, self.num_output_channels, self.kernel_height)
                    )
                )
            elif self.kernel_init_method == "test":
                kernels.append(
                    np.ones((self.num_input_channels, self.num_output_channels, self.kernel_height, self.kernel_width))
                )
            biases.append(
                np.random.normal(0, 1, 1)
            )
        return kernels, biases
    
    
    @staticmethod
    def _convolution(inp, kernel, bias, stride, zero_padding):
        kernel_inp_channels, num_output_channels, kernel_height, kernel_width = kernel.shape
        if zero_padding:
            inp = zero_pad(inp, (kernel_height, kernel_width))
        num_input_channels, input_height, input_width = inp.shape
        assert kernel_inp_channels == num_input_channels
        output_height = int((input_height-kernel_height)/stride) + 1
        output_width = int((input_width-kernel_width)/stride) + 1
        output = np.zeros((num_output_channels, output_height, output_width))
        for inp_ch, out_ch, h, w in itertools.product(
            range(num_input_channels), range(num_output_channels), range(output_height), range(output_width)
        ):
            output[out_ch, h, w] = np.sum(
                np.multiply(
                    inp[inp_ch, h*stride:(h*stride)+kernel_height, w*stride:(w*stride)+kernel_width],
                    kernel[inp_ch, out_ch, :, :]
                )
           ) + bias
        return output

    
    @staticmethod
    def _convolution_per_input(inp, kernels, biases, stride, zero_padding):
        output_maps = None
        for kernel_and_bias in zip(kernels, biases):
            kernel = get_kernel_reshaped(kernel_and_bias[0])
            bias = kernel_and_bias[1]
            conv_output = ConvolutionalLayer._convolution(inp, kernel, bias, stride, zero_padding)
            conv_output = conv_output.reshape((1, *conv_output.shape))
            if output_maps is None:
                output_maps = conv_output
            else:
                output_maps = np.vstack((
                    output_maps,
                    conv_output
                ))
        return output_maps

    
    def forward(self, inp):
        self.inp = get_conv_input_reshaped(inp)
        self.num_inputs, self.num_input_channels, self.input_height, self.input_width = self.inp.shape
        if self.kernels is None and self.biases is None:
            self.kernels, self.biases = self._init_kernels_and_biases()
            debug_calculations("ConvoluionalLayer: init kernels: {}\n init biases: {}".format(self.kernels, self.biases))
            assert len(self.kernels) == len(self.biases)
        current_output = None
        for i in range(self.num_inputs):
            output_maps = ConvolutionalLayer._convolution_per_input(
                self.inp[i, :, :, :].reshape((self.num_input_channels, self.input_height, self.input_width)),
                self.kernels, self.biases, self.stride, self.zero_padding
            )
            output_maps = output_maps.reshape(1, *output_maps.shape)
            if current_output is None:
                current_output = output_maps
            else:
                current_output = np.vstack((
                    current_output,
                    output_maps
                ))
        self.output = current_output
        assert self.num_inputs == self.output.shape[0]
        self.output_height = self.output.shape[3]
        self.output_width = self.output.shape[4]
        debug_logic("ConvolutionalLayer forward: output is: ", self.output)

    
    def backward(self, passbacks):
        assert len(passbacks) == self.num_outputs, "ConvolutionalLayer: number of passbacks is not equal to num_outputs per input/num kernels"
        debug_calculations("ConvolutionLayer backward: passbacks are: ", passbacks)
        self.kernels_grads = {}
        self.biases_grads = {}
        current_passback = np.zeros((self.num_inputs, self.num_input_channels, self.input_height, self.input_width))
        for kernel_idx, passback_for_kernel in enumerate(passbacks):
            assert passback_for_kernel.shape == (self.num_inputs, self.num_output_channels, self.output_height, self.output_width), "ConvolutionalLayer: passback shape is not equal to output shape"
            kernel = self.kernels[kernel_idx]
            bias = self.biases[kernel_idx]
            kernel_grad = np.zeros((self.num_input_channels, self.num_output_channels, self.kernel_height, self.kernel_width))
            bias_grad = 0
            for x in range(self.num_inputs):
                passback_for_x = passback_for_kernel[x, :, :, :].reshape(self.num_output_channels, self.output_height, self.output_width)
                inp_for_x = self.inp[x, :, :, :].reshape(self.num_input_channels, self.input_height, self.input_width)
                for inp_ch, out_ch in itertools.product(range(self.num_input_channels), range(self.num_output_channels)):
                    for i_hat, j_hat in itertools.product(range(self.input_height), range(self.input_width)):
                        for i_indices, j_indices in itertools.product(
                            ((i, m) for i in range(self.output_height) for m in range(self.kernel_height) if i * self.stride + m == i_hat),
                            ((j, n) for j in range(self.output_width) for n in range(self.kernel_width) if j * self.stride + n == j_hat)
                        ):
                            i, m, j, n = *i_indices, *j_indices
                            current_passback[x, inp_ch, i_hat, j_hat] += kernel[inp_ch, out_ch, m, n] * passback_for_x[out_ch, i, j]
                    for m, n in itertools.product(range(self.kernel_height), range(self.kernel_width)):
                        for h, w in itertools.product(range(self.output_height), range(self.output_width)):
                            debug_calculations("ConvolutiontalLayer backward: passback_for_x[out_ch, h, w] is: ", passback_for_x[out_ch, h, w] if h*self.stride+m < self.input_height and w*self.stride+n < self.input_width else 0)
                            debug_calculations("ConvolutiontalLayer backward: inp_for_x[inp_ch, h*self.stride+m, w*self.stride+n] is: ", inp_for_x[inp_ch, h*self.stride+m, w*self.stride+n] if h*self.stride+m < self.input_height and w*self.stride+n < self.input_width else 0)
                            kernel_grad[inp_ch, out_ch, m, n] += passback_for_x[out_ch, h, w] * inp_for_x[inp_ch, h*self.stride+m, w*self.stride+n] if h*self.stride+m < self.input_height and w*self.stride+n < self.input_width else 0                            
                            bias_grad += passback_for_x[out_ch, h, w]
            debug_calculations("ConvolutionalLayer backward: kernel_grad is: ", kernel_grad)
            current_passback /= self.num_inputs
            try:
                self.kernels_grads[kernel_idx] += kernel_grad/self.num_inputs
                self.biases_grads[kernel_idx] += bias_grad/self.num_inputs
            except KeyError:
                self.kernels_grads[kernel_idx] = kernel_grad/self.num_inputs
                self.biases_grads[kernel_idx] = bias_grad/self.num_inputs

        current_passback /= self.num_outputs
        self.passback = current_passback
        debug_logic("ConvolutionalLayer backward: self.passback is: ", self.passback)

        for i in range(self.num_outputs):
            self.kernels_grads[i] /= self.num_outputs
            self.kernels[i] -= self.learning_rate * self.kernels_grads[i]
            self.biases_grads[i] /= self.num_outputs
            self.biases[i] -= self.learning_rate * self.biases_grads[i]

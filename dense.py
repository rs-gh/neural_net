import numpy as np
from layer import Layer
from utils import (
    debug_calculations, debug_logic,
    zero_pad, 
    get_dense_input_reshaped
)


class DenseLayer(Layer):
    def __init__(self, num_neurons, weight_init_method="xavier", learning_rate=0.01):
        self.inp = None
        self.num_inputs, self.input_dim = None, None
        self.output = None
        self.passback = None
        self.weights, self.biases = None, None
        self.num_neurons = num_neurons
        self.weight_init_method = weight_init_method
        self.learning_rate = learning_rate


    def _init_weights_and_biases(self):
        if self.weight_init_method == "xavier":
            weights = np.random.multivariate_normal(
                mean=[0]*self.num_neurons,
                cov=[[0 if x!=i else 1/self.input_dim for x in range(self.num_neurons)] for i in range(self.num_neurons)],
                size=(self.input_dim,)
            )
        elif self.weight_init_method == "test":
            weights = np.ones((self.input_dim, self.num_neurons))
        biases = np.random.multivariate_normal(
            mean=[0]*self.num_neurons,
            cov=[[0 if x!=i else 1 for x in range(self.num_neurons)] for i in range(self.num_neurons)],
            size=(1,)
        )
        assert weights.shape[0] == self.input_dim, weights.shape[1] == biases.shape[1] == self.num_neurons
        return weights, biases
    
    
    def forward(self, inp):
        self.inp = get_dense_input_reshaped(inp)
        self.num_inputs, self.input_dim = self.inp.shape
        if self.weights is None and self.biases is None:
            self.weights, self.biases = self._init_weights_and_biases()
            debug_logic("DenseLayer: init weights: {}\n init biases: {}".format(self.weights, self.biases))
        x = np.hstack((
            np.ones((self.num_inputs,)).reshape(self.num_inputs, 1),
            self.inp
        ))
        weights_and_biases = np.vstack((
            self.biases,
            self.weights
        ))
        self.output = np.matmul(
            x,
            weights_and_biases
        )
        debug_logic("DenseLayer forward: output is: ", self.output)


    def backward(self, passback):
        if len(passback.shape) <= 1:
            passback = passback.reshape(1, *passback.shape)
        assert self.num_inputs == passback.shape[0], "DenseLayer backward: num_inputs does not match num passbacks"
        weights_grad = None
        current_passback = None
        for i in range(self.num_inputs):
            x = self.inp[i, :].reshape((self.input_dim, 1))
            passback_for_x = passback[i, :].reshape((self.num_neurons,))
            weights_grad_for_x = np.hstack([np.array(passback_for_x[m] * x) for m in range(self.num_neurons)])
            backprop_passback_for_x = np.matmul(
                self.weights,
                passback_for_x
            )
            if weights_grad is None:
                weights_grad = weights_grad_for_x
            else:
                weights_grad += weights_grad_for_x
            if current_passback is None:
                current_passback = backprop_passback_for_x
            else:
                current_passback = np.vstack((
                    current_passback,
                    backprop_passback_for_x
                ))
        weights_grad = weights_grad / self.num_inputs
        bias_grad = np.apply_along_axis(np.mean, axis=0, arr=passback)
        self.weights -= self.learning_rate * weights_grad
        self.biases -= self.learning_rate * bias_grad
        self.passback = current_passback
        debug_logic("DenseLayer backward: self.passback is: ", self.passback)

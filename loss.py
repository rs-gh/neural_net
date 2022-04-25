import numpy as np
from layer import Layer
from utils import (
    debug_calculations, debug_logic
)


class LossLayer(Layer):
    def __init__(self, loss_method="categorical_crossentropy"):
        self.scores = None
        self.labels = None
        self.num_inputs, self.num_labels = None, None
        self.loss = None
        self.passback = None
        self.loss_history = []
        self.loss_method = loss_method
        if loss_method == "categorical_crossentropy":
            self.loss_fn = LossLayer._categorical_crossentropy
            self.loss_fn_grad = LossLayer._categorical_crossentropy_grad


    @staticmethod
    def _categorical_crossentropy(scores, labels):
        loss = np.inner(labels, -np.log(scores))
        if not np.isfinite(loss):
            return 10
        return loss
    

    @staticmethod
    def _categorical_crossentropy_grad(scores, labels):
        return np.multiply(labels, -1.0/scores)
    

    def forward(self, scores, labels):
        if len(scores.shape) < 2:
            scores = scores.reshape(1, *scores.shape)
        if len(labels.shape) < 2:
            labels = labels.reshape(1, *labels.shape)
        self.labels = labels
        self.scores = np.where(scores == 0, 0.001, scores)
        self.num_inputs, self.num_labels = scores.shape
        current_loss = None
        for x in range(self.num_inputs):
            loss_for_x = self.loss_fn(self.scores[x, :], self.labels[x, :])
            if current_loss is None:
                current_loss = loss_for_x
            else:
                current_loss = np.vstack((
                    current_loss,
                    loss_for_x
                ))
        self.loss = current_loss
        self.loss_history.append(current_loss)
        debug_logic("LossLayer forward: loss is: ", self.loss)
    

    def backward(self):
        current_passback = None
        for x in range(self.num_inputs):
            passback_for_x = self.loss_fn_grad(self.scores[x, :], self.labels[x, :])
            if current_passback is None:
                current_passback = passback_for_x
            else:
                current_passback = np.vstack((
                    current_passback,
                    passback_for_x
                ))
        self.passback = current_passback
        debug_logic("LossLayer backward: passback is: ", self.passback)

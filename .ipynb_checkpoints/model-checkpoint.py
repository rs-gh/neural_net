import numpy as np
from utils import generate_batch, verbose_print


class Model:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        num_epochs
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_samples = X_train.shape[0]
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = None
        self.reverse_model = None
    

    def forward(self, inp, labels):
        for idx, model_layer in enumerate(self.model):
            desc, layer = model_layer
            layer_inp = inp if "input" in desc else self.model[idx-1][1].output
            if "loss" in desc:
                layer.forward(layer_inp, labels=labels)
            else:
                layer.forward(layer_inp)


    def backward(self):
        for idx, model_layer in enumerate(self.reverse_model):
            desc, layer = model_layer
            if "loss" in desc:
                layer.backward()
            else:
                layer_pb = [self.reverse_model[idx-1][1].passback] if "multi_output" in desc else self.reverse_model[idx-1][1].passback
                layer.backward(layer_pb)
    

    def train(self, verbose=True):
        for epoch in range(self.num_epochs):
            verbose_print("Training epoch {} of {}.".format(epoch+1, self.num_epochs))
            for batch_start, batch_inp, batch_labels in generate_batch(
                self.X_train, self.y_train, self.num_samples, self.batch_size
            ):
                verbose_print("Training batch of inputs between {} and {}.".format(batch_start, batch_start+self.batch_size), verbose=verbose)
                self.forward(batch_inp, batch_labels)
                verbose_print("Loss: ", np.mean(self.loss.loss), verbose=verbose)
                self.backward()


    def test(self, verbose=True):
        y_true = []
        y_pred = []
        for image, label in list(zip(self.X_test, self.y_test)):
            try:
                num_input, num_channels, height, width = image.shape
            except ValueError:
                image = image.reshape((1, 1, *image.shape))
            y_true.append(np.argmax(label))
            self.forward(image, label)
            y_pred.append(np.argmax(self.loss.scores))
        return y_true, y_pred
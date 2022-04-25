import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def plot_loss_history(model):
    loss_history = [np.mean(model.loss.loss_history[i]) for i in range(len(model.loss.loss_history))]
    plt.plot(range(len(loss_history)), loss_history)


def plot_confusion_matrix(y_true, y_pred):
    confusion_matrix_images = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(10))
    print(confusion_matrix_images)
    plt.matshow(confusion_matrix_images)
    print(classification_report(y_true=y_true, y_pred=y_pred))

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_loss(train_err, val_err, save_dir, filename):
    x1 = np.arange(0, len(val_err))
    x2 = np.arange(0, len(train_err))
    # x2 = x2 / 437.5
    plt.plot(x2, train_err, 'b')
    plt.plot(x1, val_err, 'b')
    plt.xlabel('epochs')
    plt.ylabel('MSE LOSS')
    plt.title('MSE error of Zer')
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()

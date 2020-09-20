import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):     
    plt.plot(history.history['loss'],label='Train')
    plt.plot(history.history['val_loss'],label='Develop')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim((0,1.5*np.max(history.history['val_loss'])))
    plt.legend()
    plt.show()
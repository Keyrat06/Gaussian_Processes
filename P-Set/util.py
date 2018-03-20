import numpy as np
import matplotlib.pyplot as plt

def get_low_and_high(y_bar, var):
    sigma = np.sqrt(var)
    y_low = y_bar - 1.96 * sigma
    y_high = y_bar + 1.96 * sigma
    return y_low, y_high


def pretty_plot(fig, axs, xlim=(-15,15), ylim=(-3,3), size=(16,8)):
    plt.ylim(ylim)
    plt.xlim(xlim)
    fig.set_size_inches(size)
    plt.show()
    

def get_sample_data_1():
    x = np.linspace(-5, 5, 15)
    y = 2*np.sin(x) + x
    return x, y

def scatter_raw_data(x, y):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(x, y)
    axs.set_ylabel("Y")
    axs.set_xlabel("X")
    axs.set_title("Sample Data")
    pretty_plot(fig, axs, xlim=(min(x)-1, max(x)+1), ylim=(min(y)-1, max(y)+1))
    
def visiualize_kernal(K):
    fig, axs = plt.subplots(1, 1)
    axs.imshow(K)
    fig.set_size_inches((5,5))
    plt.show()
    
def visiualize_Ks(K, KS, KSS):
    T = np.hstack((K, KS.T))
    B = np.hstack((KS, KSS))
    new_K = np.vstack((T,B))
    visiualize_kernal(new_K)
    
def get_sample_classification_data():
    x_1 = np.random.choice(np.linspace(-11, -9, 100), 5, replace=False)
    x_2 = np.random.choice(np.linspace(-1, 1, 100), 5, replace=False)
    x_3 = np.random.choice(np.linspace(9, 11, 100), 5, replace=False)
    x = np.concatenate((x_1, x_2, x_3), axis=0)
    y = -1 * np.ones(len(x))
    return x, y


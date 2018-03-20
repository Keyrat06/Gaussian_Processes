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
    S = 40
    x = np.array(sorted(np.random.choice(np.linspace(-5, 5, 100), S, replace=False)))
    y = np.sin(np.pi * x) + 2 * np.sin(np.pi/5 * x) + np.random.normal(0, 0.1, S)
    return x, y

def scatter_raw_data(x, y, sigma_n=0.1):
    fig, axs = plt.subplots(1, 1)
    for i, _ in enumerate(x):
        axs.errorbar(x[i], y[i], linewidth=2, marker='o', ms=3, capsize=2, yerr=2*sigma_n, c='b', zorder=3)
    axs.set_ylabel("Y")
    axs.set_xlabel("X")
    axs.set_title("Sample Data")
    pretty_plot(fig, axs, xlim=(min(x)-1, max(x)+1), ylim=(min(y)-1, max(y)+1))
    
def visiualize_kernel(K):
    fig, axs = plt.subplots(1, 1)
    axs.imshow(K)
    fig.set_size_inches((5,5))
    plt.axis('off')
    plt.show()
    
def visiualize_Ks(K, KS, KSS):
    T = np.hstack((K, KS.T))
    B = np.hstack((KS, KSS))
    new_K = np.vstack((T,B))
    visiualize_kernel(new_K)
    
def solve_and_visualize(regression_GP, kernel, x, y, theta, x_range=None, y_range=None):
    if x_range is None: x_range = (min(x)-1, max(x)+1)
    if y_range is None: y_range = (min(y)-1, max(y)+1)
    x_new = np.linspace(x_range[0], x_range[1], 100)
    y_bar, var = regression_GP(x_new, x, y, kernel, theta)
    y_low, y_high = get_low_and_high(y_bar, var)
    
    fig, axs = plt.subplots(1, 1)
    axs.fill_between(x_new, y_low, y_high, facecolor='r', alpha=0.5,zorder=1)
    axs.plot(x_new,y_bar,c="k",zorder=2)
    for i, _ in enumerate(x):
        axs.errorbar(x[i], y[i], linewidth=2, marker='o', ms=3, capsize=2, yerr=2*theta[-1], c='b', zorder=3)
        
    pretty_plot(fig, axs, xlim=x_range, ylim=y_range)

    
def get_sample_classification_data():
    x_1 = np.random.choice(np.linspace(-10, -8, 20), 5, replace=False)
    x_2 = np.random.choice(np.linspace(-1, 1, 20), 5, replace=False)
    x_3 = np.random.choice(np.linspace(8, 10, 20), 5, replace=False)
    x = np.concatenate((x_1, x_2, x_3), axis=0)
    y = -1 * np.ones(len(x))
    y[np.where(abs(x)<2)] = 1
    return x, y

def scatter_raw_data(x, y, sigma_n=0.1):
    fig, axs = plt.subplots(1, 1)
    axs.set_ylabel("Y")
    axs.set_xlabel("X")
    axs.set_title("Sample Data")
    for i in range(len(x)):
        if y[i] > 0:
            axs.scatter(x[i], y[i], 50, marker='+', color='g')
        else:
            axs.scatter(x[i], y[i], 50, marker='o', color='r')
    def pretty_plot_classification(fig, axs, xlim=(-15,15), ylim=(-3,3), size=(16,8)):
        plt.ylim(ylim)
        plt.xlim(xlim)
        fig.set_size_inches(size)        
        plt.show()
    



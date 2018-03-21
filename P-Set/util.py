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
    im = axs.imshow(K)
    fig.set_size_inches((5,5))
    plt.axis('off')
    plt.colorbar(im)
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
    x_1 = np.random.choice(np.linspace(-10, -5, 20), 5, replace=False)
    x_2 = np.random.choice(np.linspace(-2.5, 2.5, 20), 5, replace=False)
    x_3 = np.random.choice(np.linspace(5, 10, 20), 5, replace=False)
    x = np.concatenate((x_1, x_2, x_3), axis=0)
    y = -1 * np.ones(len(x))
    y[np.where(abs(x)<3)] = 1
    return x, y

def scatter_raw_data_classification(x, y, y_label = "Y", x_label = "X", title = "Sample Data", sigma_n=0.1):
    fig, axs = plt.subplots(1, 1)
    axs.set_ylabel(y_label)
    axs.set_xlabel(x_label)
    axs.set_title(title)
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
    pretty_plot_classification(fig, axs)
    
def temperature_example(regression_GP, optimizer, kernal, params_0):
    np.random.seed(0)
    sample_std = 5.0
    num_years = 2
    num_years_out = 1
    x = np.array(np.random.random(80 * num_years)  * 365 * num_years )
    y = - 20 * np.cos(np.pi/365*2 * x) + np.random.normal(0,sample_std, len(x))
    y_range = (5,95)
    x_samples = np.linspace(0, 365*(num_years+num_years_out), 365 * (num_years+num_years_out))
    theta = optimizer(x, y, sample_std, kernal, params_0)
    y_samples, var = regression_GP(x_samples, x, y, kernal, theta)
    y += 50; y_samples += 50
    y_low, y_high = get_low_and_high(y_samples, var)
    
    
    fig, ax = plt.subplots()
    ax.fill_between(x_samples, y_low, y_high, facecolor='r', alpha=0.5,zorder=1)
    ax.plot(x_samples,y_samples,c="k",zorder=2)
    for i, _ in enumerate(x):
        ax.errorbar(x[i], y[i], linewidth=0, marker='o', ms=5, capsize=0, yerr=0.25, c='b', zorder=3)
    ax.set_ylabel("Temperature")
    ax.set_xlabel("Date")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    years = [2016 + i for i in range(num_years+num_years_out)]
    every_other = 3
    i = 0
    dates = []
    for year in years:
        for month in months:
            if i % every_other == 0:
                dates.append("{} {}".format(month, year))
            i += 1
    ax.set_xticks(np.linspace(0,365*(num_years+num_years_out),12*(num_years+num_years_out) / every_other))
    ax.set_xticklabels(dates)
    plt.xlim((0,365*(num_years+num_years_out)))
    plt.ylim(y_range)
    fig.set_size_inches((20,8))
    plt.show()
        
        
def draw_sigmoid(sigmoid_function):    
    x = np.linspace(-10, 10, 100)
    y = sigmoid_function(x)
    fig, axs = plt.subplots(1, 1)
    axs.plot(x, y, 'r')
    axs.set_ylabel("Sigmoid(X)")
    axs.set_xlabel("X")
    pretty_plot(fig, axs, xlim=(-10,10), ylim=(0,1))
    
def sigmoid(x):
    return 1./(1+np.exp(-x))

def calculate_W(f, y):
    n = len(y)
    W = np.zeros(n)
    for j in range(n):
        sigmoid_v = sigmoid(f[j]*y[j])
        W[j] = y[j]**2 * (1-sigmoid_v)*sigmoid_v
    return W

def calculate_KP(K, W):
    return K + (1.0/W)

def pretty_plot_classify(fig, axs, xlim=(-20,20), ylim=(-1.5,1.5), size=(10,10)):
    plt.ylim(ylim)
    plt.xlim(xlim)
    fig.set_size_inches(size)
    plt.show()

def draw_latent_function(GPC, optimize_theta, kernel, x_new, x, y):
    theta = optimize_theta(x, y, kernel, params_0=[0.4, 5], sigma_n=0.0)
    f_bar, var = GPC(x_new, x, y, kernel, theta)

    fig, axs = plt.subplots(1, 1)
    for i in range(len(x)):
        if y[i] > 0:
            axs.scatter(x[i], y[i], 80, marker='+', color='g')
        else:
            axs.scatter(x[i], y[i], 80, marker='o', color='r')
    axs.plot(x_new, f_bar,  color='k')
    pretty_plot_classify(fig, axs)
    return f_bar, var

def draw_predictive_probabilities(f_bar, x_new, x, y):
    prob = np.zeros(len(f_bar))
    for i in range(len(f_bar)):
        prob[i] = sigmoid(f_bar[i])

    fig, axs = plt.subplots(1, 1)
    for i in range(len(x)):
        if y[i] > 0:
            axs.scatter(x[i], y[i], 80, marker='+', color='g')
        else:
            axs.scatter(x[i], y[i], 80, marker='o', color='r')
    axs.plot(x_new, prob,  color='k')
    pretty_plot_classify(fig, axs)
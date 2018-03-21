import util
import numpy as np
import math
from scipy.optimize import minimize

def test_simple_kernel(kernel):
    ## Test 1 ##
    x = np.array([1])
    K = kernel(x, x, [1, 1], 0.1)
    target = np.array([[ 1.01]])
    assert np.allclose(K, target), "ran kernel(np.array([1]), np.array([1]), [1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 2 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([2.5])
    K = kernel(x0, x1, [1, 1], 0.1)
    target = np.array([[ 0.32465247], [ 0.8824969 ], [ 0.8824969 ], [ 0.32465247], [ 0.04393693]])
    assert np.allclose(K, target), "ran kernel(np.array([1,2,3,4,5]), np.array([2.5]), [1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 3 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([1,2,3,4,5])
    K = kernel(x0, x1, [1, 0.5], 0.2)
    target = np.array([[1.04000000e+00, 1.35335283e-01, 3.35462628e-04, 1.52299797e-08, 1.26641655e-14],
                       [1.35335283e-01, 1.04000000e+00, 1.35335283e-01, 3.35462628e-04, 1.52299797e-08],
                       [3.35462628e-04, 1.35335283e-01, 1.04000000e+00, 1.35335283e-01, 3.35462628e-04],
                       [1.52299797e-08, 3.35462628e-04, 1.35335283e-01, 1.04000000e+00, 1.35335283e-01],
                       [1.26641655e-14, 1.52299797e-08, 3.35462628e-04, 1.35335283e-01, 1.04000000e+00]])
    assert np.allclose(K, target), "ran kernel(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), [1, 0.5], 0.2) got {}, wanted {}".format(K, target)    
                      
                      
def test_periodic_kernel(kernel):
    ## Test 1 ##
    x = np.array([1])
    K = kernel(x, x, [1, 1, 1, 1], 0.1)
    target = np.array([[2.01]])
    assert np.allclose(K, target), "ran periodic_kernel(np.array([1]), np.array([1]), [1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 2 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([2.5])
    K = kernel(x0, x1, [1, 1, 1, 1], 0.1)
    target = np.array([[0.46134892], [1.51397142], [1.51397142], [0.46134892], [0.53247503]])
    assert np.allclose(K, target), "ran periodic_kernel(np.array([1,2,3,4,5]), np.array([2.5]), [1, 1, 1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 3 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([1,2,3,4,5])
    K = kernel(x0, x1, [1, 0.5, 1, 0.5], 0.2)
    target = np.array([[2.04, 0.7668098, 0.24298264, 0.13669647, 0.19135142],
                       [0.7668098, 2.04, 0.7668098, 0.24298264, 0.13669647],
                       [0.24298264, 0.7668098, 2.04, 0.7668098, 0.24298264],
                       [ 0.13669647, 0.24298264, 0.7668098, 2.04, 0.7668098],
                       [ 0.19135142, 0.13669647, 0.24298264, 0.7668098, 2.04]])
    assert np.allclose(K, target), "ran kernel(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), [1, 0.5, 1, 0.5], 0.2) got {}, wanted {}".format(K, target)
    
    
def test_get_Ks(get_Ks, kernel):
    ## Test 1 ##
    x = np.array([1])
    x1 = np.array([1,2,3,4,5])
    theta = [1, 1, 0.1]
    K, KS, KSS = get_Ks(x, x1, kernel, theta)
    target_K = np.array([[1.01000000e+00, 6.06530660e-01, 1.35335283e-01, 1.11089965e-02, 3.35462628e-04],
                         [6.06530660e-01, 1.01000000e+00, 6.06530660e-01, 1.35335283e-01, 1.11089965e-02],
                         [1.35335283e-01, 6.06530660e-01, 1.01000000e+00, 6.06530660e-01, 1.35335283e-01],
                         [1.11089965e-02, 1.35335283e-01, 6.06530660e-01, 1.01000000e+00, 6.06530660e-01],
                         [3.35462628e-04, 1.11089965e-02, 1.35335283e-01, 6.06530660e-01, 1.01000000e+00]])
    target_KS = np.array([[1.01000000e+00, 6.06530660e-01, 1.35335283e-01, 1.11089965e-02, 3.35462628e-04]])
    target_KSS = np.array([[ 1.01]])
    assert np.allclose(KSS, target_KSS)
    assert np.allclose(KS, target_KS)
    assert np.allclose(K, target_K)
    
def test_regression(regression_GP, kernel):
    ### test 1 ###
    x = np.array([1])
    y = np.sin(x)
    theta = [1, 0.5, 0.1]
    x_samples = np.array([-1, 0, 0.5, 1, 1.5])
    y_samples, var = regression_GP(x_samples, x, y, kernel, theta)

    target_y_samples = np.array([2.79487196e-04, 1.12753182e-01, 5.05324704e-01, 8.41470985e-01, 5.05324704e-01])
    target_var = np.array([1.00999989, 0.9918657, 0.64576293, 0., 0.64576293])
    assert np.allclose(y_samples, target_y_samples), "wanted {}, got {}".format(target_y_samples, y_samples)
    assert np.allclose(var, target_var), "wanted {}, got {}".format(target_var, var)
    
    
    ### test 2 ###
    x = np.linspace(-1, 1, 5)
    y = np.sin(x)
    theta = [1, 0.5, 0.1]
    x_samples = np.array([-1.25, -1, 0, 0.5, 1, 1.5])
    y_samples, var = regression_GP(x_samples, x, y, kernel, theta)
    target_y_samples = np.array([-7.41809101e-01, -8.41470985e-01, -3.91615754e-16, 4.79425539e-01, 8.41470985e-01, 5.12639003e-01])
    target_var = np.array([1.52675208e-01, 0.00000000e+00, 1.11022302e-15, 4.44089210e-16, 0.00000000e+00, 5.30945273e-01])
    assert np.allclose(y_samples, target_y_samples), "wanted {}, got {}".format(target_y_samples, y_samples)
    assert np.allclose(var, target_var), "wanted {}, got {}".format(target_var, var)    
    
def test_regression_ouptimize_theta(regression_optimize_theta, kernel):
    ### test 1 ###
    x = np.linspace(-5, 5, 10)
    y = np.sin(x)
    sigma_n = 0.1
    theta = np.array(regression_optimize_theta(x, y, sigma_n, kernel, params_0=[0.1, 0.1]))
    target = np.array([0.94208596248476884, -1.833617068532752, 0.1])
    assert np.allclose(theta, target), "wanted {}, got {}".format(target, theta)
    
    ### test 2 ###
    x = np.linspace(-20, 20, 50)
    y = np.sin(x) + np.sin(x/5)
    sigma_n = 0.1
    theta = np.array(regression_optimize_theta(x, y, sigma_n, kernel, params_0=[0.1, 0.1]))
    target = np.array([1.39545862,  2.21679695, 0.1])

    assert np.allclose(theta, target), "wanted {}, got {}".format(target, theta)

x_1 = np.random.choice(np.linspace(-10, -5, 20), 5, replace=False)
x_2 = np.random.choice(np.linspace(-2.5, 2.5, 20), 5, replace=False)
x_3 = np.random.choice(np.linspace(5, 10, 20), 5, replace=False)
x_new = np.linspace(-15, 15, 50)
x_data = np.concatenate((x_1, x_2, x_3), axis=0)
y_data = -1 * np.ones(len(x_data))
y_data[np.where(abs(x_data)<3)] = 1 
theta_c = [0.4, 5, 0]

def target_kernel(x0, x1, params, sigma_n):
    diff = np.subtract.outer(x0, x1)
    value = params[0]**2 * np.exp( -0.5 * (1.0/params[1]**2) * diff**2)
    value[np.where(diff == 0.0)] += sigma_n**2
    return value

def target_get_Ks(x_new, x, kernel, theta):
    K = kernel(x, x, theta[:-1], theta[-1]) # K
    KS = kernel(x_new, x, theta[:-1], theta[-1]) # K*
    KSS = kernel(x_new, x_new, theta[:-1], theta[-1]) # K**
    return K, KS, KSS

K_target, KS_target, KSS_target = target_get_Ks(x_new, x_data, target_kernel, theta_c)

def target_sigmoid(x):
    return 1./(1+np.exp(-x))
    
def test_sigmoid(sigmoid):
    x = np.linspace(-5, -5, 100)
    target = [target_sigmoid(x_i) for x_i in x]
    actual = [sigmoid(x_i) for x_i in x]
    assert np.allclose(actual, target)
    x = np.linspace(-50, -5, 100)
    target = [target_sigmoid(x_i) for x_i in x]
    actual = [sigmoid(x_i) for x_i in x]
    assert np.allclose(actual, target)
    x = np.linspace(5, 50, 100)
    target = [target_sigmoid(x_i) for x_i in x]
    actual = [sigmoid(x_i) for x_i in x]
    assert np.allclose(actual, target)
    
def target_find_f(K, y):
    n = len(y) 
    f = np.zeros(n)  
    y_giv_f = np.zeros(n)
    grad = np.zeros(n)
    
    for i in range(0, 100):
        for j in range(n):
            y_giv_f[j] = target_sigmoid(f[j]*y[j])
            grad[j] = (1-y_giv_f[j])*y[j]
        f = np.array(np.matmul(K, grad)).flatten()
    for j in range(n):
        y_giv_f[j] = target_sigmoid(f[j]*y[j])
    return f, y_giv_f
    
def test_find_f(find_f, get_Ks, kernel):
    target_f, target_y_giv_f = target_find_f(K_target, y_data)
    K, KS, KSS = get_Ks(x_new, x_data, kernel, theta_c)
    actual_f, actual_y_giv_f = find_f(K, y_data)
    assert np.allclose(K_target, K)
    assert np.allclose(target_f, actual_f)
    assert np.allclose(target_y_giv_f, actual_y_giv_f)
    return actual_f, actual_y_giv_f
    
def target_W(f, y):
    n = len(y)
    W = np.zeros(n)
    for j in range(n):
        sigmoid_v = target_sigmoid(f[j]*y[j])
        W[j] = y[j]**2 * (1-sigmoid_v)*sigmoid_v
    return W
    
def test_calc_W(calc_W, find_f, get_Ks, kernel):
    target_f, target_y_giv_f = target_find_f(K_target, y_data)
    K, KS, KSS = get_Ks(x_new, x_data, kernel, theta_c)
    actual_f, actual_y_giv_f = find_f(K, y_data)
    target = target_W(target_f, y_data)
    actual = calc_W(actual_f, y_data)
    assert np.allclose(actual, target)
    return actual

def target_KP(K, W):
    return K + (1.0/W)

def test_calc_KP(calculate_KP, calc_W, find_f, get_Ks, kernel):
    target_f, target_y_giv_f = target_find_f(K_target, y_data)
    K, KS, KSS = get_Ks(x_new, x_data, kernel, theta_c)
    actual_f, actual_y_giv_f = find_f(K, y_data)
    W_target = target_W(target_f, y_data)
    W_actual = calc_W(actual_f, y_data)
    actual = calculate_KP(K, W_actual)
    target = target_KP(K_target, W_target)    
    assert np.allclose(actual, target)
    return actual

def target_GPC(x_new, x, y, kernel, theta):
    K = kernel(x, x, theta[:-1], theta[-1]) # K
    KS = kernel(x_new, x, theta[:-1], theta[-1]) # K*
    KSS = kernel(x_new, x_new, theta[:-1], theta[-1]) # K**
    
    f, y_giv_f = target_find_f(K, y)
    W = target_W(f, y)
    
    KP = target_KP(K, W)

    f_bar = np.matmul(np.matmul(KS, np.linalg.inv(K)), f)    
    var = KSS - KS.dot(np.linalg.inv(KP).dot(KS.T))
    var = np.diagonal(var)
    return(f_bar.squeeze(), var.squeeze())
    
def test_GPC(GPC, calculate_KP, calc_W, find_f, get_Ks, kernel):
    target_f, target_var = target_GPC(x_new, x_data, y_data, target_kernel, theta_c)
    actual_f, actual_var = GPC(x_new, x_data, y_data, kernel, theta_c)
    assert np.allclose(actual_f, target_f)
    assert np.allclose(actual_var, target_var)
    return actual_f, actual_var

def target_optimize_theta(x, y, kernel, params_0=[0.1, 0.1], sigma_n=0.1):
    def log_pY(theta):
        K = np.matrix(target_kernel(x, x, theta, sigma_n))
        f, y_giv_f = target_find_f(K, y)
        W = target_W(f, y)
        inv_k = np.linalg.inv(K)
        log_k = np.log(np.linalg.det(K) * np.linalg.det(inv_k+W))
        Y_giv_f = np.prod(y_giv_f)
        output = 0.5 * np.matmul(np.matmul(f.T, inv_k),f)
        output += 0.5 * log_k
        output -= np.log(Y_giv_f)
        return output

    res = minimize(log_pY, params_0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
    return list(res.x) + [sigma_n]
    
def test_optimize_theta(optimize_theta, kernel):
    actual = optimize_theta(x_data, y_data, kernel)
    target = target_optimize_theta(x_data, y_data, target_kernel)
    assert np.allclose(actual, target)
    return actual
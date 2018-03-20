import util
import numpy as np
import math

def test_simple_kernal(kernal):
    ## Test 1 ##
    x = np.array([1])
    K = kernel(x, x, [1, 1], 0.1)
    target = np.array([[ 1.01]])
    assert np.allclose(K, target), "ran kernel(np.array([1]), np.array([1]), [1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 2 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([2.5])
    K = kernal(x0, x1, [1, 1], 0.1)
    target = np.array([[ 0.32465247], [ 0.8824969 ], [ 0.8824969 ], [ 0.32465247], [ 0.04393693]])
    assert np.allclose(K, target), "ran kernal(np.array([1,2,3,4,5]), np.array([2.5]), [1, 1], 0.1) got {}, wanted {}".format(K, target)
    
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
                      
                      
def test_periodic_kernal(kernal):
    ## Test 1 ##
    x = np.array([1])
    K = kernal(x, x, [1, 1, 1, 1], 0.1)
    target = np.array([[2.01]])
    assert np.allclose(K, target), "ran periodic_kernal(np.array([1]), np.array([1]), [1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 2 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([2.5])
    K = kernal(x0, x1, [1, 1, 1, 1], 0.1)
    target = np.array([[0.46134892], [1.51397142], [1.51397142], [0.46134892], [0.53247503]])
    assert np.allclose(K, target), "ran periodic_kernal(np.array([1,2,3,4,5]), np.array([2.5]), [1, 1, 1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 3 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([1,2,3,4,5])
    K = kernal(x0, x1, [1, 0.5, 1, 0.5], 0.2)
    target = np.array([[2.04, 0.7668098, 0.24298264, 0.13669647, 0.19135142],
                       [0.7668098, 2.04, 0.7668098, 0.24298264, 0.13669647],
                       [0.24298264, 0.7668098, 2.04, 0.7668098, 0.24298264],
                       [ 0.13669647, 0.24298264, 0.7668098, 2.04, 0.7668098],
                       [ 0.19135142, 0.13669647, 0.24298264, 0.7668098, 2.04]])
    assert np.allclose(K, target), "ran kernal(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), [1, 0.5, 1, 0.5], 0.2) got {}, wanted {}".format(K, target)
    
    
def test_get_Ks(get_Ks, kernal):
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
    
def test_regression(regression_GP, kernal):
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
    target = np.array([0.94208596248476884, -1.833617068532752, 0.1])

    assert np.allclose(theta, target), "wanted {}, got {}".format(target, theta)
    
def test_sigmoid(sigmoid):
    x = np.linspace(-5, -5, 100)
    target = [1./(1+np.exp(-x_i)) for x_i in x]
    actual = [sigmoid(x_i) for x_i in x]
    assert np.allclose(actual, target)
    x = np.linspace(-50, -5, 100)
    target = [1./(1+np.exp(-x_i)) for x_i in x]
    actual = [sigmoid(x_i) for x_i in x]
    assert np.allclose(actual, target)
    x = np.linspace(5, 50, 100)
    target = [1./(1+np.exp(-x_i)) for x_i in x]
    actual = [sigmoid(x_i) for x_i in x]
    assert np.allclose(actual, target)
    
    
    

    
    
    
    
    
    
    
    
    
    
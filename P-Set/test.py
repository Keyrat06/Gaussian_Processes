import util
import numpy as np


def test_simple_kernal(kernal):
    ## Test 1 ##
    x = np.array([1])
    K = kernal(x, x, [1, 1], 0.1)
    target = np.array([[1.1]])
    assert np.allclose(K, target), "ran kernal(np.array([1]), np.array([1]), [1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 2 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([2.5])
    K = kernal(x0, x1, [1, 1], 0.1)
    target = np.array([[ 0.32465247], [ 0.8824969 ], [ 0.8824969 ], [ 0.32465247], [ 0.04393693]])
    assert np.allclose(K, target), "ran kernal(np.array([1,2,3,4,5]), np.array([2.5]), [1, 1], 0.1) got {}, wanted {}".format(K, target)
    
    ## Test 3 ##
    x0 = np.array([1,2,3,4,5])
    x1 = np.array([1,2,3,4,5])
    K = kernal(x0, x1, [1, 0.5], 0.2)
    target = np.array([[1.20000000e+00, 1.35335283e-01, 3.35462628e-04, 1.52299797e-08, 1.26641655e-14],
                       [1.35335283e-01, 1.20000000e+00, 1.35335283e-01, 3.35462628e-04, 1.52299797e-08],
                       [3.35462628e-04, 1.35335283e-01, 1.20000000e+00, 1.35335283e-01, 3.35462628e-04],
                       [1.52299797e-08, 3.35462628e-04, 1.35335283e-01, 1.20000000e+00, 1.35335283e-01],
                       [1.26641655e-14, 1.52299797e-08, 3.35462628e-04, 1.35335283e-01, 1.20000000e+00]])
    assert np.allclose(K, target), "ran kernal(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), [1, 0.5], 0.2) got {}, wanted {}".format(K, target)
    
                      
                      
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
    K, KS, KSS = get_Ks(x, x1, kernal, theta)
    target_K = np.array([[1.10000000e+00, 6.06530660e-01, 1.35335283e-01, 1.11089965e-02, 3.35462628e-04],
                         [6.06530660e-01, 1.10000000e+00, 6.06530660e-01, 1.35335283e-01, 1.11089965e-02],
                         [1.35335283e-01, 6.06530660e-01, 1.10000000e+00, 6.06530660e-01, 1.35335283e-01],
                         [1.11089965e-02, 1.35335283e-01, 6.06530660e-01, 1.10000000e+00, 6.06530660e-01],
                         [3.35462628e-04, 1.11089965e-02, 1.35335283e-01, 6.06530660e-01, 1.10000000e+00]])
    target_KS = np.array([[1.10000000e+00, 6.06530660e-01, 1.35335283e-01, 1.11089965e-02, 3.35462628e-04]])
    target_KSS = np.array([[ 1.1]])
    assert np.allclose(KSS, target_KSS)
    assert np.allclose(KS, target_KS)
    assert np.allclose(K, target_K)
    
def test_regression(regression_GP, kernal):
    ### test 1 ###
    x = np.array([1])
    y = np.sin(x)
    theta = [1, 0.5, 0.1]
    x_samples = np.array([-1, 0, 0.5, 1, 1.5])
    y_samples, var = regression_GP(x_samples, x, y, kernal, theta)
    target_y_samples = np.array([2.56620062e-04, 1.03527922e-01, 4.63979956e-01, 8.41470985e-01, 4.63979956e-01])
    target_var = np.array([1.0999999, 1.08334942, 0.76556414, 0., 0.76556414])
    assert np.allclose(y_samples, target_y_samples)
    assert np.allclose(var, target_var)
    
    
    ### test 2 ###
    x = np.linspace(-1, 1, 5)
    y = np.sin(x)
    theta = [1, 0.5, 0.1]
    x_samples = np.array([-1.25, -1, 0, 0.5, 1, 1.5])
    y_samples, var = regression_GP(x_samples, x, y, kernal, theta)
    target_y_samples = np.array([-6.69504402e-01, -8.41470985e-01, -2.16376135e-17, 4.79425539e-01, 8.41470985e-01, 4.56950246e-01])
    target_var = np.array([3.54233090e-01, 0.00000000e+00, 4.44089210e-16, 0.00000000e+00, -2.22044605e-16, 7.04870405e-01])
    assert np.allclose(y_samples, target_y_samples)
    assert np.allclose(var, target_var)

    
    

    
    
    
    
    
    
    
    
    
    
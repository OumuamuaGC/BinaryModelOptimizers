import numpy as np

import optimizer as optmz
import visualizer as vs
from func import obj_svm, grad_svm, obj_lr, grad_lr
import func as f



obj = obj_svm
grad = grad_svm


##### Test on various optimizers

result_list = []

"""

# Mushrooms, 10 consecutive tol's, svm

opt = optmz.SgdOptions(tol=1e-3, max_iter=100000, learning_rate=1, decay='lin', batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.stochastic_gradient_descent(obj, grad, opt, x_start)
result_list.append(('Linear', result))

opt = optmz.SgdOptions(tol=1e-3, max_iter=100000, learning_rate=1, decay='sqrt', batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.stochastic_gradient_descent(obj, grad, opt, x_start)
result_list.append(('SquareRoot', result))

opt = optmz.SgdOptions(tol=1e-3, max_iter=100000, learning_rate=1, decay='log', batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.stochastic_gradient_descent(obj, grad, opt, x_start)
result_list.append(('Log', result))

opt = optmz.SgdOptions(tol=1e-3, max_iter=100000, learning_rate=1, decay='', batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.stochastic_gradient_descent(obj, grad, opt, x_start)
result_list.append(('NoDecay', result))
"""


"""
# Test on AdaGrad
opt = optmz.SgdOptions(tol=1e-4, max_iter=100000, learning_rate=1, batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.AdaGrad(obj, grad, opt, x_start)
result_list.append(('AdaGrad', result))
"""

"""
# Mushrooms, 100 consecutive tol's, svm

# Test on RMSProp
opt = optmz.SgdOptions(tol=1e-4, max_iter=100000, learning_rate=0.1, gamma=0.9, batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.RMSProp(obj, grad, opt, x_start)
result_list.append(('RMSProp', result))



# Test on ADAM
opt = optmz.SgdOptions(tol=1e-4, max_iter=100000, learning_rate=0.1, gamma=0.9, beta1=0.9, beta2=0.999, batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.Adam(obj, grad, opt, x_start)
result_list.append(('Adam', result))



# Test on AdaDelta
opt = optmz.SgdOptions(tol=1e-4, max_iter=100000, learning_rate=0.1, gamma=0.99, epsilon=1e-5, batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.AdaDelta(obj, grad, opt, x_start)
result_list.append(('AdaDelta', result))
"""

"""
# Test on gradient method with BB step
opt = optmz.SuperOptions(tol=1e-4, max_iter=100000, s=1, sigma=0.5, gamma=0.1)
x_start = np.random.normal(0, 1, 113)
result = optmz.gradient_method(obj, grad, opt, x_start)
result_list.append(('GM', result))
"""

"""
# Test on AGM_Pro
opt = optmz.SuperOptions(tol=1e-4, max_iter=100000, eta=0.1, alpha=0.1)
x_start = np.random.normal(0, 1, 113)
result = optmz.AGM_Pro(obj, grad, opt, x_start)
result_list.append(('AGM', result))
"""

"""
# Mushrooms, 100 consecutive, svm  

# Test on gradient method with BB step
opt = optmz.SuperOptions(tol=1e-4, max_iter=100000, s=1, sigma=0.5, gamma=0.1)
x_start = np.random.normal(0, 1, 113)
result = optmz.gradient_method_BBstep(obj, grad, opt, x_start, use_armijo=False)
result_list.append(('BB Step', result))


# Test on ADAM
opt = optmz.SgdOptions(tol=1e-4, max_iter=100000, learning_rate=0.1, gamma=0.9, beta1=0.9, beta2=0.999, batch_size=64)
x_start = np.random.normal(0, 1, 113)
result = optmz.Adam(obj, grad, opt, x_start)
result_list.append(('Adam', result))
"""



"""
# Test on AGM with determined Lipschitz constant
opt = optmz.SuperOptions(tol=1e-6, max_iter=100000, eta=0.1, alpha=0.1)
x_start = np.random.normal(0, 1, 113)
result = optmz.accelerated_gradient_method(obj, grad, f.L_cal(), opt, x_start)
result_list.append(result)
"""


opt = optmz.SgdOptions(tol=1e-4, max_iter=300000, learning_rate=1e-3, gamma=0.9, beta1=0.9, beta2=0.999, batch_size=128)
x_start = np.random.normal(0, 1, 11)
result = optmz.Adam(obj, grad, opt, x_start)
result_list.append(('Adam', result))


vs.convergence_analysis(result_list)

import numpy as np


class Options:
    """ Parameter container """
    def __init__(self, tol, max_iter):
        """ 
            Tol: stop criterion
            Max_iter: avoid infinite iterating
        """
        self.tol = tol
        self.max_iter = max_iter


class SuperOptions(Options):
    """ Parameter containers for advanced optimizers """
    def __init__(self, tol, max_iter, beta1=None, beta2=None, s=None, sigma=None, gamma=None, eta=None, alpha=None):
        """
            Parameters for advanced optimizers
        """
        Options.__init__(self, tol, max_iter)
        self.beta1 = beta1
        self.beta2 = beta2
        self.s = s
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        self.alpha = alpha


class SgdOptions(Options):
    """ Parameter containers for stochastic optimizers """
    def __init__(self, tol, max_iter, learning_rate=0.001, decay='lin', batch_size=1, epsilon=10e-6, gamma=None, beta1=None, beta2=None):
        """
            Parameters for stochastic optimizers
            Learning rate: eta
            Decay: decay curve of the learning rate
        """
        Options.__init__(self, tol, max_iter)
        self.lr = learning_rate
        self.batch_size = batch_size
        if decay == 'lin':
            self.decay = lambda k: 1 / (1 + 0.01 * k)
        elif decay == 'sqrt':
            self.decay = lambda k: 1 / (1 + 0.01 * np.sqrt(k))
        elif decay == 'log':
            self.decay = lambda k: 1 / (1 + 0.01 * np.log(k))
        elif decay == 'exp':
            self.decay = lambda k: 0.5 ** k
        else:
            self.decay = lambda k: 1
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2


class Logger:
    """ Record the data during optimization """
    def __init__(self):
        """
            A data logger that records all the useful information during the optimization
            Method: the optimizer name
            Traj: the convergent path of x
            Iteratives: the number of iteratives to convergence
            Start: initial point
            Minimizer: convergent point
        """
        self.method = ""
        self.traj = []
        self.iteratives = 0
        self.start = None
        self.minimizer = None



def stochastic_gradient_descent(obj, grad, opt, x_start):
    """
        Stochastic Gradient Descent Method
        Input:
        Return: 
    """
    # Initialize the data logger
    result = Logger()
    result.method = "SGD Method"
    result.start = np.array(x_start)
    tol_checker = 0
    # Initialize x and the iterator
    x_k = np.array(x_start)
    k = 0
    while k < opt.max_iter:
        result.traj.append(x_k)
        grad_k = grad(x_k, sgd=True, size=opt.batch_size)
        if np.linalg.norm(grad_k) <= opt.tol:
            tol_checker += 1
            if tol_checker > 10:
                break
        else:
            tol_checker = 0
        eta = opt.lr
        beta = opt.decay(k)
        x_k = x_k - (eta * beta / opt.batch_size) * grad_k
        k += 1
        print("Iteration [{}]: gradient norm --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def AdaGrad(obj, grad, opt, x_start):
    """
        Adaptive Gradient Method
        Input:
        Output:
    """
    # Initialize the data logger
    result = Logger()
    result.method = "AdaGrad"
    result.start = np.array(x_start)
    tol_checker = 0
    # Initialize iterative variables
    x_k = np.array(x_start)
    k = 0
    s_k = np.zeros(x_start.shape[0])
    while k < opt.max_iter:
        result.traj.append(x_k)
        grad_k = grad(x_k, sgd=True, size=opt.batch_size)
        if np.linalg.norm(grad_k) <= opt.tol:
            tol_checker += 1
            if tol_checker > 100:
                break
        else:
            tol_checker = 0
        s_k = s_k + grad_k * grad_k
        eta = opt.lr
        eps = opt.epsilon
        x_k = x_k - ((eta / opt.batch_size) / np.sqrt(s_k + eps)) * grad_k
        k += 1
        print("Iteration [{}]: gradient norm --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def RMSProp(obj, grad, opt, x_start):
    """
        RMSProp
        Input:
        Output:
    """
    # Initialize the data logger
    result = Logger()
    result.method = "RMSProp"
    result.start = np.array(x_start)
    tol_checker = 0
    # Initialize x and the iterator
    x_k = np.array(x_start)
    k = 0
    s_k = np.zeros(x_start.shape[0])
    while k < opt.max_iter:
        result.traj.append(x_k)
        grad_k = grad(x_k, sgd=True, size=opt.batch_size)
        if np.linalg.norm(grad_k) <= opt.tol:
            tol_checker += 1
            if tol_checker > 100:
                break
        else:
            tol_checker = 0
        s_k = opt.gamma * s_k + (1 - opt.gamma) * grad_k * grad_k
        eta = opt.lr
        eps = opt.epsilon
        x_k = x_k - ((eta / opt.batch_size) / np.sqrt(s_k + eps)) * grad_k
        k += 1
        print("Iteration [{}]: gradient norm --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def AdaDelta(obj, grad, opt, x_start):
    """
        AdaDelta
        Input:
        Output:
    """
    # Initialize the data logger
    result = Logger()
    result.method = "AdaDelta"
    result.start = np.array(x_start)
    tol_checker = 0
    # Initialize iterative variables
    x_k = np.array(x_start)
    k = 0
    s_k = np.zeros(x_start.shape[0])
    delta_k = np.zeros(x_start.shape[0])
    while k < opt.max_iter:
        result.traj.append(x_k)
        grad_k = grad(x_k, sgd=True, size=opt.batch_size)
        if np.linalg.norm(grad_k) <= opt.tol:
            tol_checker += 1
            if tol_checker > 100:
                break
        else:
            tol_checker = 0
        s_k = opt.gamma * s_k + (1 - opt.gamma) * grad_k * grad_k
        eps = opt.epsilon
        grad_prime = np.sqrt((delta_k + eps) / (s_k + eps)) * grad_k
        x_k = x_k - grad_prime
        delta_k = opt.gamma * delta_k + (1 - opt.gamma) * grad_prime * grad_prime
        k += 1
        print("Iteration [{}]: gradient norm --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def Adam(obj, grad, opt, x_start):
    """
        Adam
        Input:
        Output:
    """
    # Initialize the data logger
    result = Logger()
    result.method = "Adam"
    result.start = np.array(x_start)
    tol_checker = 0
    # Initialize x and the iterator
    x_k = np.array(x_start)
    k = 0
    s_k = np.zeros(x_start.shape[0])
    v_k = np.zeros(x_start.shape[0])
    while k < opt.max_iter:
        result.traj.append(x_k)
        grad_k = grad(x_k, sgd=True, size=opt.batch_size)
        if np.linalg.norm(grad_k) <= opt.tol:
            tol_checker += 1
            if tol_checker > 100:
                break
        else:
            tol_checker = 0
        v_k = opt.beta1 * v_k + (1 - opt.beta1) * grad_k
        s_k = opt.beta2 * s_k + (1 - opt.beta2) * grad_k * grad_k
        v_k_corr = v_k / (1 - opt.beta1 ** (k + 1))
        s_k_corr = s_k / (1 - opt.beta2 ** (k + 1))
        eta = opt.lr
        eps = opt.epsilon
        grad_prime = eta * v_k_corr / (np.sqrt(s_k_corr) + eps)
        x_k = x_k - grad_prime
        k += 1
        print("Iteration [{}]: gradient norm --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def newton_glob(obj, grad, hess, opt, x_start):
    """
        Globalized Newton's Method
        Input: obj - objective function object
               grad - gradient object
               hess - hessian object
               opt - parameter container
               x_start - initial point
        Return: the data logger object containing: method type, trajectory, number of iteratives, start point, convergent point
    """
    # Initilize the data logger
    result = Logger()
    result.method = "Globalized Newton's Method"
    result.start = np.array(x_start)
    # Initialize x and the iterator
    x_k = np.array(x_start)
    k = 0
    while k < opt.max_iter:
        # Record the trajectory and print it
        result.traj.append(x_k)
        #print(">>> Iterative: --- {:2d} ---\n>>> X(k): --- {} ---".format(k, x_k))
        grad_k = grad(x_k)
        # Check stop criterion
        if np.linalg.norm(grad_k) <= opt.tol: 
            break
        # Compute the Newton direction
        hess_k = hess(x_k)
        d_k = np.linalg.solve(hess_k, - grad_k)
        # Check descent criterion
        if not newton_descent_direction(grad_k, d_k, opt):
            d_k = - grad_k
            print("Iteration: ---{}---  Step: --- Gradient ---".format(k), end='')
        else:
            print("Iteration: ---{}---  Step: --- Newton ---".format(k), end='')
        # Search step size by Armijo method
        alpha_k = Armijo_line_search(obj, x_k, grad_k, d_k, opt)
        print("  Alpha: --- {} ---".format(alpha_k))
        # Update x and the iterator
        x_k = x_k + alpha_k * d_k
        k += 1

    result.iteratives, result.minimizer = k, x_k
    return result


def newton_descent_direction(grad_k, d_k, opt):
    """
        Check if the direction is a descent direction of Newton's method
    """
    lhs = - np.inner(grad_k, d_k)
    rhs = opt.beta1 * min(1, np.linalg.norm(d_k) ** opt.beta2) * np.inner(d_k, d_k)
    return lhs >= rhs


def gradient_method(obj, grad, opt, x_start):
    """ Gradient Descend Method """
    
    # Initialize the data logger
    result = Logger()
    result.method = "Gradient Method"
    result.start = np.array(x_start)
    # Initialize the iterator
    k = 0
    x_k = np.array(x_start)
    while k < opt.max_iter:
        result.traj.append(x_k)
        # Evaluate the gradient at xk
        grad_k = grad(x_k)
        if np.linalg.norm(grad_k) <= opt.tol:
            break
        # Search the step size
        alpha_k = Armijo_line_search(obj, x_k, grad_k, - grad_k, opt)
        x_k = x_k - alpha_k * grad_k
        k += 1
        print("Iteration [{}]: gradient norm --- {} --- at x --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def momentum_gradient_method(obj, grad, opt, x_start):
    """ Gradient descend method with momentum """
    pass


def gradient_method_BBstep(obj, grad, opt, x_start, use_armijo=False):
    """ Gradient Descend Method with Barzilai Borwein Step"""
    # Initialize the data logger
    result = Logger()
    result.method = "Gradient Method with BB Step"
    result.start = np.array(x_start)
    # Initialize the iterator
    k = 0
    x_k = np.array(x_start)
    x_prev = None
    grad_prev = None
    while k < opt.max_iter:
        result.traj.append(x_k)
        # Evaluate the gradient at xk
        grad_k = grad(x_k)
        if np.linalg.norm(grad_k) <= opt.tol:
            break
        # Search the step size
        if k == 0:
            alpha_k = Armijo_line_search(obj, x_k, grad_k, - grad_k, opt)
        else:
            alpha_k = Barzilai_Borwein_step(x_k, x_prev, grad_k, grad_prev, version=0)
            if use_armijo:
                opt.s = alpha_k
                alpha_k = Armijo_line_search(obj, x_k, grad_k, - grad_k, opt)
        x_prev = np.array(x_k)
        x_k = x_k - alpha_k * grad_k
        grad_prev = grad_k
        k += 1
        print("Iteration [{}]: gradient norm --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def Barzilai_Borwein_step(x_k, x_prev, grad_k, grad_prev, version=0):
    """ Evaluate the Barzilai-Borwein step by gradients """
    if version == 0:
        return np.inner((x_k - x_prev), (grad_k - grad_prev)) / np.inner((grad_k - grad_prev), (grad_k - grad_prev))
    else:
        return np.inner((x_k - x_prev), (x_k - x_prev)) / np.inner((x_k - x_prev), (grad_k - grad_prev))


def accelerated_gradient_method(obj, grad, L, opt, x_start):
    """
        Accelerated Gradient Method
        Input:
            Objective function and its gradient
            Lipschitz constant of the objective function
            A SuperOption parameter data container
            Start point
        Output:
            A data logger object
    """
    # Initialize the data logger
    result = Logger()
    result.method = "Accelerated Gradient Method"
    result.start = np.array(x_start)

    # Initializing AGM parameters
    k = 0
    t_k = 1
    t_prev = 1
    x_k = np.array(x_start)
    x_prev = np.array(x_start)

    while k < opt.max_iter:
        result.traj.append(x_k)
        # Evaluate the gradient at xk
        grad_k = grad(x_k)
        if np.linalg.norm(grad_k) <= opt.tol:
            break
        # AGM iterative step
        beta_k = (t_prev - 1) / t_k
        y_next = x_k + beta_k * (x_k - x_prev)
        alpha_k = 1 / L
        x_prev = x_k
        x_k = y_next - alpha_k * grad(y_next)
        # Update coefficients
        t_prev = t_k
        t_k = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
        k += 1
        print("Iteration [{}]: gradient norm --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def AGM_Pro(obj, grad, opt, x_start):
    """
        Accelerated Gradient Method without Lipschitz constant
        Input:
            Objective function and its gradient
            A SuperOption parameter data container
            Start point
        Output:
            A data logger object
    """
    # Initialize the data logger
    result = Logger()
    result.method = "Accelerated Gradient Method"
    result.start = np.array(x_start)

    # Initializing AGM parameters
    k = 0
    t_k = 1
    t_prev = 1
    x_k = np.array(x_start)
    x_prev = np.array(x_start)
    alpha_k = opt.alpha
    eta = opt.eta

    while k < opt.max_iter:
        result.traj.append(x_k)
        # Evaluate the gradient at xk
        grad_k = grad(x_k)
        if np.linalg.norm(grad_k) <= opt.tol:
            break
        # AGM iterative step
        beta_k = (t_prev - 1) / t_k
        y_next = x_k + beta_k * (x_k - x_prev)
        grad_y_next = grad(y_next)
        obj_y_next = obj(y_next)
        x_next = y_next - alpha_k * grad_y_next
        while obj(x_next) - obj_y_next > - 0.5 * alpha_k * np.inner(grad_y_next, grad_y_next):
            print(k)
            alpha_k *= eta
            x_next = y_next - alpha_k * grad_y_next
        x_prev = x_k
        x_k = x_next
        # Update coefficients
        t_prev = t_k
        t_k = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
        k += 1
        print("Iteration [{}]: gradient norm --- {} --- at x --- {} ---".format(k, np.linalg.norm(grad_k), x_k))

    result.iteratives, result.minimizer = k, x_k
    return result


def inertial_gradient_method(obj, grad, opt, x_start):
    """
        Inertial Gradient Method
        Input:
            Objective function and its gradient
            A SuperOption data logger
            Start point
        Output:
            A data logger objectt
    """
    # Initialize the data logger
    result = Logger()
    result.method = "Inertial Gradient Method"
    result.start = np.array(x_start)

    # Initializing IGM parameters
    k = 0
    x_k = np.array(x_start)
    x_prev = np.array(x_start)

    while k < opt.max_iter:
        result.traj.append(x_k)
        # Evaluate the gradient at xk
        grad_k = grad(x_k)
        if np.linalg.norm(grad_k) <= opt.tol:
            break
        # IGM iterative step
        alpha = 1.99 * (1 - opt.beta) / opt.l
        y_next = x_k + opt.beta * (x_k - x_prev)
        x_bar = y_next - alpha * grad_k
        l = opt.l
        while obj(x_bar) - obj(x_k) > np.inner(grad_k, x_bar - x_k) + 0.5 * l * np.inner(x_bar - x_k, x_bar - x_k):
            l *= 2
            alpha = 1.99 * (1 - opt.beta) / l
            x_bar = y_next - alpha * grad_k
        x_k = x_bar
        k += 1

    result.iteratives, result.minimizer = k, x_k
    return result


def Armijo_line_search(obj, x_k, grad_k, d_k, opt):
    """
        Armijo Line Search
        Input: 
            Objective function object
            Current k-th x
            Gradient at k-th iterative
            Direction
            Parameter container
        Output: the alpha that satisfies the Armijo condition
    """
    alpha = opt.s
    obj_k = obj(x_k)
    while obj(x_k + alpha * d_k) > obj_k + opt.gamma * alpha * np.inner(grad_k, d_k):
        alpha *= opt.sigma
    return alpha
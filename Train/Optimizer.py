import numpy as np
from queue import Queue



class GradientDescent:
    """ The family of deterministic gradient descent method """

    def __init__(self, model, lr, X=None, y=None, step_search="GD"):
        """
        """
        # For vanilla GD
        self.model = model
        self.step_search = step_search
        self.learning_rate = lr
        # For GD with special step size search method
        if step_search != "GD":
            self.X = X
            self.y = y
        if step_search == "BB":
            self.prev_params = [None, None]


    def step(self):
        """
            Update parameters
        """ 
        if self.step_search == "Armijo":
            step_size = self.Armijo_line_search()
        elif self.step_search == "BB":
            step_size = self.Borwein_Barzilai()
        else:
            step_size = self.learning_rate
        self.model.W = self.model.W - step_size * self.model.grad


    def Armijo_line_search(self, s=1, gamma=0.1, sigma=0.5):
        """
            Armijo Line Search for graident descent method
                Input: hyperparameters for Armijo line search
                Output: the step size alpha that satisfies the Armijo condition
        """
        alpha = s
        while True:
            RHS = self.model.loss(self.X, self.y) - gamma * alpha * np.sum(self.model.grad ** 2)
            self.model.W -= alpha * self.model.grad
            LHS = self.model.loss(self.X, self.y)
            self.model.W += alpha * self.model.grad
            if LHS > RHS:
                alpha *= sigma
            else:
                break
        return alpha


    def Borwein_Barzilai(self):
        """
            Barzilai-Borwein step size
                Output: the step size obtained from previous parameters and gradient    
        """
        if self.prev_params[0] is None: # The first iteration: no previous parameters or gradient, use Armijo instead
            alpha = self.Armijo_line_search()
        else:
            alpha = np.sum((self.model.W - self.prev_params[0]) * (self.model.grad - self.prev_params[1])) / \
                              np.sum((self.model.grad - self.prev_params[1]) * (self.model.grad - self.prev_params[1]))
        self.prev_params[0] = self.model.W.copy()
        self.prev_params[1] = self.model.grad.copy()
        return alpha




class AcceleratedGradientDescent:
    """ Nesterov Accelerated Gradient Descent Method """

    def __init__(self, model, X, y, lr=0.0001):
        self.model = model
        self.learning_rate = lr
        self.t_prev = 1
        self.t_curr = 1
        self.W_prev = model.W.copy()
        self.X = X
        self.y = y

    
    def step(self):
        # Update iterative coefficients
        beta = (self.t_prev - 1) / self.t_curr
        self.t_prev = self.t_curr
        self.t_curr = 0.5 * (1 + np.sqrt(1 + 4 * self.t_curr ** 2))
        # NAG procedure: extrapolate and check descent
        W_extrapolate = self.model.W + beta * (self.model.W - self.W_prev)
        self.W_prev = self.model.W.copy()
        L = self.model.Lipschitz(self.X)
        if L is None and self.learning_rate == 0: # NAG variant: searching the step size
            alpha = 0.1
            eta = 0.5
            self.model.W = W_extrapolate
            grad_extrapolate = self.model.eval_grad(W_extrapolate, self.X, self.y)
            loss_extrapolate = self.model.loss(self.X, self.y)
            self.model.W = W_extrapolate - alpha * grad_extrapolate  
            while self.model.loss(self.X, self.y) - loss_extrapolate > - 0.5 * alpha * np.sum(grad_extrapolate ** 2):
                alpha *= eta
                self.model.W = W_extrapolate - alpha * grad_extrapolate
        else: # Ordinary NAG: fixed step size (Lipschitz constant or a user-defined learning rate)
            if L is None:
                alpha = self.learning_rate
            else:
                alpha = 1 / L
            self.model.W = W_extrapolate - alpha * self.model.eval_grad(W_extrapolate, self.X, self.y)
        



class LimitedMemoryBFGS:
    """ Quasi-Newton Method: Limited-memory BFGS """

    def __init__(self, model, X, y, memo_size=7):
        self.model = model
        self.X = X
        self.y = y
        # Set up the memory buffer
        self.memo_size = memo_size
        self.memo_s = Queue(memo_size)
        self.memo_y = Queue(memo_size)


    def step(self):
        prev_W = self.model.W.copy()
        prev_grad = self.model.grad.copy()

        if self.memo_s.empty():
            alpha = self.Armijo_line_search()
            self.model.W -= alpha * self.model.grad
        else:
            pass


        # Pop the oldest pair of s and y, add the newest pair
        if self.memo_s.full():
            self.memo_s.get()
        if self.memo_y.full():
            self.memo_y.get()
        self.memo_s.put(self.model.W - prev_W)
        self.memo_y.put(self.model.eval_grad(self.model.W, self.X, self.y) - prev_grad)



    def Armijo_line_search(self, s=1, gamma=0.1, sigma=0.5):
        """
            Armijo Line Search for graident descent method
                Input: hyperparameters for Armijo line search
                Output: the step size alpha that satisfies the Armijo condition
        """
        alpha = s
        while True:
            RHS = self.model.loss(self.X, self.y) - gamma * alpha * np.sum(self.model.grad ** 2)
            self.model.W -= alpha * self.model.grad
            LHS = self.model.loss(self.X, self.y)
            self.model.W += alpha * self.model.grad
            if LHS > RHS:
                alpha *= sigma
            else:
                break
        return alpha
        


class StochasticGradientDescent:
    """ SGD on mini-batch """
    
    def __init__(self, model, lr):
        """
        """
        self.model = model
        self.learning_rate = lr
        self.iteration = 0
        self.epoch = 0
        self.X = None
        self.y = None


    def feed_batch(self, X, y):
        """
            Feed a mini-batch to the optimizer
                Input: mini-batch
        """
        self.X = X
        self.y = y
        self.epoch += 1


    def step(self):
        """
            Update parameters
        """ 
        self.model.W = self.model.W - self.learning_rate * self.model.grad
        self.iteration += 1



class Adam(StochasticGradientDescent):
    """ Adam: adaptive + momentum """
    def __init__(self):
        super().__init__(Adam)

    

        
import numpy as np



class RobustLinearRegression:
    """ Robust Linear Regression Model """

    def __init__(self, mu):
        """
            Hyperparameters for Robust Linear Regression
                Mu: smoothing radius for Huber norm
            Parameters:
                W: vector of regression coefficients
                Grad: gradient of loss with respect to parameters
        """
        self.mu = mu
        self.W = None
        self.grad = None


    def param_initializer(self, nDim, method="zero"):
        """
            Initialize model parameters
                nDim: dimension of feature vector
                Method: all-zero or normally-distributed values
        """
        if method == "normal":
            self.W = np.random.randn(nDim, 1)
        elif method == "zero":
            self.W = np.zeros((nDim, 1))
        else:
            raise Exception(" Initializer Not Defined! ")


    def loss(self, X, y):
        """
            Evaluate the value of the loss function 
                Input: design matrix and label vector
                Output: the scalar loss value
        """
        return np.sum(self.huber(self.forward(X) - y))
    

    def forward(self, X):
        """
            Evaluate the predicted values
                Input: design matrix
                Output: vector of predicted labels
        """
        return np.matmul(X, self.W)


    def backward(self, X, y):
        """
            Evaluate the gradient of the loss function
                Input: design matrix and label vector
                Output: updating the gradient vector
        """
        self.grad = np.matmul(X.T, self.huber_grad(np.matmul(X, self.W) - y))


    def huber(self, y):
        """
            Huber norm
                Input: a numpy array
                Output: element-wise Huber norm over the array
        """
        return np.piecewise(y, [np.abs(y) >= self.mu], [lambda y : np.abs(y), lambda y : 0.5 * y ** 2 / self.mu + 0.5 * self.mu])

    
    def huber_grad(self, y):
        """
            Derivative for Huber norm
                Input: a numpy array
                Output: element-wise Huber derivative over the array
        """
        return np.piecewise(y, [y >= self.mu, y <= - self.mu], [1, -1, lambda y : y / self.mu])
        


class SVM:
    """ Support Vector Machine """

    def __init__(self, lmd=0.1, delta=0.01):
        """
            Hyperparameters for SVM
                Lmd: scaling parameter for the marginal term
                Delta: smoothing radius for Huber norm
            Parameters:
                W: vector of linear coefficients
                Grad: gradient of loss with respect to parameters
        """
        self.lmd = lmd
        self.delta = delta
        self.W = None
        self.grad = None


    def param_initializer(self, nDim, initializer="Zero"):
        """
            Initialize model parameters
                nDim: dimension of feature vector
                Method: all-zero or normally-distributed values
        """
        if initializer == "Normal":
            self.W = np.random.randn(nDim, 1)
        elif initializer == "Zero":
            self.W = np.zeros((nDim, 1))
        else:
            raise Exception(" Initializer Not Defined! ")


    def loss(self, X, y):
        """
            Evaluate the value of the loss function 
                Input: design matrix and label vector
                Output: the scalar loss value
        """
        return 0.5 * self.lmd * np.sum(self.W[:-1] ** 2) + 1 / X.shape[0] * np.sum(self.huber(1 - y * (X.dot(self.W[:-1]) + self.W[-1:])))


    def forward(self, X):
        """
            Evaluate the predicted values
                Input: design matrix
                Output: vector of predicted labels
        """
        pred = X.dot(self.W[:-1]) + self.W[-1:]
        return np.piecewise(pred, [pred > 0], [1, -1]) 


    def backward(self, X, y):
        """
            Evaluate the gradient of the loss function
                Input: design matrix and label vector
                Output: updating the gradient vector
        """
        z = 1 - y * (X.dot(self.W[:-1]) + self.W[-1:])
        self.grad = self.lmd * np.row_stack((self.W[:-1], 0)) - 1 / X.shape[0] * np.column_stack((y * X, y)).T.dot(self.huber_grad(z))



    def huber(self, y):
        """
            Huber norm
                Input: a numpy array
                Output: element-wise Huber norm over the array
        """
        return np.piecewise(y, [y <= 0, y > self.delta], [0, lambda y : y - self.delta / 2, lambda y : y ** 2 / (2 * self.delta)])


    def huber_grad(self, y):
        """
            Derivative for Huber norm
                Input: a numpy array
                Output: element-wise Huber derivative over the array
        """
        return np.piecewise(y, [y <= 0, y > self.delta], [0, 1, lambda y : y / self.delta])


    def eval_grad(self, W, X, y):
        """
            Evaluate the gradient of the loss function
                Input: design matrix and label vector
                Output: updating the gradient vector
        """
        z = 1 - y * (X.dot(W[:-1]) + W[-1:])
        return self.lmd * np.row_stack((W[:-1], 0)) - 1 / X.shape[0] * np.column_stack((y * X, y)).T.dot(self.huber_grad(z))


    def Lipschitz(self, X):
        """
            Lipschitz constant
                Output: None for SVM
        """
        return None



class LogisticRegression:
    """ Logistic Regression Model """

    def __init__(self, lmd=0.1):
        """
            Hyperparameters for Logistic Regression
                Lmd: scaling parameter for the L-2 regularization term
            Parameters:
                W: vector of linear coefficients
                Grad: gradient of loss with respect to parameters
        """
        self.lmd = lmd
        self.W = None
        self.grad = None


    def param_initializer(self, nDim, initializer="Zero"):
        """
            Initialize model parameters
                nDim: dimension of feature vector
                Method: all-zero or normally-distributed values
        """
        if initializer == "Normal":
            self.W = np.random.randn(nDim, 1)
        elif initializer == "Zero":
            self.W = np.zeros((nDim, 1))
        else:
            raise Exception(" Initializer Not Defined! ")


    def loss(self, X, y):
        """
            Evaluate the value of the loss function 
                Input: design matrix and label vector
                Output: the scalar loss value
        """
        return 0.5 * self.lmd * np.sum(self.W[:-1] ** 2) + np.mean(np.log(1 + np.exp(- y * (np.matmul(X, self.W[:-1]) + self.W[-1:]))))


    def forward(self, X):
        """
            Evaluate the predicted values
                Input: design matrix
                Output: vector of predicted labels
        """
        sigmoid = 1 / (1 + np.exp(- np.matmul(X, self.W[:-1]) - self.W[-1:]))
        return np.piecewise(sigmoid, [sigmoid > 0.5], [1, -1]) 


    def backward(self, X, y):
        """
            Evaluate the gradient of the loss function
                Input: design matrix and label vector
                Output: updating the gradient vector
        """
        mle_grad = lambda t : 1 - 1 / (1 + np.exp(t))
        z = - y * (np.matmul(X, self.W[:-1]) + self.W[-1:])
        self.grad = self.lmd * np.row_stack((self.W[:-1], 0)) - np.matmul((np.column_stack((y * X, y))).T, mle_grad(z)) / y.shape[0]


    def hessian(self, X, y):
        """
            Evaluate the hessian of the loss function
                Input:
                Output: hessian
        """
        pass


    def eval_grad(self, W, X, y):
        """
            Evaluate the gradient of the loss function
                Input: design matrix and label vector
                Output: updating the gradient vector
        """
        mle_grad = lambda t : 1 - 1 / (1 + np.exp(t))
        z = - y * (np.matmul(X, W[:-1]) + W[-1:])
        return self.lmd * np.row_stack((W[:-1], 0)) - np.matmul((np.column_stack((y * X, y))).T, mle_grad(z)) / y.shape[0]


    def Lipschitz(self, X):
        """
            Calculate the Lipschitz constant
                Input: design matrix
                Output: sample-related Lipschitz constant
        """
        return 0.25 * np.sum(X ** 2) / X.shape[0]




if __name__ == "__main__":
    print("")
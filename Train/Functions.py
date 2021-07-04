import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np



class Logger:
    """ Record data during iterative optimization process """

    def __init__(self, method, start=None, end=None):
        """
            A data logger that records all the useful information during the optimization
                Method: optimizer name
                Traj: the convergent trajectory for various metrics
                Start: initialized values of parameters
                Minimizer: true convergent values of parameters
                Metrics: evaluate the value of some specifc metric 
        """
        self.method = method
        self.metric_traj = {}
        self.start = np.array(start)
        self.minimizer = np.array(end)

        self.metrics = {
            "ParamError" : ( lambda W, W_star : np.linalg.norm(W - W_star) ),
            "Gradient" : ( lambda grad : np.linalg.norm(grad) ),
            "TrainLoss" : ( lambda loss : loss ),
            "TestLoss" : ( lambda loss : loss ),
            "TestAccu" : ( lambda label, pred : np.sum((label * pred) > 0) / label.shape[0] )
        }
    

    def record(self, metric, args):
        """
            Record the metric value to the corresponding trajectory at each epoch
                Metric: metric name
                Args: the arguments that are needed for evaluating the metric value, wrapped in a list
        """
        metric_value = self.metrics[metric](*args)
        if metric not in self.metric_traj:
            self.metric_traj[metric] = [metric_value]
        else:
            self.metric_traj[metric].append(metric_value)


    def plot_metric(self, metric):
        """
            Plot the metric-epoch curve
                Input: metric name   
                Output: visualize the curve 
        """
        traj = self.metric_traj[metric]
        plt.figure()
        sns.set()
        plt.title("Curve for {} ~ Epoch".format(metric))
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.plot(list(range(len(traj))), traj)
        #plt.savefig("./image/param_error.png")
        plt.show()
    


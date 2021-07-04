import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np


def convergence_analysis(result_list):
    """
        Plot the log(||xk - x*||) ~ k curve
        Input:
            A list of result objects   
        Output:
            Show the plot image and save it  
    """
    legend_list = []
    #opt_method = result_list[0].method
    plt.figure()
    sns.set()
    plt.title("Logarithmic plot of ||xk - x*|| with {}".format("Adam and BBstep"))
    plt.xlabel("k")
    plt.ylabel("Log(||xk - x*||)")
    for decay, result in result_list:
        log_dist_norm = [np.log(np.linalg.norm(xk - result.minimizer)) for xk in result.traj[:-1]]
        plt.plot(list(range(len(log_dist_norm))), log_dist_norm)
        legend_list.append(decay)
    plt.legend(legend_list) 
    plt.show()
    #plt.savefig("./image/log_x_norm_{}.png".format(opt_method))

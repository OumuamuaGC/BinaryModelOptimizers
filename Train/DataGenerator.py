import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



class NormalGenerator:
    """ Generate synthetic 2D points that follow a normal distribution """
    
    def __init__(self, c1, c2, sigma1, sigma2, size1, size2):
        """
            C1: center for cluster 1
            C2: center for cluster 2
            Sigma1: standard deviation for cluster 1
            Sigma2: standard deviation for cluster 2
            Size1: number of points in cluster 1
            Size2: number of points in cluster 2
        """
        self.c1 = c1
        self.c2 = c2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.size1 = size1
        self.size2 = size2
        self.a1 = None
        self.b1 = None
        self.a2 = None
        self.b2 = None
        self.normal()


    def normal(self):
        """
            Generate two clusters
            Output: the feature matrix and label vector of two clusters
        """
        self.a1 = np.random.normal(loc=self.c1, scale=self.sigma1, size=(self.size1, 2))
        self.b1 = np.ones(shape=(self.size1, 1))
        self.a2 = np.random.normal(loc=self.c2, scale=self.sigma2, size=(self.size2, 2))
        self.b2 = - np.ones(shape=(self.size2, 1))


    def prepare(self):
        """
            Merge two clusters together and shuffle data points
            Output: the feature matrix and label vector for all data points
        """
        A = np.append(self.a1, self.a2, axis=0)
        b = np.append(self.b1, self.b2, axis=0)
        A, b = shuffle(A, b, random_state=0)
        return A, b


    def show_scatter(self):
        """
            Show the scatter of two clusters
        """
        plt.scatter(self.a1[:, 0], self.a1[:, 1], c="red", alpha=0.5, s=10)
        plt.scatter(self.a2[:, 0], self.a2[:, 1], c="blue", alpha=0.5, s=10)
        plt.scatter(0, 0, marker="D", c="black", alpha=0.8)
        plt.scatter(2, 2, marker="D", c="black", alpha=0.8)
        plt.show()



if __name__ == '__main__':
    generator = NormalGenerator([0, 0], [2, 2], 1, 1, 600, 600)
    A, b = generator.prepare()
    print(A)
    print(b)
    generator.show_scatter()

    

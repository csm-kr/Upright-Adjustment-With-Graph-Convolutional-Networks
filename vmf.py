import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


class VonMisesFisher(object):
    def __init__(self, mu, kappa):
        """
        np.ndarray
        :param mu: np.ndarray - [np.float64]
        :param kappa:
        """
        self.mu = mu
        self.kappa = kappa
        self.pdf_constant = self.kappa / ((2 * np.pi) * (1. - np.exp(-2. * self.kappa)))

    def pdf(self, X):
        """
        The part that receives a point X on the sphere and outputs the probability of vmf
        :param X: np.ndarray(3,)
        :return: scalar : np.float64 ()
        """
        if self.kappa == 0:
            return .25 / np.pi
        else:
            return self.pdf_constant * np.exp(self.kappa * (self.mu.dot(X) - 1.))

    def pdfs(self, X):
        """
        Part that receives multiple points X on the sphere and outputs the probability of vmf
        :param X: np.ndarray - (1000, 3)
        :return: Pdfs = np.ndarray - (1000, 1)
        """
        self.mu = np.expand_dims(self.mu, axis=0)
        X = np.transpose(X)
        if self.kappa == 0:
            return .25 / np.pi
        else:
            return self.pdf_constant * np.exp(self.kappa * (self.mu.dot(X) - 1.))


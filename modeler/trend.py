"""
=========================================================================

Code for the trend block

=========================================================================

"""
import numpy as np


# create trend block
# We create the trend using the block class

class trend:
    """

    Args:
        degree: the degree of the polynomial. 1: constant; 2: linear...
        discount: the discount factor
        name: the name of the trend block
        meanPriorValue: the value to set the prior state mean. Default to 0.
        diagCovPrior: the value to set the prior covariance. Default to a diagonal
           matrix with 1e7 on the diagonal.


    Attributes:
        d: the dimension of the latent states of the polynomial trend
        blockType: the type of the block, in this case, 'trend'
        name: the name of the trend block, to be supplied by user
              used in modeling and result extraction
        discount: the discount factor for this block. Details please refer
                  to the @kalmanFilter
        FF: the evaluation matrix for this block
        GG: the transition matrix for this block
        covStatePrior: the prior guess of the covariance matrix of the latent states
        meanStatePrior: the prior guess of the mean of the latent states

    """

    def __init__(self,
                 degree = 1,
                 discount = 0.99,
                 name = 'trend',
                 meanPriorValue = 0,
                 diagCovPrior = 100):

        self.d = degree
        self.name = name
        self.blockType = 'trend'
        self.discount = np.ones(self.d) * discount

        # Initialize all basic quantities
        self.FF = None
        self.GG = None
        self.covStatePrior = None
        self.meanStatePrior = None

        # create all basic quantities
        self.createFF()
        self.createGG()
        self.createCovStatePrior(diagCovPrior=diagCovPrior)
        self.createMeanStatePrior()

    def createFF(self):
        """ Create the evaluation matrix

        """
        self.FF = np.matrix(np.zeros((1, self.d)))
        self.FF[0, 0] = 1

    def createGG(self):
        """Create the transition matrix

        The transition matrix of trend takes
        a form of \n

        [[1 1 1 1],\n
        [0 1 1 1],\n
        [0 0 1 1],\n
        [0 0 0 1]]

        """
        self.GG = np.matrix(np.zeros((self.d, self.d)))
        self.GG[np.triu_indices(self.d)] = 1

    def createCovStatePrior(self, diagCovPrior=1e7):
        """Create the prior covariance matrix for the latent states.

        """
        self.covStatePrior = np.matrix(np.eye(self.d)) * diagCovPrior

    def createMeanStatePrior(self, meanPriorValue=0):
        """ Create the prior latent state

        """
        self.meanStatePrior = np.matrix(np.ones((self.d, 1))) * meanPriorValue

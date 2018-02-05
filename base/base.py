"""
=================================================================

Code for the base model structure

=================================================================

This piece of code provides the basic model structure for dynamic linear model.
It stores all the necessary components for kalmanFilter and save the results

"""
# dependencies
import numpy as np

# define the basic structure for a dlm model
class base:
    """ The baseModel class that provides the basic model structure for dlm.

    Attributes:
        GG: the transition matrix G
        FF: the evaluation F
        obsNoiseVar: the variance of the observation noise
        stateVar: the covariance of the underlying states
        stateMean: the latent states
        obs: the expectation of the observation
        obsVar: the variance of the observation

    Methods:
        initializeObservation: initialize the obs and obsVar
    """

    # define the components of a baseModel
    def __init__(self, GG = None, FF = None, obsNoiseVar = None, \
                 stateVar = None, stateMean = None):
        self.GG = GG
        self.FF = FF
        self.obsNoiseVar = obsNoiseVar
        self.stateVar = stateVar
        self.stateMean = stateMean
        self.obsMean = None
        self.obsVar = None



    # initialize the observation mean and variance
    def initializeObservation(self):
        """ Initialize the value of obs and obsVar

        """
        self.obsMean = np.dot(self.FF, self.stateMean)
        self.obsVar = np.dot(np.dot(self.FF, self.stateVar), self.FF.T) + \
        self.obsNoiseVar

    # checking if the dimension matches with each other
    def validation(self):
        """ Validate the model components are consistent

        """
        pass

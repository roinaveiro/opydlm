# this class provide all the model building operations for constructing
# customized model
import numpy as np
from opydlm.base.base import base
from opydlm.modeler.matrixTools import matrixTools as mt

class builder:
    """
    Attributes:
        model: the model structure from @baseModel
        stateMeanPrior: the prior mean of the latent state
        stateCovPrior: the prior of the covariance of the latent states
        obsNoiseVar: the prior of the observation noise
        discount: the discounting factor

    Methods:
        add: add new block
        ls:  list out all blocks
        initialize: assemble all blocks
    """

    # create members
    def __init__(self):

        # the basic model structure for running kalman filter
        self.model = None
        self.initialized = False

        # record the prior guess on the latent state and system covariance
        self.stateMeanPrior = None
        self.stateCovPrior = None
        self.obsNoiseVar = None

        # record the discount factor for the model
        self.discount = None

        # components included
        self.blocks = {}

        # store the index (location in the latent states) of all components
        # can be used to extract information for each componnet
        self.componentIndex = {}


    # The function that allows the user to add components
    def add(self, block):
        """ Add a new model component to the builder.

        """

        self.__add__(block)

    def __add__(self, block):

        self.blocks[block.name] = block
        return self

    # print all existing blocks
    def ls(self):
        """ List out all the existing components to the model

        """
        for name in self.blocks:
                blck = self.blocks[name]
                print(blck.name + ' (degree = ' + str(blck.d) + ')')
                print(' ')



    def initialize(self, noise=1):

        """ Initialize the model. It construct the base by assembling all
        quantities from the components. Models are typically initialized when
        fitting.

        Args:
            noise: the initial guess of the variance of the observation noise.
        """

        # construct transition, evaluation, prior state, prior covariance

        FF = None
        GG = None
        stateMean = None
        stateVar = None
        self.discount = np.array([])


        #loop for the case of having several blocks
        currentIndex = 0  # used for compute the index
        for i in self.blocks:
            blck = self.blocks[i]
            GG = mt.matrixAddInDiag(GG, blck.GG)
            FF = mt.matrixAddByCol(FF, blck.FF)
            stateMean = mt.matrixAddByRow(stateMean, blck.meanStatePrior)
            stateVar = mt.matrixAddInDiag(stateVar, blck.covStatePrior)
            self.discount =  np.concatenate((self.discount, blck.discount))
            self.componentIndex[i] = (currentIndex, currentIndex + blck.d - 1)
            currentIndex += blck.d




        self.stateMeanPrior = stateMean
        self.stateCovPrior = stateVar
        self.obsNoiseVar = np.matrix(noise)
        self.model = base(GG=GG,
                               FF=FF,
                               obsNoiseVar=np.matrix(noise),
                               stateVar=stateVar,
                               stateMean=stateMean)
        self.model.initializeObservation()

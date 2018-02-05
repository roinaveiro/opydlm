from opydlm.modeler.builder import builder

class odlm:

    def __init__(self, data, **options):

        self.data = list(data)
        self.n = len(data)
        self.builder = builder()
        self.time = None

    def add(self, component):
        """ Add new modeling component to the dlm.

        Currently support: trend.

        Args:
            component: the modeling component

        Returns:
            A dlm object with added component.

        """
        self.__add__(component)

    def __add__(self, component):
        self.builder.__add__(component)
        return self

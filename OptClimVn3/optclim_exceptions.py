"""
Define all exceptions used by OptClimVn3

"""


class submitModel(Exception):
    """
    Error when need to run a model!
    Inherits everything from Exception
    """
    pass

class enoughProvisionalCases(Exception):
    """"
    Error when generating provisional cases and have enough.
    Inherits from Exception.
    """

    pass

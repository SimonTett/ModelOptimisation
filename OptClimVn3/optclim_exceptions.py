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

class useCreatedModel(Exception):
    """" 
    Error when using a created model which can happen when reusing old models and doing 2nd run.
    Inherits from Exception.
    """

    pass



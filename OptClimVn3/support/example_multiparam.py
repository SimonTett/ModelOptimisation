# Example function for multiple params
# will just return the control + difference between plus4k and control values.
import typing

import pandas as pd

import Model
import logging



def ctl_plus4k(models: typing.Dict[typing.Hashable,Model.Model]) -> typing.Optional[pd.Series]:
    """
    Return control values concatenated with differences from plus4k case. This is an example case
    :param models: Dict of models. Should have the keys control and plus4k
    :return:Pandas series (or None) of concatenated series of ctl and delta. delta values have index begining delta_
    """

    ctl = models['control'].simulated_obs # get the obs
    plus4k = models['plus4k'].simulated_obs
    if (ctl is None) or (plus4k is None):
        return None  # not computed ctl or plus4k cases so just return None
    new_index = ['delta_'+idx for idx in ctl.index] # new index for delta
    delta = (plus4k - ctl).set_axis(new_index) # compute delta and set axis
    result:pd.Series = pd.concat([ctl,delta]) # concat values together.
    return result

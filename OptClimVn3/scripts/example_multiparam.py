# Example function for multiple params
# will just return the control + difference between plus4k and control values + models.
import typing
import pandas as pd
from StudyConfig import type_basic
from Model import Model # for type hints.
def ctl_plus4k(get_model: typing.Callable[[dict[str,type_basic]],typing.Optional[Model]],
               parameter_dict:dict[str,dict[str,type_basic]],
               use_cache:bool = True) -> tuple[list[typing.Optional[Model]],typing.Optional[pd.Series]]:

    """
    Return control values concatenated with differences from plus4k case. This is an example case
    :param get_model: fn to compute/read simulated observations.
      Should take a dict of parameters and return a Model or None.
    :param parameter_dict: Dict of parameters for models. Should have the keys control and plus4k
    :param use_cache: Passed to compute_simulated_observations
    :return: List of models/None  (or list of None) & Pandas series (or None) of concatenated series of ctl and delta.

    """


    # get models and simulated obs. This case has no dependencies between models so is straightforward.
    # a more complex case would be where one model depends on another.
    # in that case would need to run models in expected order checking that they exist before running dependent models.
    # more complex cases could be handled by modifying parameters in models based on results from other models.
    all_models=dict()
    all_sim_obs = dict()
    for name, param_dict in parameter_dict.items():
        model = get_model(param_dict) # this should create a new model or return an existing one.
        if not isinstance(model,(Model,type(None))):
            raise ValueError(f"get_model returned {type(model)} instead of a Model")
        all_models[name] = model # store the model (or None) in the dict.
        if model is None: # None means we failed to get a model.
            all_sim_obs[name] = None
        else: # have  model so compute simulated obs.
            all_sim_obs[name] = model.compute_simulated_observations(use_cache=use_cache)


    # check whether any of the simulated observations are None. If so return the models and None for the series.
    if any([sim_obs is None for sim_obs in all_sim_obs.values()]):
        return list(all_models.values()),None
    ctl:pd.Series = all_sim_obs['control'] # get the obs
    plus4k:pd.Series = all_sim_obs['plus4k'] # get the plus-4k obs

    new_index = ['delta_'+idx for idx in ctl.index] # new index for delta
    delta = (plus4k - ctl).set_axis(new_index) # compute delta and set axis
    result:pd.Series = pd.concat([ctl,delta]) # concat values together.
    return list(all_models.values()),result



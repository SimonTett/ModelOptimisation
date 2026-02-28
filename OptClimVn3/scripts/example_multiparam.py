# Example function for multiple params
# will just return the control + difference between plus4k and control values.
import typing
import pandas as pd

type_basic = int|float|str|bool
def ctl_plus4k(run_submit_instance,
                      parameter_dict:dict[str,dict[str,type_basic]]) -> typing.Optional[pd.Series]:

    """
    Return control values concatenated with differences from plus4k case. This is an example case
    :param run_submit_instance: a runSubmit instance
    :param parameter_dict: Dict of parameters for models. Should have the keys control and plus4k
    :return:Pandas series (or None) of concatenated series of ctl and delta.
     delta values have index beginning delta_
    """
    models = dict()

    # get models and simulated obs. This case has no dependencies between models so is straightforward.
    # a more complex case would be where one model depends on another.
    # in that case would need to run models in expected order checking that they exist before running dependent models.
    # more complex cases could be handled by modifying parameters in models based on results from other models.
    for name, param_dict in parameter_dict.items():
        models[name] = run_submit_instance.make_model(param_dict,reference_name=name)
        # this will create a new model or return an existing one.


    # get the simulated obs for this model. If None model not been run yet.
    # test that all models ran and produced obs. Any that are None will mean can't compute result.
    # This implies no dependencies between models. If there are dependencies then return None when
    # dependant model obs is None.
    if any([m.simulated_obs is None for m in models.values()]):
        return None
    ctl = models['control'].simulated_obs # get the obs
    plus4k = models['plus4k'].simulated_obs

    new_index = ['delta_'+idx for idx in ctl.index] # new index for delta
    delta = (plus4k - ctl).set_axis(new_index) # compute delta and set axis
    result:pd.Series = pd.concat([ctl,delta]) # concat values together.
    return result

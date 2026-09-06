from __future__ import annotations

import functools
import logging
import pathlib
import random
import typing
import warnings

import dfols  # needed for DFOLS stuff. Eventaully move into class defn.
import numpy as np
import pandas as pd

import genericLib
import optclim_exceptions
from Model import Model
from StudyConfig import OptClimConfigVn3
from SubmitStudy import SubmitStudy
from model_base import model_base

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")


type_none = type(None)  # type for None



class LogicalInfo(model_base):
    """ Class to hold logical information about parameters, observations etc. 

    This to be used in runSubmit to hold information about the logical observations, parameters and related information.
      Idea being that logical param set might correspond to multiple model evaluations.
      As a class it is really a place to hold related things together and make maintenance easier.
      It is a subclass of model_base so has access to to_dict and from_dict methods which
      make it easy to save and restore the information.

      It has only a small number of  methods and for most use runSubmit will need to reach inside this class. This does not feel ideal.
      An alternative would be to pass appropriate information from runSubmit to methods here.

      It has the following attributes:
        iteration_count: int -- count of iterations. Used in generating  names.
        count_within_iteration: int -- count of models within an iteration. Used in generating  names.
        names: dict[str,str] -- dict of logical names indexed by key generated from parameters.
        parameters: dict[str,pd.Series] -- dict of parameters indexed by logical name.
        observations: dict[str,pd.Series] -- dict of observations indexed by logical name.
        cost: dict[str,float] -- dict of cost indexed by logical name.
        models: dict[str,list[Model]] == dict of list of models indexed by logical_name
    
    """

    def __init__(self):
        super().__init__()

        self.iteration_count: int = 0  # count of iterations. Used in generating logical names.
        self.count_within_iteration: int = 0  # count of models within an iteration. Used in generating logical names.
        self.names: dict[str, str] = dict()  # dict of logical names indexed by key generated from parameters.
        self.parameters: dict[str, pd.Series] = dict()  # dict of logical parameters indexed by logical name.
        self.model_keys: dict[str, list[str]] = dict()  # dict of list of keys to models indexed by logical_name.
        self.observations: dict[str, pd.Series] = dict()  # dict of observations indexed by logical name.
        self.cost: dict[str, float] = dict()  # dict of cost indexed by logical name.


    @staticmethod
    def key(params: dict) -> str:
        """
        Generate a key for a set of parameters. This is based on sorting the items in params and then generating a string.
        :param params: dictionary of parameters. Uses Study.key fn.
        :return: key string.
        """
        key = SubmitStudy.key(params)  # actually using Study key fn. Could also  include fpFmt if needed.
        return key



    def completed_iteration(self):
        """
        Mark the end of an iteration. Increases iteration count and resets count within iteration.
        :return: None
        """
        self.iteration_count += 1
        self.count_within_iteration = 0

    def name(self, params: dict) -> str:
        """
        Generate a name for a set of parameters. This is based on the iteration number and index within iteration.
        :param params: dictionary of parameters.
        :return: logical name.
        """
        key = self.key(params)
        try:
            name = self.names[key]  # already have a name for this set of parameters.
            my_logger.debug(f"Already got name {name}")
        except KeyError:  # need to generate a new name.
            name = f"I{self.iteration_count}_i{self.count_within_iteration}"
            self.count_within_iteration += 1  # increase count for next time.
            self.names[key] = name
            my_logger.debug(f"Generated {name}")


        return name

    
    def params(self, params: dict[str,type_param]) -> tuple[str,pd.Series]:
        """
        Set the parameters
        :param params: dict of  parameters.
        :return: A generated name & pandas series of the parameters with name = generated name
        if name is not in self.parameters then it is added. If it is already there then it is updated.
        updating updates the key used.
        """
        name = self.name(params)
        if name in self.parameters: # got the name already
            my_logger.info(f"Updating params for {name}")
        self.parameters[name] = pd.Series(params).rename(name)
        return name,self.parameters[name]



    def obs(self,name: str, observations:typing.Optional[pd.Series]=None) -> typing.Optional[pd.Series]:
        """
        set/get the observations -- potentially overwriting existing observations.

        :param name: logical name.
        :param observations: observations to set. If None will just return current obs corresponding to name
        :return: the updated observations or None if not found
        """

        if observations is not None: # got some obs 
            if name in self.observations: 
                my_logger.info(f"Overwriting observations for {name}")
            observations = observations.rename(name)
            self.observations[name] = observations
        else:
            if name not in self.observations:
                return None
        return self.observations[name]



    def update_parameters(self, name, new_params: dict[str,type_param]) -> pd.Series:
        """
        Update  the parameters inplace for a given logical name
        :param name: logical name.
        :param new_params: dict of  parameters.
        :return: the updated parameters
        if name is not in self.parameters then it is added. If it is already there then it is updated.
        updating updates the key used.

        This will eventually be removed.
        """
        if name not in self.parameters:
            my_logger.warning(f"Unknown params {new_params}. Storing")
            name,params = self.params(new_params)
        else: # already got it. So up`date. Tricky part is dealing with key changes.
            my_logger.info(f"Updating `params for {name}")
            orig_params = self.parameters[name].to_dict()  # original params
            key = self.key(orig_params)  # original key
            new_key = self.key(new_params)
            old_name = self.names.pop(key)  # remove old key/name mapping
            if old_name != name:
                raise ValueError(f"Logical name {name} does not match stored name {old_name}")  # SHOULD NOT HAPPEN
            self.names[new_key] = name
            name, params = self.params(new_params) # store params
        # now have param_values and store them.
        return params





    @classmethod
    def from_dict(cls, dct: dict) -> LogicalInfo:
        """
        Create LogicalInfo from a dictionary. Uses super class method.
        Updates "old" format which used list of 'keys' to dict of 'models'.
        :param dct: dictionary to parse.
        :return: LogicalInfo object.
        """
        # deal with old version which uses keys and set models to it.
        # This code can be removed once old keys based data is no longer in use or has been converted.
        if 'keys' in dct and not ('models'  in dct or 'models_keys' in dct):
            dct['model_keys'] = dct.pop('keys')
            my_logger.warning(
                'LogicalInfo.from_dict -- converting keys to model_keys -- please update saved data to new format')

        elif 'keys' in dct and 'models' in dct:
            raise ValueError("LogicalInfo.from_dict -- found both 'keys' and 'models' in dict -- cannot proceed")
        elif 'models' in dct:
            # for a while models was a dict of dicts with the inner dict being model names + keys.
            # if this is so need to flatten them.
            model_dct = {}
            convert_message = True
            for logical_name, models in dct['models'].items():
                if isinstance(models, dict):
                    model_dct[logical_name] = list(models.values())  # just want the values as a list.
                    if convert_message:
                        my_logger.warning('Converting model keys from dict to list')
                        convert_message = False  # only want message to come out once
                else:
                    model_dct[logical_name] = models

            # at this point models are actually keys. So we are good to go
            dct['model_keys'] = model_dct  # replaced with updated dct.
        else:
            pass  # we are good!

        # and we changed obs to observations...
        if 'obs' in dct:
            dct['observations'] = dct.pop('obs')
            my_logger.warning('LogicalInfo.from_dict -- converting obs to observations -- please save configuration.')
            # check all elements are pandas series.
            for k, v in dct['observations'].items():
                if not isinstance(v,pd.Series):
                    raise ValueError(f"observations for {k} is not a pd.Series but {type(v)} ")

        obj: LogicalInfo = super().from_dict(dct)



        return obj

    def keys_to_models(self, model_index: dict[str, Model],
                       update_models: bool = False) -> dict[str, Model]:
        """
        This is a support function to be called from runSubmit.from_dict after the LogicalInfo object has been created.
        It converts models stored as keys to Model objects using model_index OR updates the models.

        :param model_index: dict of models indexed by key.
        :return: Nada as models updated inplace.
        """
        raise NotImplementedError("This method no longer supported. Code now uses the keys directly  and then pull from model_index")
        names_to_remove=[] # list of names to remove if we do not find a model.
        for name, model_list in self.models.items(): # get logical name and list of models or keys for this logical name
            final_model_list = []
            got_model =True
            for model_or_key in model_list:
                if update_models:
                    key = runSubmit.key_for_model(model_or_key)
                else:
                    key = model_or_key
                try:
                    final_model_list += [model_index[key]]  # this is a model
                except KeyError:
                    my_logger.warning(f"Key {key} not found in model_index. Will delete {name}.")
                    names_to_remove+=[name]
                    
                    #raise ValueError(f"Key {key} not found in model_index")
            # done dealing with models for this logical name.
            if final_model_list: # got some models for this name?
                self.models[name] = final_model_list  # update models to be Model objects.


        if len(names_to_remove)>0:
            my_logger.warning(f"Removing {names_to_remove} from cost, observations, params & models.")
        for name in names_to_remove: # remove things. 

            self.cost.pop(name,None)
            self.observations.pop(name,None)
            self.parameters.pop(name,None)
            self.models.pop(name,None)
        

    # End of LogicalInfo class.


class runSubmit(SubmitStudy):
    # Has the following additional attributes over SubmitStudy (and Study)
    # _logical_info: LogicalInfo -- holds information about logical names, parameters, observations, cost and models.
    #  This is a private attribute as it is really only for use within this class. It is not intended to be used outside this class.


    """
          Class   to deal with running various algorithms. (not all of which are optimization).
          It is a specialisation of SubmitStudy and is separated out to make maintenance easier.
          Idea is that each method should use cases that exist and deal with parallelism.
          If it finds out that cases are missing it should raise optclim_exceptions.submitModel.

          Functions should  return a finalConfig which contains
          whatever additional information they consider useful.

       To add a new algorithm then either add it as a method in here or subclass this and add your own runMethods.
       runJacobian is fairly simple and might form a good model for this. See runAlgorithm.py which is the script that
       runs the whole system.


     For *now* all models will use the same post-processing.
    So only thing that can differ is fixed parameters or reference model. Not even model type etc.
    Type information for the function is:
     typing.Callable[[runSubmit, dict], typing.Optional[pd.Series]]

    """

    def __init__(self,
                 config: typing.Optional[OptClimConfigVn3],
                 name: typing.Optional[str] = None,
                 rootDir: typing.Optional[pathlib.Path] = None,
                 refDir: typing.Optional[pathlib.Path] = None,
                 models: typing.Optional[typing.List["Model"]] = None,
                 model_name: typing.Optional[str] = None,
                 config_path: typing.Optional[pathlib.Path] = None,
                 next_iter_cmd: typing.Optional[typing.List[str]] = None):
        super().__init__(config, name, rootDir, refDir, models, model_name, config_path, next_iter_cmd)

        # _logical_info holds information for per optimisation parameter info.
        # Currently, largely a bag of attributes which this class reaches into as it needs to.
        # Having _logical_info as private for now as not sure if it will be needed outside this class.
        self._logical_info: LogicalInfo = LogicalInfo()
        if models:
            for model in models:
                self.model_used_status(model, 'initial')  # set model status to 'initial'


    def transform_check(self, observations: pd.Series,
                        transform: typing.Optional[pd.DataFrame] = None,
                        scale: bool = False,
                        residual: bool = False,
                        ) -> pd.Series:

        """
        Check and transform observations
        :param observations: observations to be checked and transformed. Will reduce them to the obsNames in self.config.obsNames()

        :param scale: If True scale observations
        :param residual: If True observations are relative to target values
        :param transform: Transform matrix

        Obs are transformed in the order scale, residual, transform.
        return: pandas series of transformed observations

        Checks are that no missing observations, no desired observations are missing and no nulls are in transformed series.
        """

        obsNames = self.config.obsNames()
        missing_obs = set(obsNames) - set(observations.index)
        if len(missing_obs) > 0:  # trigger error as missing observations
            raise ValueError(f"Missing {' '.join(missing_obs)} from {observations.name}")
        # force fixed order and select only those observations we want.
        observations = observations.reindex(obsNames)
        # note using obsNames as specified. transform (if supplied) can change names.
        if scale:  # scale sim observations.
            observations *= self.config.scales()
        if residual:  # difference from target observations
            tgt = self.config.targets(scale=scale)
            observations -= tgt
        if transform is not None:  # apply transform if required.
            observations = observations @ transform.T  # observations in nsim x nobs; transform  is nev x nobs.
            # observations = transform@observations.T # how we should do it.
        null = observations.isnull()
        if np.any(null):
            raise ValueError("Obs contains null values at: " + ", ".join(observations.index[null]))
        return observations

    def xxlogical_models(self, name,models:typing.Optional[list[Model]]=None) -> list[Model]:
        """
        Get (or set) list of models associated with a logical name.
        :param name: logical name
        :param models: optional list of models
        :return: list of models associated with this logical name.
        """
        if models is not None:
            self.models[name] = models # store the models
        return self._logical_info.models[name] # return them

    def logical_params(self, normalize: bool = False,
                       ) -> pd.DataFrame:
        """
        Return a dataframe of parameters for all logical names.
        :param normalize: Normalize parameters to [0,1] based on param limits.
          ensembleMember will be normalized by ensemble size. See StudyConfig.paramRanges()

        :return: df of parameters indexed by logical name.
        """

        # now construct dataframe.
        df = pd.DataFrame(self._logical_info.parameters).T
        if normalize:  # Normalize by param ranges
            ranges = self.config.paramRanges(df.columns, ensemble=True).reindex(columns=df.columns)
            df = (df - ranges.loc['minParam', :]) / ranges.loc['rangeParam', :]
        return df

    def update_logical_params(self, name: str, parameters: list[str],
                              key_mapping: dict[str, str]) -> pd.Series:
        """
        Update the parameters for a given logical name.
           Checks that all existing parameters are in params
         and then updates the parameters.
          Provided for case when parameters are updated for model sims after initial storage.
        :param name: logical name.
        :param parameters: list of parameters to update. Parameters updated will be these + any existing parameters.
        :param key_mapping: mapping from old_key to new_key. Used to update self._logical_info.model_keys
        :return: pandas series of updated parameters but logical_info is updated in place.

        Note this method does not write anything to disk. Just changes values internally.
        This method is fragile and in-place updating is not really necessary.
         Trick is to update models and start a new study using those models.
        """
        # first check existing params are all in params.
        orig_params = self._logical_info.parameters[name].to_dict()
        params = list(set(list(orig_params.keys()) + parameters))  # combine existing and new params.
        # Need to update the model parameters from the models...

        ref_param_values = None
        param_values = None
        keys_to_update = [key_mapping[key] for key in self._logical_info.model_keys[name]]
        self._logical_info.model_keys[name] = keys_to_update
        for key in keys_to_update:
            model = self.model_index[key]
            param_values = pd.Series(model.parameters).reindex(params).rename(model.name)
            # check all values are there
            # may not be nesc as I think reindex fails if params missing.
            if param_values.isnull().any():
                missing = param_values.index[param_values.isnull()].tolist()
                raise ValueError(f"Missing params {missing} for logical name {name} in model {model}")
            # check params are the same for all models.
            if ref_param_values is None:  # first time in. Store param_values
                ref_param_values = param_values
            elif not param_values.equals(ref_param_values):  # values differ so raise error
                raise ValueError(f"Parameter values for logical name {name} differ between models. "
                                 f"Model {model} has params {param_values} but expected {ref_param_values}")
            else:
                pass  # all ok.


        new_param = self._logical_info.update_parameters(name,param_values.to_dict())  # update the parameters in place.
        return new_param

    def logical_cost(self) -> pd.Series:
        """
        Return a series of costs for all logical names computing costs as needed.
        :return: series with costs
        """
        transform = None
        for name in self._logical_info.names.values(): # loop over list of names
            if name not in self._logical_info.cost:
                # Compute the cost for this logical name
                obs = self._logical_info.observations.get(name)
                if obs is not None:
                    if transform is None:
                        transform = self.config.transform_matrix(scale=True)
                    obs = self.transform_check(obs, transform=transform, scale=True, residual=True)
                    # compute the cost which uses the transformed data.
                    self._logical_info.cost[name] = (obs ** 2).mean()  # store the mean square diff.


        return pd.Series(self._logical_info.cost).rename("cost")

    def logical_obs(self,
                    normalize: bool = False,
                    scale: bool = True,
                    obs_names: typing.Union[bool, list[str], None] = None) -> pd.DataFrame:
        """
        DO NOT USE. Use simulated_obs() instead.
        :param normalize:
        :param scale:
        :param obs_names:
        :return:
        """
        raise NotImplementedError("Use simulated_observations() instead.")
    def simulated_observations(self,
                    normalize: bool = False,
                    scale: bool = True,
                    obs_names: typing.Union[bool, list[str], None] = None,
                               use_cache:bool = True) -> typing.Optional[pd.DataFrame]:
        """
        Return a dataframe of observations for all logical params.
        :param normalize: normalize the observations by error estimates from target
        :param scale: scale the observations by self.config.scales()
        :param obs_names: Names to use, If None return everything in self.config.obsNames().
           If True uses all observations in self._logical_info.observations. This might fail with normalize if not all observations are present in tgt.
        :param use_cache: Whether to use cached observations or not.
        :return: dataframe of observations or None if nothing found
        """

        # use compute_simulated_observations to actually compute the observations for each logical name. This will use the cache if available.
        obs = []
        for name, params in self._logical_info.parameters.items():
            sim_obs = self.compute_simulated_observations(params.to_dict(), use_cache=use_cache)
            if sim_obs is not None:
                obs.append(sim_obs)

        if len(obs) == 0:
            return None
        observations = pd.DataFrame(obs)
        # deal with obsNames
        if isinstance(obs_names, bool) and obs_names:
            obs_names = observations.columns
        elif obs_names is None or (isinstance(obs_names, list) and not obs_names):
            obs_names = self.config.obsNames()
        else:
            pass  # anything else is assumed to be a list of obsNames

        observations = observations.reindex(columns=obs_names).dropna(axis=1)

        if scale:  # scale ?
            observations *= self.config.scales(obsNames=obs_names)

        if normalize:  # normalize
            tgt = self.config.targets(scale=scale, obsNames=obs_names)
            observations -= tgt  # difference from tgt.
            cov = self.config.Covariances(scale=scale)  # get covariances.
            errCov = cov['CovTotal']  # just want the total
            sd = pd.Series(np.sqrt(np.diag(errCov)),
                           index=errCov.index)  # square root of diagonal elements. Need to reindex.
            sd = sd.reindex(obs_names)  # extract only those we want.
            observations /= sd  # normalise by SD

        return observations

    def models(self,name:str,models:typing.Optional[list[Model]]=None) -> list[Model]:
        """
        Return list of models corresponding to name (a logical name) or set them.
        :param models: list of models. If passed in will be used to set _logical_info.model_keys[names]
        """
        if models is not None:
            keys = [self.key_for_model(model) for model in models]
            # check for missing keys,
            missing_keys = [key for key in keys if key not in self.model_index.keys()]
            if len(missing_keys) > 0:
                raise ValueError(f"Missing keys {missing_keys}")
            self._logical_info.model_keys[name] = keys

        keys = self._logical_info.model_keys[name] # will fail if name not present,
        for key in keys:
            if key not in self.model_index:
                raise ValueError(f"Missing key {key} for name:{name}")
        models = [self.model_index[key] for key in keys]  # will also fail if model key not present.
        return models

    type_multi_model_fn = typing.Callable[
        ["runSubmit", dict[str, dict]], tuple[list[Model|None], typing.Optional[pd.Series]]]

    # multi model fn takes as args a runSubmit obj and a dict and returns
    #    a list of Models (or Nones)  and pandas series (of observations)/None (if sims do not exist)
    type_param = typing.Union[float,int,str,bool] # allowed types for parameters

    def _set_ref_path(self, param: dict[str, type_param]):
        """
        Set the reference path for a given parameter in a param dict.
        If reference is not in param then set it to self.refDir.
        Then expand the path and convert to posix path.
        :param param: dict of parameters
        :return: Nada. Modifes param in place.
        Not intended for public use. This is a support function for gen_param_dict.
        """
        ref = param.get('reference', self.refDir)
        # get reference value  if they are there; if not use self.refDir.
        ref = self.expand(ref).as_posix()  # expand and convert to posix path.
        param.update(reference=ref)  # add in reference params



    def gen_param_dict(self, param: dict) -> list[dict]:
        """
        Generate a list of parameter dicts by merging params, fixed_param and ensemble member.
        :param param: dictionary of parameters
        :return: list of combined dictionaries of parameters. 1 member per ensemble member.

        Consider renaming ensembleMember to random seed or similar.
        """



        fixed_params = self.config.fixedParams()
        multi_config_fn = self.config.fixed_param_function()
        n_ensemble = self.config.ensembleSize()
        if 'ensembleMember' in param: # we are setting ensemble_members explicitly. So set n_ensemble to 1.
            n_ensemble = 1
            my_logger.warning("Hacky code -- setting n_ensemble to 1 as ensembleMember is set in params. Consider refactoring")
        param_list = []
        # If ensembleMember in param
        if n_ensemble > 1:
            if 'ensembleMember' in param:
                raise NotImplementedError(
                    "Do not pass ensembleMember in params when n_ensemble > 1. "
                    "It is generated automatically by this method. ")
            if 'ensembleMember' in fixed_params:
                raise NotImplementedError(
                    "Do not set ensembleMember in fixed_params when n_ensemble > 1. "
                    "It is generated automatically by this method.")
        for ens in range(n_ensemble):
            if n_ensemble == 1: # never set ensembleMember when n_ensemble is 1
                ens_param = {}  # empty dict
            else:
                ens_param = dict(ensembleMember=ens)  # set ensemble member

            if multi_config_fn is None:  # simple calculation. Just need to add ens_param to params.
                full_params = fixed_params | param | ens_param  # needs to be using  python 3.9+ for | operator. params has higher precidence than fixed_params
                self._set_ref_path(full_params)
            else: # multi config function. Need to call it to get the full params for each ensemble member.
                full_params = dict()
                for key, fixed in fixed_params.items():
                    full_params[  key] = fixed | param | ens_param
                    # needs to be using  python 3.9+ for | operator. param has higher precidence than fixed
                    self._set_ref_path(full_params[key])
            param_list.append(full_params)

        return param_list

    def compute_simulated_observations(self,
                        params: dict[str,type_param],
                        use_cache: bool = True,
                        ) -> typing.Optional[pd.Series]:
        """
        Compute the simulated observations for a *single* parameter set. This could involve running multiple models and averaging the results/doing some processing.
        If any simulated obs is None then return None.
        :param params: dictionary of parameters and values
        :param use_cache: If True use the cache. Setting use_cache to False will force regeneration of obs all the way down to the underlying models.
        :param reference_name: name for reference.
        :return: pandas series of simulated observations or None if some run is needed.


        This method caches generated observations, models & params.

        It ensemble averages the simulated observations for each ensemble member and returns the average.

        """



        name = self._logical_info.name(params)  # get the name based on the params.
        if use_cache:   # got a name and we have observations for it.
            sim_obs = self._logical_info.obs(name)
            if sim_obs is not None:
                my_logger.debug("Used cache to return simulated observations")
                models = self.models(name)
                for model in models:
                    self.model_used_status(model, 'called')
                return sim_obs

        # otherwise we need to compute it.
        self._logical_info.params(params)  # store the parameters
        all_sim_obs=[] # list of simulated observations for each ensemble member.
        params_list = self.gen_param_dict(params) # generate a list of parameter dicts for each ensemble member.
        multi_config_fn = self.config.fixed_param_function() # are we using a multi config fn?
        all_models = [] # list of models
        for param in params_list:
            if multi_config_fn:
                models,sim_obs = multi_config_fn(self.get_model,param,use_cache=use_cache)
                # check got back what we expected. Fail if not.
                if not isinstance(sim_obs,(pd.Series,type_none)):
                    raise ValueError(f"multi_config_fn returned {type(sim_obs)} instead of pd.Series or None")
                if not isinstance(models,list):
                    raise ValueError(f"multi_config_fn returned {type(models)} instead of list")
                for indx,m in enumerate(models):
                    if not isinstance(m,(Model,type_none)):
                        raise ValueError(f"{multi_config_fn.__name__} returned {type(m)} at index {indx} instead of Model or None")
                all_models += models
            else:
                sim_obs  = super().compute_simulated_observations(param,use_cache=use_cache)
                model = self.get_model(param) # will have a model now (which might be just created)
                all_models += [model] # getting key (which might return None
            all_sim_obs.append(sim_obs) # append the obs.


        # got all observations we need so can compute mean.
        if any(sim_obs is None for sim_obs in all_sim_obs):
            return None  # missing some data so can't compute mean. Calling level can deal with this.
        # make average
        sim_obs = pd.concat(all_sim_obs, axis=1).mean(axis=1).rename(name)  # average over ensemble members.
        # and update the cache.
        # verify models are models and not Nones.
        for model in all_models:
            if not isinstance(model, Model):
                raise ValueError(f"Model {model} is not a Model")

        self._logical_info.obs(name,sim_obs)
        self.models(name,all_models) # store all the models.
        return self._logical_info.obs(name)




    def update_obs(self,use_cache:bool = False) -> pd.DataFrame:
        """
        Reload observations for all logical names. This will recompute the simulated observations for all logical names.
        :param use_cache: If True use the cache. Setting use_cache to False (default) will force regeneration of obs all the way down to the underlying models.
        :return: dataframe of obs.
        """
        params = self._logical_info.parameters.values()
        obs=[]
        for param in params:
            sim_obs = self.compute_simulated_observations(param.to_dict(), use_cache=use_cache)
            if sim_obs is not None:
                obs.append(sim_obs)
        obs = pd.DataFrame(obs) # return dataframe

        return obs




    @classmethod
    def from_dict(cls, dct: dict) -> runSubmit:
        """
        Generate runSubmit object from a dict. Main difference from super class method
          is that it uses the keys in ._logical_info.models to get the model keys and
           then populates ._logical_info.models with the appropriate models in model_info
           Do this to reduce memory foot print so models in logical_info are the same as in model_index

        Note  to_dict is not needed as superclass version does what is needed for runSubmit. logicalInfo has its own to_dict
        :param dct: dict to be parsed
        :return:runSubmit
        """

        # handle legacy model_status if needed
        model_status = dct.pop('model_status',{}) # use this to set model_status directly.
        obj: runSubmit = super(runSubmit, cls).from_dict(dct)  # create the runSubmit object
        # deal with legacy when have no _logical_info.parameter set. In this case we just iterate over the models in model_index and set the parameters to those in the model.
        if '_logical_info' not in dct:
            my_logger.warning('legacy change: adding existing models to _logical_info.parameters')
            for model in obj.model_index.values():
                name, param = obj._logical_info.params(model.parameters)  # add the parameters to logical_info
                obj._logical_info.model_keys[name] = [obj.key_for_model(model)]  # add the model to logical_info
                obj.compute_simulated_observations(model.parameters)  # compute the simulated observations for this model.


        for key,status in model_status.items(): # if model_status was not found then the dict will be empty so nothing will happen.
            try:
                model = self.model_index[key]
                obj.set_model_used_status(model,status) # CONSIDER changing name from status as that could confuse.
                my_logger.warning(f"Updating model {model} to status:{status}")
            except KeyError:
                my_logger.warning(f"Failed to find {key} in model_index. Skipping")
        # Set model_used_status to 'unknown' for any models that do not have a status set.
        for model in obj.model_index.values():
            obj.model_used_status(model,'unknown',update=False) # set model status to 'unknown' if no model status is set.

        return obj

    def update_params(self, update_parameters: list[str]) -> dict[str, str]:
        """
        Update in-place the parameters in logical_info for all logical names
        Calls super_class method and then handles logical_info
        :param update_parameters: list of parameter names to update.
        :return: Key mapping from old to new keys
        """



        key_mapping = super().update_params(update_parameters)  # call the super  class method.
        # key mapping is map from old keys in model_index to new ones in model_index.
        # Will need this to update ._logical_info.model_keys.
        names = list(self._logical_info.names.values())
        for name in names:  # sort out the logical info
            if name in self._logical_info.model_keys:
                self.update_logical_params(name, update_parameters,key_mapping)

        return key_mapping # return the key mapping in case it is needed.

    def _xxxxcopyConfig(self, direct: pathlib.Path,
                   extra_files: typing.Optional[list[pathlib.Path]] = None,
                   keep_list: typing.Optional[list[pathlib.Path]] = None,
                   new_config_name: typing.Optional[str] = None,
                   update_paths: bool = True) -> runSubmit:
        """
        Copy the config to a new directory. Uses super class method and then updates LogicalInfo. No longer needed
        :param direct: directory to copy to.
        :param extra_files: extra files to copy.
        :param keep_list: list of files to keep in the new directory.
        :param update_paths: if True update paths in config to point to new directory.
        :return: new runSubmit object with config copied to new directory.

        Not needed any more as copyConfig base class does what is needed.
        """
        # if extra_files is None:
        #    extra_files = []
        # extra_files += [self.config.config_path.relate_to(self.rootDir)]  # always copy config file.
        new_run_submit = super().copyConfig(direct, extra_files, keep_list=keep_list, update_paths=update_paths,new_config_name=new_config_name)

        return new_run_submit

    def create_model(self, params: dict,
                     dump: bool = True) -> typing.Optional[Model]:
        """
        runSubmit version of create_model. Call super class and set model_status
        :param params: dict of parameters to create the model.
        :param dump: if True dump the model to disk
        :param reference_name: Reference name for model
        :return: newly created model. If failed to create model will raise optclim_exceptions.submitModel
        """
        # call the super class.
        model = super().create_model(params, dump=dump)
        if model is  None:  # might get None if stop is set or something went wrong.
            return model # just return None
        else:
            self.model_used_status(model, 'called')
            # Check model.status is as expected.
            allowed_status = ['CREATED','INSTANTIATED','PROCESSED']
            if model.status not in allowed_status:  # # not processed/Instantiated/Created so raise ValueError and complain.
                raise ValueError(f"{model} status not in {allowed_status} but is {model.status}")
            # model is created or instantiated then controlling algorithm deals with it.

        return model

    def read_model_configs(self, path_list: list[pathlib.Path]) -> list[Model]:
        """
        Reads model configs from a list of paths. Sets status to read
        :param path_list: List of paths
        :return: list of models.
        """

        models = super().read_model_configs(path_list)

        for model in models:
            self.model_used_status(model, 'read')

        return models


    def reset_models_used(self) -> list[Model]:
        """
       Reset models that had model_used_status 'called' (and so used) to 'not_called'
        :return: List of models that were reset.
        """
        models_called = [model for model in self.model_index.values() if self.model_used_status(model) == 'called']
        if len(models_called) >0 :
            my_logger.info(f"Setting {len(models_called)} used_status to not_called")
            for model in models_called:
                self.model_used_status(model, 'not_called')

        return models_called

    type_model_used_status = typing.Literal[
        'initial', 'unknown', 'called', 'read', 'not_called']  # allowed status for model_status
    def model_used_status(self, model: Model,
                          status: typing.Optional[type_model_used_status] = None,
                          update:bool = True) -> typing.Optional[type_model_used_status] :
        """
        Set/get  model_status
        :param model: A model object
        :param status: One of 'initial','called','not_called','unknown','read' or None.
         If None then only the current value of model.study_properties['model_status'] will be returned
        :param update -- if False then properties will be set only if they do not exist.
        :return: Nada

        Design notes -- this could be a model method, but the idea behind study_properties is that studies set values and
          know about keys. So this is here in SubmitStudy.
        """

        if status is not None:
            allowed_status = ['initial', 'called', 'not_called', 'unknown', 'read']
            if status not in allowed_status:
                raise ValueError(f"Invalid status {status}. Must be one of {' '.join(allowed_status)}")
            if update or  ('model_status' not in model.study_properties): # updating or model_status not present in study_properties
                model.study_properties['model_status'] = status


        return model.study_properties.get('model_status')


    def check_deterministic(self, error: genericLib.error_handle_types = 'fail') -> bool:
        """
        Check that the model runs are deterministic. Done by using model_used_status to see if any are  not_called
        :param error: Used to in call to genericLib.error_handle to determine whether to raise an error, warn or ignore.
          See genericLib.error_handle for allowed values and what is done.
        :return: True if deterministic, False otherwise.
        """
        # check no models have status not_called
        uncalled_models = [str(model) for model in self.model_index.values() if self.model_used_status(model) == 'not_called']  # list of uncalled models
        if uncalled_models:
            message = f'Have {len(uncalled_models)} models not called. Unused models are:\n' \
                      + "\n".join(uncalled_models)
            genericLib.error_handle(message, error)

        return len(uncalled_models) == 0  # return True if no missing keys, False otherwise.


    def stdFunction(self, params: np.ndarray,
                    df: bool = False,
                    raiseError: bool = True,
                    ensemble_average: bool = True,
                    transform: typing.Optional[pd.DataFrame] = None,
                    scale: bool = False,
                    residual: bool = False,
                    sumSquare: bool = False) -> typing.Union[np.ndarray, pd.DataFrame]:
        """
        Standard Function used for running model. Returns values from cache if already got it.
          If not got cached model then raises runModelError exception or returns NaN
           after generating new model.


        It is a method rather than a class function as it makes use of some information in the object.
        So to actually get used as a function it needs to be converted to a function. This is done by genOptFunction which
          uses partial.function to do it. But this function does all the work.

        :param params -- a numpy array with the parameter values.
                These parameters should  be ordered as in self.paramNames()
        :param df  -- If True return all observations (read in after processing) as a dataframe.
        :param raiseError  -- If True raise optclim_exceptions.submitModel  if any requested obs are None.
                This should cause generation & submission of models that need running.
                Else return NaN for any obs that are None.
        :param ensemble_average -- If True average the ensemble members.

        The four  parameters below they are applied in the order: scale, residual, transform, sumSquare.
        The first three are handled in comp_logical_info.observations and occur after any ensemble averaging.
        :param scale   -- if True scale observations  by self.scales()

        :param residual  -- if True remove target (self.target()) from observations

        :param transform  -- if provided transform the model observations by matrix multiplying them by this matrix.
                  It should be  N*nobs where nobs are the number of sim ons and  0 <= N <= nobs.
                  One application of this is to transform the data into a basis of eigenvectors of an
                    error covariance matrix. The transform matrix should be provided as a pandas datarray.
                  Column names being the observations names. Index being sensible labels for rows which will be new observations names.
                  Do be careful to make sure transform has same scaling as here...

        :param sumSquare (default False) -- if True return the sum of squares of the observations after any processing.


        Using stdFunction -- this  is a method as it needs to know various bits of information contained in ModelSubmit..
        To actually use it with optimisation function you need to call runSubmit.genOptFunction(**kwargs).
        That will give you a function suitable for many optimisation functions.  If you have more complex needs you
          likely need to add the runSubmit object to the list of arguments and then do runSubmit.stdFunction(.... )
          This might require you to create  a partial function. Your life will probably be easier if you set df=True
          and work with dataframes in your function.
        """

        paramNames = self.config.paramNames()
        # params can be a 2D array...
        if params.ndim == 1:
            use_params = params.reshape(1, -1)

        elif params.ndim == 2:
            use_params = params

        else:
            raise Exception("params should be 1 or 2d ")
        nsim = use_params.shape[0]  # no of simulations we want to do which is the no of parameter sets provided
        nparams = use_params.shape[1]  # no of params only used to check that parameter array is as expected.
        if nparams != len(paramNames):
            raise ValueError(
                "No of parameters %i not consistent with no of varying parameters %i\n" % (nparams, len(paramNames)) +
                "Params: " + repr(use_params) + "\n paramNames: " + repr(paramNames))

        # column names affected by transform so use that if provided
        if transform is not None:
            if len(transform.index) == 0:
                print("Transform is \n", transform)
                raise ValueError("Transform has 0 len index. Fix your covariance. Exiting")
            obsNames = transform.index
        else:
            obsNames = self.config.obsNames()
        nObs = len(obsNames)  # How many observations are we expecting?
        if nObs == 0:  # Got zero. Something gone wrong
            raise ValueError("No observations found. Check your configuration file ")

        result = []  # empty list. Will fill with series from analysis and then make into a dataframe.
        empty = pd.Series(np.repeat(np.nan, nObs), index=obsNames)
        params = [dict(zip(paramNames, use_params[indx, :])) for indx in range(0, nsim)] # list of param dicts

        # deal with no ensemble averaging.
        n_ensemble = self.config.ensembleSize()
        if ensemble_average is False and n_ensemble >1:
            my_logger.info(f"No ensemble averaging and n_ensemble={n_ensemble}. Generating ensemble list.")
            all_params = []
            for param in params:
                all_params += [ param | dict(ensembleMember=ens_index)  for ens_index in range(0, n_ensemble)]
            params = all_params # rename it.

        for param in params:  # iterate over the simulations.
            observations = self.compute_simulated_observations(param)

            if observations is not None:
                observations = self.transform_check(observations, transform=transform, scale=scale, residual=residual)
                # compute the cost which uses the transformed data.
                #self._logical_info.cost[observations.name] = (observations ** 2).mean()  # store the mean square diff.
            result.append(observations)

        sim_obs = pd.DataFrame([empty if r is None else r for r in result])  # replace None with nans & convert to a df.
        all_new = all([r is None for r in result])
        if all_new:  # all models are new so want a new iteration
            self._logical_info.completed_iteration()  # mark end of iteration
        all_ran = all([r is not None for r in result])
        if not all_ran and raiseError:
            # want to raise error if any of result is None meaning at least one model needs to be ran.
            raise optclim_exceptions.submitModel

        if sumSquare:
            sim_obs = (sim_obs ** 2).sum(axis=1)

        if not df:  # want it as values not  a dataframe
            sim_obs = np.squeeze(sim_obs.values)

        return sim_obs

    def genOptFunction(self, **kwargs):
        """

        :return: a function suitable for use in an optimisation  algorithm (or anything else that uses the framework).
        by generating a partial function which converts stdFunction from a method to a function in the argument list
        and any adds extra args (as named arguments) that are present. You can use this to give you something suitable
        for most optimisation methods or use your own function to wrap stdFunction.


        """

        fn = functools.partial(self.stdFunction, **kwargs)

        return fn

    def delete(self):
        """
        runSubmit version of delete. Calls super class delete and then resets logical_info and model_status to empty.
        :return:
        """
        super().delete()
        self.reset_models_used()

    def plot(self,
             fname: typing.Optional[pathlib.Path] = None,
             fig_name: str = 'monitor',
             cost: typing.Optional[pd.Series] = None,
             obs: typing.Optional[pd.DataFrame] = None,
             params: typing.Optional[pd.DataFrame] = None,
             savefig_kwargs: typing.Optional[dict] = None, ) -> \
            typing.Optional[tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]]:

        """
        Calls superclass plot using, by default, logical values
        :return: fix and axes from superclass plot method. See that for details. Will return None if data missing.
        """
        # obsNames = self.config.obsNames()
        if cost is None:
            cost = self.logical_cost()
        if obs is None:
            obs = self.simulated_observations(scale=True, normalize=True).dropna()
        if params is None:
            params = self.logical_params(normalize=True)
        # call the super class plotter to actually plot
        fig_axs = super().plot(fname=fname, fig_name=fig_name,
                                cost=cost, obs=obs, params=params,
                                savefig_kwargs=savefig_kwargs)
        return fig_axs

    def runOptimized(self, stop: bool = False) -> StudyConfig:
        """
        :arg self
        Run optimised case using reference model which may be potentially different from configuration used to optimize.
        Cares about number of ensemble members (default is 1 if not set) and the optimum parameters which should be set.
        """

        # basic idea is that users takes final json file and edits it to suit their needs
        # a better approach might be to pass in the optimum values and a list of configurations to loop over
        # as use case is running optimised configurations in different reference cases.
        # this would be quite a long way from usual framing and might need a bit of engineering.
        # Want to benefit from Submits caching. But Submit assumes config defined at creation..
        # but need to embed it along with each model... That could be done through adding an option to
        # stdfunction to include a configuration overriding whatever is in the Submit method.
        # Then this code would loop over those configurations calling stdFunction with them.
        # Turn of raising error in stdFunction and then check for null in output. If so raise runModelError.
        # Also need to modify fixedParams (which are no longer fixed) or at least are overwritten
        # Actually coming to conclusion that this case is pushing Submit too far and better
        # to do raw model create and modification as already done. That script has access to
        # the final config (optimum param values, nEns) and n configurations.  Trick is
        # not running more than once... Which is what Submit gives...
        # Hard will come back when I actually have a need!
        if stop:
            raise NotImplementedError("runOptimized stop not implemented as have no use case")
        start = self.config.optimumParams()
        modelFn = self.genOptFunction(df=True)
        obsSeries = modelFn(start.values).squeeze()  # if the simu
        # should run ensemble avg etc as side effect...and if things don't exist raise modelError
        # now to set up information having done all cases because if we have got here modelError has not been raised.
        finalConfig = self.runConfig()  # get final runInfo
        finalConfig.beginParam(start)  # setup the begin values!
        finalConfig.optimumParams(None)
        finalConfig.best_obs(best_obs=obsSeries)
        return finalConfig

    def rangeAwarePerturbations(self, baseVals: pd.Series, parLimits: pd.DataFrame, steps: pd.Series) -> pd.Series:
        """
        Generate perturb param values  towards the centre of the valid range for each
        parameter. Used in runJacobian

        Inputs

            arg: baseVals: parameters at which Jacobian computed at.
            arg: parLimits: dataset max,min limits. parLimits.loc[:,'maxParm'] are the max values;
              parLimits.loc[:,'minParam'] are the min values;
            arg: steps: stepsizes for each parameter.

        Inputs, except parLimits, are all pandas series
        Returns pandas series array defining the perturbed parameter values
        """

        my_logger.debug(f"in rangeAwarePerturbations with {baseVals}")
        # derive the centre-of-valid-range
        centres = (parLimits.loc['minParam', :] + parLimits.loc['maxParam', :]) * 0.5
        deltaParam = np.abs(steps)
        sgn = np.sign((centres - baseVals))
        L = (sgn < 0)
        deltaParam[L] *= -1

        return deltaParam

    def runJacobian(self, scale: bool = False, stop: bool = False) -> StudyConfig:
        """
        run Jacobian cases.
        Rather crude (first order accurate) estimate. Evaluate functions at
          optimum values +/- delta where delta is specified in the StudyConfig used to generate Submit
          +/- is to keep perturbations towards the centre pf the domain.
        A more accurate version would run 2nd order differences around the base point.
        But (as yet) no need for this so not implemented. Probably best done with a centred option.
        The Jacobian computed is the transformed Jacobian.

        :arg self -- a Submit object.
        :param scale -- If True apply scalings.
        :returns a configuration. The following methods should work on it:
                return: finalConfig -- a studyConfig. The following methods should give you useful data:
                finalConfig.transJacobian() -- the  transformed Jacobian matrix at the optimum pt
                finalConfig.hessian() -- the  hessian computed from J^T J at the optimum pt.

            runs runConfig to provide generic info. (See documentation for that)

        """
        # TODO add maxfun which probably means outer loop is over ensemble members
        # rather than over parameters.
        if stop:
            raise NotImplementedError("runJacobian stop not implemented as have no use case")
        configData = self.config
        Tmat = configData.transform_matrix(scale=scale)
        modelFn = self.genOptFunction(raiseError=True, df=True, residual=True, transform=Tmat, scale=scale)
        base = configData.optimumParams()  # try with optimum parameters
        if base is None:  # if none go with the begin parameters.
            base = configData.beginParam()

        paramRanges = configData.paramRanges()
        steps = configData.steps()  # see what the steps are
        delta = self.rangeAwarePerturbations(base, paramRanges, steps)  # compute the actual deltas
        params = [base]  # list of parameter values to run at -- including the base. Needed to compute jac

        for p, v in delta.items():  # iterate over parameters
            param = base[:].rename(p)
            param.loc[p] += delta.loc[p]
            params.append(param)

        params = pd.DataFrame(params)  # all the params together to maximize parallelism
        # check params all in range.
        if np.any(params > paramRanges.loc['maxParam', :]):
            print("Some parameters too large\n ", paramRanges.loc['maxParam', :],
                  "\n", params)
            print(params > paramRanges.loc['maxParam', :])
            raise ValueError
        if np.any(params < paramRanges.loc['minParam', :]):
            print("Some parameters too small\n ", paramRanges.loc['minParam', :],
                  "\n", params)
            print(params < paramRanges.loc['minParam', :])
            raise ValueError
        observations = modelFn(params.values)  # compute the observations where we need to. This may generate model simulations
        dobs = observations.iloc[1:, :] - observations.iloc[0, :]
        dobs = dobs.set_index(delta.index)
        jac = dobs.div(delta, axis=0).T  # compute the Jacobian
        finalConfig = self.runConfig(scale=scale, transJacobian=jac)  # get the configuration.
        return finalConfig

    ## TODO make runDFOLS its own class.

    # handy fns for runDFOLS. Once DFOLS has its own class these can be turned into methods.
    def dfols_eval_db(self,
                      scale: bool = True) -> typing.Optional[dfols.EvaluationDatabase]:
        """
        Create an EvaluationDatabase object for DFOLS using the current logical observations and costs.
        See:
           https://numericalalgorithmsgroup.github.io/dfols/build/html/userguide.html#using-initial-evaluation-database

        Uses self.config.dfols_config()['evaluation_database'] which should contain:

         parameters: Path to csv file of previous parameter values. Header should be param names, col 0 names
         simulated_observations: Path to csv file of previous simulated observations. Header should be observations names, col 0 names
         start_index: If specified then index of first row to use from csv files.
           If None then use initParams. If minimum use minimum value from database.

        :param scale: if True (default) apply scaling.
        :return: dfols.EvaluationDatabase object or None if self.config.dfols_config()['evaluation_database'] not found
        """
        param_names = self.config.paramNames()
        obs_names = self.config.obsNames()
        trans_matrix_kw = self.config.get('Covariances',{}).get('transform_matrix',{})
        if trans_matrix_kw is None:
            trans_matrix_kw = {}
        transform = self.config.transform_matrix(scale=scale,**trans_matrix_kw)
        eval_config = self.config.DFOLS_config().get('evaluation_database')
        if eval_config is None:
            return None
        my_logger.info("Using evaluation database")
        # get the parameter values by reading from csv file and reindex to get wanted params in right order.
        params = pd.read_csv(self.expand(eval_config['parameters']), index_col=0).reindex(param_names, axis=1)
        if params.isnull().any().any():  # check params
            raise ValueError("Some parameters in evaluation database are missing. Check parameter names.")
        my_logger.debug(f"parameters shape: {params.shape} ")
        observations = pd.read_csv(self.expand(eval_config['simulated_observations']), index_col=0).reindex(obs_names, axis=1)
        if observations.isnull().any().any():  # check observations
            raise ValueError("Some observations in evaluation database are missing. Check observation names.")
        my_logger.debug(f"Simulated_obs shape: {observations.shape} ")
        trans_obs = [self.transform_check(o, transform=transform, scale=scale, residual=True) for name, o in
                     observations.iterrows()]
        trans_obs = pd.DataFrame(trans_obs)
        if trans_obs.isnull().any().any():  # check transformed observations
            raise ValueError(
                "Some transformed observations in evaluation database are NaN. Check transform matrix etc.")

        # Work out how to start dfols.
        how_to_start = eval_config.get('start_index', None)
        my_logger.debug(f"start index: {how_to_start}")

        init_params = self.config.beginParam()  # init params to match against if needed.
        if how_to_start is None:  # use initParams
            mask = (params == init_params).all(axis=1)  # maybe do close match???
            # this will trigger an error in params differ. This is wanted behaviour!
            idx = mask[mask].index[0] if mask.any() else None  # index of init_params in params

        elif how_to_start == 'minimum':  # use minimum cost from database
            cost = (trans_obs ** 2).sum(axis=1)
            idx = cost.idxmin()
        else:  # use specified index
            idx = how_to_start

        # check that idx is in params (if not None)
        if (idx is not None) and (idx not in params.index):
            raise ValueError(f"Index {idx} not found in parameters")

        # now create the evaluation database
        eval_db = dfols.EvaluationDatabase()
        count_start = 0
        for index, row in params.iterrows():
            if index == idx:
                count_start += 1
                if count_start > 1:
                    raise ValueError(f"Start index {idx} found multiple times in evaluation database")
                my_logger.info(f"DFOLS evaluation database: starting evaluation at index {index}")
            eval_db.append(row.values, trans_obs.loc[index].values, make_starting_eval=(index == idx))
        # check that we found the starting evaluation
        if idx is None:  # No starting evaluation
            my_logger.info("DFOLS evaluation database: no starting evaluation found; using default start")
            eval_db.append(row.values, None, make_starting_eval=True)
        elif count_start != 1:
            raise ValueError(f"Start index {idx} not found in evaluation database")
        else:  # we are OK and nothing to do.
            pass

        return eval_db

    ## end of eval_db function

    def dfols_write_final_config(self, solution, scale: bool = True):
        """
        Write out the final configuration after DFOLS has completed.
        Uses runConfig to generate the configuration and then adds in DFOLS solution information.
        Saves the configuration to a json file.
        :param self:
        :param solution: dfols solution object
        :param scale: scaling or not
        :return: final configuration saved to file.
        """
        tMat = self.config.transform_matrix(scale=scale)
        var_param_names = self.config.paramNames()
        filename = self.rootDir / (self.config.fileName().stem + "_final.json")  # final config
        best = pd.Series(solution.x, index=var_param_names)  # best soln from DFOLS
        # now compute jacobian and wrap it up as a dataframe.
        jacobian = solution.jacobian
        jacobian = pd.DataFrame(jacobian, columns=var_param_names, index=tMat.index)
        # create the final configuration
        finalConfig = self.runConfig(scale=scale, add_cost=True, filename=filename, transJacobian=jacobian,
                                     best=best)  # get final runInfo
        solution.diagnostic_info.index = range(0, solution.diagnostic_info.shape[0])
        finalConfig.dfols_solution(solution=solution)
        finalConfig.save()  # save the config
        return finalConfig

    def runDFOLS(self, scale=True, stop=False) -> OptClimConfigVn3:
        """
        run DFOLS algorithm. It runs until new models need to be ran or DFOLS complets.
                If new models are needed then runModelError will be raised.
                DFOLS raises np.linalg.linalg.LinAlgError if it tries to do calculations with nan (which is what it gets
                when model has not been ran). This then triggers a runModelError. The callee of this method
                should trap that error and then run the necessary models using Submit.submit.

        :param scale (default True). If True scale data for transform matrix and in calculations of observations
        :param stop If True stop the algorithm by setting maxfn to 1+current no of logical observations.

        See StudyConfig.scalings()
        It should not make any difference if scale is True or False. But if regularisation is done
        then it is better to have all (diagonal) elements of the covariance matrix have roughly the same mag
        which is really what scaling does for you. (for example converting kg/sec/m^2 to mm/day). Current implementation
        of tranMatrix truncates removing all eigenvectors and eigenvalues when egivenvalues < 1E-6 * max(evalues)

        return: finalConfig -- a studyConfig. The following methods should give you useful data:

                # Generic stuff (that is probably more useful)
                finalConfig.transJacobian() -- the final transformed Jacobian matrix at the optimum pt
                finalConfig.hessian() -- the final hessian computed from J^T J at the optimum pt.
                finalConfig.get_dataFrameInfo('diagnostic_info') -- Diagnostic info from DFOLS. See DFOLS documentation.
                finalConfig.optimumParams() -- optimum parameters.

            also can get generic info and cost info:
            runs runCost & runConfig to provide  info. (See documentation of those methods for what they provide)

        """

        rng_seed = 123456  # will potentially run dfols twice so can get intermediate results from it.
        random.seed(rng_seed)  # make sure rng as used by DFOLS takes same values every time it is run.
        configData = self.config
        var_param_names = configData.paramNames()
        dfols_config = configData.DFOLS_config()

        # deal with Evaluation database
        x0 = self.dfols_eval_db(scale=scale)  # will return evaluation database if key exists or None if it doesn't
        if x0 is None:  # use beginParam to get intial values
            x0 = configData.beginParam(paramNames=var_param_names).values  # initial parameter values
        # Sensible defaults  for DFOLS -- which can be overwritten by config file
        userParams = {'logging.save_diagnostic_info': True,
                      'logging.save_xk': True,
                      'noise.quit_on_noise_level': True,
                      'general.check_objfun_for_overflow': False,
                      'init.run_in_parallel': False,  # run in parallel
                      'interpolation.throw_error_on_nans': True,  # make an error happen.
                      'restarts.throw_error_on_nans': True,  # ALSO make an error happen.
                      }

        prange = configData.paramRanges(paramNames=var_param_names)
        prange = (prange.loc['minParam', :].values, prange.loc['maxParam', :].values)
        # update the user parameters from the configuration.
        userParams = configData.DFOLS_userParams(userParams=userParams)
        # Check following params are True or not set.
        # interpolation.throw_error_on_nans & restarts.throw_error_on_nans
        not_set_true = []
        for param in ['interpolation.throw_error_on_nans', 'restarts.throw_error_on_nans']:
            value = userParams.get(param)
            if value is False:  # only raise an error if False. If None (because not set) or True then we are OK.
                not_set_true.append(param)  # add to list of bad params.
            userParams[param] = True  # force it to be true regardless

        if not_set_true:
            m = f'Set {" ".join(not_set_true)} parameters to True or do not set'
            raise ValueError(m)

        # setup transform matrix and optFn
        tMat = configData.transform_matrix(scale=scale)

        optFn = self.genOptFunction(transform=tMat, residual=True, raiseError=False, scale=scale)
        rhobeg = dfols_config.get('rhobeg', 1e-1)
        rhoend = dfols_config.get('rhoend', 1e-3)
        do_logging = dfols_config.get('do_logging', False)

        if stop:
            dfols_config['maxfun'] = len(self.logical_cost()) + 1
            self.update_history(f"DFOLS stopped with maxfun = {dfols_config['maxfun']}")
            # +1 allows cases that have been run but not added to logical observations/cost
            self.config.DFOLS_config(dfols_config)  # store modified config

        # Now actually run DFOLS
        try:
            with warnings.catch_warnings():  # catch the complaints from DFOLS about NaNs encountered...
                warnings.filterwarnings('ignore')  # Ignore all warnings...
                # TODO consider rewrtting this so a dict gets setup and passed in.
                solution = dfols.solve(optFn, x0, do_logging=do_logging,
                                       objfun_has_noise=True,
                                       bounds=prange, scaling_within_bounds=True,
                                       maxfun=dfols_config.get('maxfun', 100),
                                       rhobeg=rhobeg,
                                       rhoend=rhoend,
                                       user_params=userParams)

        except np.linalg.linalg.LinAlgError: # time to submit new models.
            n_inst_models = len(self.models_to_instantiate())
            my_logger.info(f"Have just generated {n_inst_models} to instantiate")
            neval = len(self.logical_cost())
            if (neval > 1):  # got some evaluations.
                # Run DFOLS again with reduced number of fn evals to provide some diagnostic info.
                
                my_logger.debug('Running DFOLS again with reduced number of fn evals to get diagnostic info')
                random.seed(rng_seed)  # reset rng seed back to first value.
                with warnings.catch_warnings():  # catch the complaints from DFOLS about NaNs encountered...
                    warnings.filterwarnings('ignore')  # Ignore all warnings...
                    # catch created models (generated on first tiem around). Occurs becuase have new runs mixed in with old runs.
                    try:
                        solution = dfols.solve(optFn, x0, do_logging=False,
                                               objfun_has_noise=True,
                                               bounds=prange, scaling_within_bounds=True,
                                               maxfun=len(self.logical_cost()),  # should get it to terminate.
                                               rhobeg=rhobeg,
                                               rhoend=rhoend,
                                               user_params=userParams)
                        # this will give diagnostic info from DFOLS now to use it.
                        # need to wrap the best sol and put in other information into the final results file.
                        finalConfig = self.dfols_write_final_config(solution, scale=scale)
                    except optclim_exceptions.useCreatedModel as e:
                        my_logger.warning(f"Trying to run a created model {e}") 
            raise optclim_exceptions.submitModel("dfols failed with lin alg error")
            # this is how DFOLS tells us it got NaN which then triggers running the next set of simulations.

        # Code here will be run when DFOLS has completed.
        # It mostly puts stuff in the final JSON file so can easily be looked at for subsequent analysis.
        if solution.flag not in (solution.EXIT_SUCCESS, solution.EXIT_MAXFUN_WARNING):
            raise ValueError(f"dfols failed with flag {solution.flag} error : {solution.msg}")

        else: # suceeded.
            print(f"dfols completed  with flag {solution.flag}: {solution.msg}")

        finalConfig = self.dfols_write_final_config(solution, scale=scale)

        return finalConfig

    def runGaussNewton(self, verbose=False, scale=True, stop: bool = False) -> OptClimConfigVn3:
        """

        param: verbose if True produce more verbose output.
        param: scale if True apply scaling (default is True)
        Run Gauss Newton algorithm as used in Tett et al, 2017.
        return: finalConfig -- a studyConfig. The following methods should give you useful data:
                finalConfig.GNparams() -- the best parameters (min error)
                finalConfig.GNcost() -- the cost for each function evaluation
                finalConfig.GNalpha() -- the values of alpha used for next optimisation in the linesearch
                finalConfig.GNhessian() -- the diagnosed Hessians (in transformed space) at each iteration
                # Generic stuff (that is probably more useful)
                finalConfig.transJacobian() -- the final transformed Jacobian matrix at the optimum pt
                finalConfig.hessian() -- the final hessian computed from J^T J at the optimum pt.
                finalConfig.optimumParams() -- optimum parameters.

            also can get generic info and cost info:
            as runs runCost & runConfig to provide  info. (See documentation of those methods for what they provide)

        """
        import Optimise
        if stop:
            raise NotImplementedError("runGaussNewton stop not implemented as have no use case")

        # extract internal covariance and transform it.
        configData = self.config
        optimise = configData.optimise().copy()  # get optimisation info
        intCov = configData.Covariances(trace=verbose, scale=scale)['CovIntVar']
        # Scaling done for compatibility with optFunction.
        # need to transform intCov. errCov should be I after transform.
        tMat = configData.transform_matrix(scale=scale)
        intCov = tMat.dot(intCov).dot(tMat.T)
        # This is correct-- it is the internal covariance transformed
        optimise['sigma'] = False  # wrapped optimisation into cost function.
        optimise['deterministicPerturb'] = True  # deterministic perturbations.
        paramNames = configData.paramNames()
        nObs = tMat.shape[
            0]  # might be a smaller because some evals in the covariance matrix are close to zero (or -ve)
        start = configData.beginParam(paramNames=paramNames)
        optFn = self.genOptFunction(transform=tMat, scale=scale, residual=True, raiseError=True)
        # TODO have maxfun which limits the number of fn evaluations.
        best, status, info = Optimise.gaussNewton(optFn, start.values,
                                                  configData.paramRanges(paramNames=paramNames).values.T,
                                                  configData.steps(paramNames=paramNames).values,
                                                  np.zeros(nObs), optimise,
                                                  cov=np.identity(nObs), cov_iv=intCov, trace=verbose)
        filename = self.rootDir / (self.config.fileName().stem + "_final.json")  # final config file name
        jacobian = pd.DataFrame(info['jacobian'][-1, :, :].T, columns=paramNames, index=tMat.index)
        best = pd.Series(best, index=paramNames, name=self.config.name())  # wrap best result as pandas series
        finalConfig = self.runConfig(scale=scale, add_cost=True, filename=filename,
                                     best=best, transJacobian=jacobian)  # get final runInfo
        finalConfig.GNstatus(status)
        # Store the GN specific stuff. TODO consider removing these and just store the info.
        finalConfig.GNparams(info['bestParams'])
        finalConfig.GNcost(info['err_constraint'])
        finalConfig.GNalpha(info['alpha'])

        print("status", status)

        return finalConfig

    ## run_params
    def run_params(self, ensemble_average: bool = True, scale: bool = True, stop: bool = False) -> OptClimConfigVn3:
        """
        Run the model for a set of parameters specified in the configuration file.
        This is a simple run of the model for a set of parameters. It does not do any optimisation.

        It does not do any scaling, residual or transform. It just runs the model(s) and returns the simulated observations.
        The parameters to use are specified in the configuration file.
        The fixed parameters are also specified in the configuration file.
        :param ensemble_average -- if True (default) and if ensemble size > 1 then average the ensemble members.
        :return: finalConfig -- a studyConfig. The following methods should give you useful data:
                finalConfig.observations() -- the observations

        """
        if stop:
            raise NotImplementedError("run_params stop not implemented as have no use case")

        params_dir = self.config.optimise()  # get the parameters to run
        params = self.get_parameters(params_dir)  # convert to dataframe

        observations = self.stdFunction(params.values, df=True, raiseError=True, ensemble_average=ensemble_average,
                               scale=scale)

        filename = self.rootDir / (self.config.fileName().stem + "_final.json")  # final config file name
        final_config = self.runConfig(add_cost=False, filename=filename)  # get final runInfo

        return final_config

    def get_parameters(self, dict_in: dict) -> pd.DataFrame:
        """
        Convert a dictionary of parameters to a pandas DataFrame.
        :param dict_in: dictionary of parameters
           Uses the following keys:
            - parameters: list of dictionaries of parameters
            - index: optional list of index values for the DataFrame
            - scale: optional boolean to indicate if parameters should be scaled to their ranges.
              Default is False.
        uses self.config to get standard parameters (to fill missing with) and parameter ranges (if scale set).
        :return: pandas DataFrame of parameters
        """

        param_list: list[dict] = dict_in['parameters']

        # check keys match
        keys = set(param_list[0].keys())
        for d in param_list:
            if set(d.keys()) != keys:
                raise ValueError("All dictionaries in the list must have the same keys.")

        # convert to DataFrame
        index = dict_in.get('index')
        params = pd.DataFrame(param_list, index=index)
        # apply scaling if needed
        param_names = self.config.paramNames()
        params = params.reindex(columns=param_names)
        param_range = self.config.paramRanges(paramNames=param_names)  # get param range
        if dict_in.get('scale', False):
            params = params * param_range.loc['rangeParam', :] + param_range.loc['minParam', :]

        # fill missing values with standard values
        std = self.config.standardParam(paramNames=param_names)
        params = params.fillna(std)
        # check params are within ranges
        L = (params < param_range.loc['minParam', :]) | (params > param_range.loc['maxParam', :])
        if L.any().any():
            raise ValueError(f"Parameters out of range:\n{params[L]}")
        return params

    def runPYSOT(self, scale=True, stop: bool = False) -> OptClimConfigVn3:

        """
        Run PYSOT algorithm

        Not been ran or tested. Likely needs various things set up to make it work..
        :params scale -- if True scale data internally by scalings.

        :returns finalConfig (a studyConfig)
            from which you can get generic info and cost info:
            See documentation of runCost & runConfig methods  to see what they provide.
        """

        # pySOT -- probably won't work without some work. conda install conda-forge pysot will install it.
        import pySOT
        raise NotImplementedError('pysot not well implemented. ')
        if stop:
            raise NotImplementedError("runPYSOT stop not implemented as have no use case")
        configData = self.config
        optimise = configData.optimise().copy_files()  # get optimisation info
        tMat = configData.transform_matrix()
        optFn = self.genOptFunction(transform=tMat, residual=True)  # need scale??
        paramNames = self.paramNames()
        from pySOT.experimental_design import SymmetricLatinHypercube
        from pySOT.strategy import SRBFStrategy, DYCORSStrategy  # , SOPStrategy
        from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail, \
            SurrogateUnitBox  # will not work anymore as SurrogateUnitBox not defined.
        from poap.controller import SerialController
        from pySOT.optimization_problems import OptimizationProblem
        # Wrapper written for pySOT 0.2.2 (installed from conda-forge)
        # written by Lindon Roberts
        # Based on
        # https://github.com/dme65/pySOT/blob/master/pySOT/examples/example_simple.py
        # Expect optimise parameters:
        #  - maxfun: total number of evaluations allowed, default 100
        #  - initial_npts: number of initial evaluations, default 2*n+1 where n is the number of variables to optimise
        pysot_config = optimise.get('pysot', {})

        # Light wrapper of objfun for pySOT framework
        class WrappedObjFun(OptimizationProblem):
            def __init__(self):
                self.lb = configData.paramRanges(paramNames=paramNames).loc['minParam', :].values  # lower bounds
                self.ub = configData.paramRanges(paramNames=paramNames).loc['maxParam', :].values  # upper bounds
                self.dim = len(self.lb)  # dimensionality
                self.info = "Wrapper to DFOLS cost function"  # info
                self.int_var = np.array([])  # integer variables
                self.cont_var = np.arange(self.dim)  # continuous variables
                self.dfols_residual_function = optFn

            def eval(self, x):
                # Return same cost function as DFO-LS gets
                residuals = self.dfols_residual_function(
                    x)  # i.e. if DFO-LS asked for the model cost at x, it would get the vector "residuals"
                dfols_cost = np.dot(residuals,
                                    residuals)  # sum of squares (no constant in front) - matches DFO-LS internal cost function
                return dfols_cost

        data = WrappedObjFun()  # instantiate wrapped objective function

        # Initial design of points
        slhd = SymmetricLatinHypercube(dim=data.dim, num_pts=pysot_config.get('initial_npts', 2 * data.dim + 1))

        # Choice of surrogate model (cubic RBF interpolant with a linear tail)
        rbf = SurrogateUnitBox(RBFInterpolant(dim=data.dim, kernel=CubicKernel(), tail=LinearTail(data.dim)),
                               lb=data.lb, ub=data.ub)

        # Use the serial controller (uses only one thread), SRBF strategy to find new points
        controller = SerialController(data.eval)
        strategy = pysot_config.get('strategy', 'SRBF')
        maxfun = pysot_config.get('maxfun', 100)
        if strategy == 'SRBF':
            controller.strategy = SRBFStrategy(max_evals=maxfun, opt_prob=data, exp_design=slhd, surrogate=rbf)
        elif strategy == 'DYCORS':
            controller.strategy = DYCORSStrategy(max_evals=maxfun, opt_prob=data, exp_design=slhd, surrogate=rbf)
        else:
            raise RuntimeError("Unknown pySOT strategy: %s (expect SRBF or DYCORS)" % strategy)

        # Run the optimization
        result = controller.run()

        # code here will be run when PYSOT has completed. It is mostly is to put stuff in the final JSON file
        # Gather key outputs: optimal x, optimal objective value, number of objective evaluations used
        xmin = result.params[0]
        fmin = result.value
        nf = len(controller.fevals)

        # need to wrap best soln xmin.
        best = pd.Series(xmin, index=paramNames)
        filename = self.rootDir / (self.config.fileName().stem + "_final.json")  # final config file name
        finalConfig = self.runConfig(scale=scale, add_cost=True, filename=filename,
                                     best=best)  # get final runInfo

        print("PYSOT completed")
        return finalConfig

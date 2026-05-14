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

type_model_status = typing.Literal[
    'initial', 'unknown', 'called', 'read', 'not_called']  # allowed status for model_status


class LogicalInfo(model_base):
    """ Class to hold logical information about parameters, observations etc. 

    This to be used in runSubmit to hold information about the logical obs, parameters and related information.
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
        obs: dict[str,pd.Series] -- dict of obs indexed by logical name.
        cost: dict[str,float] -- dict of cost indexed by logical name.
        models: dict[str,list[Model]] == dict of list of models indexed by logical_name
    
    """

    def __init__(self):
        super().__init__()

        self.iteration_count: int = 0  # count of iterations. Used in generating logical names.
        self.count_within_iteration: int = 0  # count of models within an iteration. Used in generating logical names.
        self.names: dict[str, str] = dict()  # dict of logical names indexed by key generated from parameters.
        self.parameters: dict[str, pd.Series] = dict()  # dict of logical parameters indexed by logical name.
        self.models: dict[str, list[Model]] = dict()  # dict of list of models indexed by logical_name.
        self.obs: dict[str, pd.Series] = dict()  # dict of obs indexed by logical name.
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
        It also stores the parameters in self._logical_parameters if they are not already there.
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
            # store parameters
            param_series = pd.Series(params).rename(name)  # these are the params that the algorithm varies.
            self.parameters[name] = param_series
            my_logger.info(f"Generated {name}")

        return name

    def update_params(self, name: str, params: pd.Series) -> str:
        """
        Update the parameters inplace for a given logical name.
        :param name: logical name.
        :param params: new parameters.
        :return: the new key (as a string)
        """
        orig_params = self.parameters[name].to_dict()  # original params
        key = self.key(orig_params)  # original key
        new_key = self.key(params.to_dict())
        old_name = self.names.pop(key)  # remove old key/name mapping
        if old_name != name:
            raise ValueError(f"Logical name {name} does not match stored name {old_name}")  # SHOULD NOT HAPPEN
        self.names[new_key] = name
        # now have param_values and store them.
        self.parameters[name] = params.rename(name)
        return key

    def to_dict(self) -> dict:
        """
        Convert logical info to a dictionary. Uses super class but models are stored as keys
        keys generated by run class method runSubmit.key_for_model(model).
         See runSubmit.from_dict which converts the keys back to models.
         This is not a particularly satisfactory design for the LogicalInfo class.
        :return:
        """

        dct = super().to_dict()
        model_dct = dct.get('models', {})  # get models dict (key is logical name, values are list of models)
        models_as_keys: dict[str, list] = {}
        for logical_name, model_list in model_dct.items():
            models_as_keys[logical_name] = [runSubmit.key_for_model(m) for m in model_list]
        dct['models'] = models_as_keys  # store the keys
        return dct

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
        if 'keys' in dct and 'models' not in dct:
            dct['models'] = dct.pop('keys')
            my_logger.warning(
                'LogicalInfo.from_dict -- converting keys to models -- please update saved data to new format')

        elif 'keys' in dct and 'models' in dct:
            raise ValueError("LogicalInfo.from_dict -- found both 'keys' and 'models' in dict -- cannot proceed")
        else:
            pass  # we are good!

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
        dct['models'] = model_dct  # replaced with updated dct.

        obj: LogicalInfo = super().from_dict(dct)

        # conversion of keys to models (by pointing by reference to models in model_list handled in runSubmit.from_dict
        return obj

    def keys_to_models(self, model_index: dict[str, Model],
                       update_models: bool = False) -> dict[str, Model]:
        """
        This is a support function to be called from runSubmit.from_dict after the LogicalInfo object has been created.
        It converts models stored as keys to Model objects using model_index OR updates the models.

        :param model_index: dict of models indexed by key.
        :return: Nada as models updated inplace.
        """
        for name, model_list in self.models.items():
            final_model_list = []
            for model_or_key in model_list:
                if update_models:
                    key = runSubmit.key_for_model(model_or_key)
                else:
                    key = model_or_key
                try:
                    final_model_list += [model_index[key]]  # this is a model
                except KeyError:
                    raise ValueError(f"Key {key} not found in model_index")
            # done dealing with models for this logical name.
            self.models[name] = final_model_list  # update models to be Model objects.

    # End of LogicalInfo class.


class runSubmit(SubmitStudy):
    # Has the following additional attributes over SubmitStudy (and Study)
    # _logical_info: LogicalInfo -- holds information about logical names, parameters, obs, cost and models.
    #  This is a private attribute as it is really only for use within this class. It is not intended to be used outside this class.
    # model_status: dict[str, type_model_status]  # dict of *model* statuses indexed by model_key.

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

        self.model_status: dict[str, type_model_status] = dict()  # allows checking of deterministic running.
        # set status for models in model_status.
        if models is not None:
            for model in models:
                self.set_model_status(model, 'initial')

    def make_model(self, params: dict, reference_name: typing.Optional[str] = None) -> Model:
        """
        Make a model from a dictionary of parameters. If model already exists then return that model.
        If fail to make model then raise optclim_exceptions.submitModel.
        If model already exists then behaviour depends on model state  though model_status is set to 'called'
          Instantiated -- raise optclim_exceptions.submitModel as model needs to be submitted.
          Processed -- return model
          Anything else (including Created) -- raise ValueError as should not be here

        :param params: dictionary of parameters
        :param reference_name: name of reference model to use. Pass None if want Model default behaviour.
        :return: Model object
        """

        # add reference model to params if not already there.
        if 'reference' not in params and self.refDir is not None:
            params['reference'] = self.refDir

        model = self.get_model(params)

        if model is None:  # no model so time to create one.
            model = self.create_model(params, reference_name=reference_name,
                                      dump=False)  # returns None if no model was created.
            if model is None:  # no model can be created.
                raise optclim_exceptions.submitModel
                # Immediately raise exception as None means no model created and nothing else can be done
                # will run out any remaining models.
            # otherwise just return model. To do this as go through creation path once.
            return model
        else:  # std path -- model already exists so set status and potentially update reference.
            # TODO -- might be able to get rid of this branch.
            if reference_name is not None:
                model.update_reference_name(reference_name)  # reset reference_name if provided.
            self.set_model_status(model, 'called')  # we are calling the model.

        # Check model.status
        if model.status in ["INSTANTIATED"]:  # model is created or instantiated.
            my_logger.debug(f"Model {model} has been instantiated but not run -- need to submit")
            raise optclim_exceptions.submitModel
        elif model.status in ["PROCESSED"]:  # model has been Processed
            my_logger.debug(f"Using existing model {model}")
        else:  # not processed/Instantiated so raise ValueError and complain.
            raise ValueError(f"{model} status != PROCESSED but is {model.status}")
        return model

    def transform_check(self, obs: pd.Series,
                        transform: typing.Optional[pd.DataFrame] = None,
                        scale: bool = False,
                        residual: bool = False,
                        ) -> pd.Series:

        """
        Check and transform obs
        :param obs: observations to be checked and transformed. Will reduce them to the obsNames in self.config.obsNames()

        :param scale: If True scale obs
        :param residual: If True obs are relative to target values
        :param transform: Transform matrix

        Obs are transformed in the order scale, residual, transform.
        return: pandas series of transformed obs

        Checks are that no missing observations, no desired obs are missing and no nulls are in transformed series.
        """

        obsNames = self.config.obsNames()
        missing_obs = set(obsNames) - set(obs.index)
        if len(missing_obs) > 0:  # trigger error as missing obs
            raise ValueError(f"Missing {' '.join(missing_obs)} from {obs.name}")
        # force fixed order and select only those obs we want.
        obs = obs.reindex(obsNames)
        # note using obsNames as specified. transform (if supplied) can change names.
        if scale:  # scale sim obs.
            obs *= self.config.scales()
        if residual:  # difference from target obs
            tgt = self.config.targets(scale=scale)
            obs -= tgt
        if transform is not None:  # apply transform if required.
            obs = obs @ transform.T  # obs in nsim x nobs; transform  is nev x nobs.
            # obs = transform@obs.T # how we should do it.
        null = obs.isnull()
        if np.any(null):
            raise ValueError("Obs contains null values at: " + ", ".join(obs.index[null]))
        return obs

    def logical_models(self, name) -> list[Model]:
        """
        Get list of models associated with a logical name.
        :param name: logical name
        :return: list of models associated with this logical name.
        """
        return self._logical_info.models[name]

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
                              update_models: bool = True) -> pd.Series:
        """
        Update the parameters for a given logical name.
           Checks that all existing parameters are in params
         and then updates the parameters.
          Provided for case when parameters are updated for model sims after initial storage.
        :param name: logical name.
        :param parameters: list of parameters to update. Parameters updated will be these + any existing parameters.
        :return: pandas series of updated parameters but logical_info is updated in place.

        Note this method does not write anything to disk. Just changes values internally.
        """
        # first check existing params are all in params.
        orig_params = self._logical_info.parameters[name].to_dict()
        params = list(set(list(orig_params.keys()) + parameters))  # combine existing and new params.
        # Need to update the model parameters from the models...

        ref_param_values = None
        param_values = None
        for model in self._logical_info.models[name]:
            if update_models:
                key = self.key_for_model(model)  # original key
                param_values = model.update_params(params)  # update the model parameters.
                new_key = self.key_for_model(model)
                if new_key != key:
                    self.model_index.pop(key)  # remove model from the index with original key.
                    self.model_index[new_key] = model  # add it back in with updated key.

            else:
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
        # Done dealing with models

        if param_values is None:
            raise ValueError(f"No models found for logical name {name}")  # Could be a warning and return empty series.
        new_key = self._logical_info.update_params(name, param_values)  # update the parameters in place.
        return param_values

    def logical_cost(self) -> pd.Series:
        """
        Return a series of costs for all logical names.
        :return: series with costs
        """
        return pd.Series(self._logical_info.cost).rename("cost")

    def logical_obs(self,
                    normalize: bool = False,
                    scale: bool = True,
                    obs_names: typing.Union[bool, list[str], None] = None) -> pd.DataFrame:
        """
        Return a dataframe of observations for all logical names.
        :param normalize: normalise the obs by error estimates from target
        :param scale: scale the obs by self.config.scales()
        :param obs_names: Names to use, If None -- use config obsNames.
           If True uses all obs in self._logical_info.obs. This might fail with normalize if not all obs are present in tgt.
        :return: dataframe of obs
        """
        obs_df = pd.DataFrame(self._logical_info.obs).T
        if obs_names is None:
            obs_names = self.config.obsNames()  # use config supplied one.
        elif isinstance(obs_names, bool) and obs_names:  # if True
            obs_names = obs_df.columns
        else:
            pass

        obs_df = obs_df.reindex(columns=obs_names)

        if scale:  # scale ?
            obs_df *= self.config.scales(obsNames=obs_names)

        if normalize:  # normalize
            tgt = self.config.targets(scale=scale, obsNames=obs_names)
            obs_df -= tgt  # difference from tgt.
            cov = self.config.Covariances(scale=scale)  # get covariances.
            errCov = cov['CovTotal']  # just want the total
            sd = pd.Series(np.sqrt(np.diag(errCov)),
                           index=errCov.index)  # square root of diagonal elements. Need to reindex.
            sd = sd.reindex(obs_names)  # extract only those we want.
            obs_df /= sd  # normalise by SD

        return obs_df

    type_multi_model_fn = typing.Callable[
        ["runSubmit", dict[str, dict]], tuple[list[Model], typing.Optional[pd.Series]]]

    # mult model fn takes as args a runSubmit obj and a dict and returns
    #    a list of Models and pandas series (of obs)/None (if sims do not exist)
    def comp_logical_obs(self,
                         params: dict,
                         fixed_params: dict,
                         n_ensemble: int = 1,
                         multi_config_fn: typing.Optional[type_multi_model_fn] = None,
                         transform: typing.Optional[pd.DataFrame] = None,
                         scale: bool = False,
                         residual: bool = False,
                         ) -> typing.Optional[pd.Series]:
        """
        Compute the observations for a given set of parameters.
        Broadly this function generates (via make_model) the model(s) needed to compute the observations or gets the observations from those models that have been ran.
        It does any ensemble averaging needed.
          Adds in fixed params and calls make_model or multi_config_fn to actually get data from the model.
          If n_ensemble > 1 then runs ensemble and averages the results.
          Stores the parameters, obs and cost in self._logical_info.parameters, self._logical_info.obs and self._logical_info.cost respectively.
            For the later two only once obs are generated.  Cost is sum of squares of obs (after processing by transform_check) divided by no of obs.
        :param params: dictionary of parameters which vary
        :param fixed_params: dictionary of fixed parameters.
           If multi_config_fn is None then this is a simple dict otherwise it is a dict of dict with keys being interpreted by the function.
        :param n_ensemble: number of ensemble members. If set to 1 (default) then no ensembleMember parameter is added.
        :param multi_config_fn: function to call if using multiple models to compute obs.
        The following parameters are applied after ensemble averaging (if any) and using transform_check method.
        :param transform -- transform matrix to apply to obs.
        :param scale -- if True scale obs by self.config.scales()
        :param residual -- if True compute obs as difference from target.


        :return: pandas series of potentially transformed obs  or None if some run is needed.
        """

        model_fail = False
        ## Check that ensembleMember is not in params or fixed_params when n_ensemble > 1
        # I am not convinced this is the best way to do it but for now it will do.
        if n_ensemble > 1:
            if multi_config_fn is not None:
                for k, v in fixed_params.items():
                    if 'ensembleMember' in v:
                        raise ValueError(f"ensembleMember is in fixed_params for model {k} when n_ensemble > 1")
            elif 'ensembleMember' in params:
                raise ValueError("ensembleMember set in params when n_ensemble > 1")
            elif 'ensembleMember' in fixed_params:  #
                raise ValueError("ensembleMember is in fixed_params when n_ensemble > 1")
            else:
                pass  # all ok.

        name = self._logical_info.name(params)  # get the name based on the params. This will also store the params.
        # For now no caching of obs or cost. Could be done if needed. TODO insert caching if needed.
        ## Try and compute all observations wanted.
        obs = []  # where we will store the obs for each ensemble member.
        models_created = []
        for ens_member in range(n_ensemble):  # loop over ensemble members
            # We try and get all the ensemble members and then return None if any need running.
            # Do this so have a full list of cases to run to allow parallelism.
            sim_obs = None  # where we will store the obs for this ensemble member. Set to None to start with so can check if we need to run a model
            if n_ensemble == 1:
                ens_param = {}  # empty dict
            else:
                ens_param = dict(ensembleMember=ens_member)  # set ensemble member

            # note that ensembleMember will be overwritten if it is in fixed_params or params. Checked above.
            if multi_config_fn is None:  # simple calculation
                full_params = ens_param | params | fixed_params  # needs to be using  python 3.9+ for | operator.
                full_params.update(reference=self.expand(
                    full_params.get('reference', self.refDir)).as_posix())  # add in reference params if they are there.
                model = self.make_model(full_params)  # create the model.
                models_created.append(model)
                if model is not None:
                    sim_obs = model.simulated_obs
            else:
                # set up dict containing all parameters for each model and then run multi_config_fn.
                all_params = {k: (ens_param | params | fp) for k, fp in
                              fixed_params.items()}  # needs to be using  python 3.9+ for | operator.
                # Make sure reference is in each set of params.
                # now call multi_config_fn on the dict that was constructed.
                models, sim_obs = multi_config_fn(self, all_params)  # obs will be None if any models need running.
                models_created += models  # add the models created to the list of models created during this call.
                if sim_obs is not None and not isinstance(sim_obs, pd.Series):
                    raise ValueError(f"multi_config_fn should return a pandas Series or None but got {type(sim_obs)}")

            if sim_obs is None:
                model_fail = True  # flag that we need to return None once we have looped over ensemble members.
            else:
                # work out name for this ensemble member so when make a data array have unique index. And store the obs
                obs.append(sim_obs.rename(f"r{ens_member}"))  # store the obs and name it by ensemble member.
        ## Done loop over ensemble members. Now for final processing.
        self._logical_info.models[name] = models_created  # store all models created during this call.
        if model_fail:  # some model needs running so return None
            return None
        # otherwise all models we need have ran.
        if len(obs) != n_ensemble:
            raise ValueError(f"Logic error -- expected {n_ensemble} ensemble members but got {len(obs)}")
        # now have all the ensemble members.
        if n_ensemble > 1:  # ensemble avg obs if necessary
            obs = pd.DataFrame(obs).mean(axis=0)
        else:
            obs = obs[0]  # single obs.
        obs = obs.rename(name)  # rename series to logical name.
        # now  got obs so can store them
        self._logical_info.obs[name] = obs  # store obs

        # now apply transform fn and compute cost
        obs = self.transform_check(obs, transform=transform, scale=scale, residual=residual)
        n_obs = len(obs)
        self._logical_info.cost[name] = (obs ** 2).sum() / n_obs  # store the avg cost
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
        obj: runSubmit = super(runSubmit, cls).from_dict(dct)  # create the runSubmit object
        obj._logical_info.keys_to_models(obj.model_index)  # create the models in logical_info from the keys.
        # legacy if model_status is not in dct then fill it in from the model_index set status to unknown
        if 'model_status' not in dct:
            my_logger.warning('legacy change: adding existing models to model_status with status "unknown"')
            for model in obj.model_index.values():
                obj.set_model_status(model, 'unknown')
        # Check for keys in model_status not in model_index and delete them
        keys = list(obj.model_status.keys())
        for key in keys:
            if key not in obj.model_index:
                del (obj.model_status[key])  # remove the status for unneded model.
                my_logger.warning(f'Key {key} not in model_index. Removed from model_status')
        # check keys are the same and fail if not
        if set(obj.model_status.keys()) != set(obj.model_index.keys()):
            raise ValueError(
                f"model_status keys {set(obj.model_status.keys())} do not match model_index keys {set(obj.model_index.keys())}")

        return obj

    def update_params(self, update_parameters: list[str]):
        """
        Update in-place the parameters in logical_info for all logical names
        Calls super_class method and then handles logical_info
        :param update_parameters: list of parameter names to update.
        :return: None
        """
        super().update_params(update_parameters)  # call the super  class method.
        names = list(self._logical_info.names.values())
        for name in names:  # sort out the logical info
            self.update_logical_params(name, update_parameters)

    def copyConfig(self, direct: pathlib.Path,
                   extra_files: typing.Optional[list[pathlib.Path]] = None,
                   update_paths: bool = True) -> runSubmit:
        """
        Copy the config to a new directory. Uses super class method and then updates LogicalInfo
        :param direct: directory to copy to.
        :param extra_files: extra files to copy.
        :param update_paths: if True update paths in config to point to new directory.
        :return: new runSubmit object with config copied to new directory.
        """
        # if extra_files is None:
        #    extra_files = []
        # extra_files += [self.config.config_path.relate_to(self.rootDir)]  # always copy config file.
        new_run_submit = super().copyConfig(direct, extra_files, update_paths)
        # update the logical info model info.
        new_run_submit._logical_info.keys_to_models(new_run_submit.model_index, update_models=True)
        return new_run_submit

    def create_model(self, params: dict,
                     dump: bool = True,
                     reference_name: typing.Optional[str] = None) -> typing.Optional[Model]:
        """
        runSubmit version of create_model. Call super class and set model_status
        :param params: dict of parameters to create the model.
        :param dump: if True dump the model to disk
        :param reference_name: Reference name for model
        :return: newly created model
        """
        # call the super class.
        model = super().create_model(params, dump=dump, reference_name=reference_name)
        if model is not None:  # might get None if stop is set.
            self.set_model_status(model, 'called')
            if reference_name is not None:  #
                model.update_reference_name(reference_name)
                # reset reference_name if provided. May not be needed as this really to deal with legacy case where
                # reference was not set and here we are creating a new model.

        return model

    def read_model_configs(self, path_list: list[pathlib.Path]) -> list[Model]:
        """
        Reads model configs from a list of paths. Sets status to read
        :param path_list: List of paths
        :return: list of models.
        """

        models = super().read_model_configs(path_list)
        for model in models:
            self.set_model_status(model, 'read')

        return models

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
        :param df  -- If True return all obs (read in after processing) as a dataframe.
        :param raiseError  -- If True raise optclim_exceptions.submitModel  if any requested models do not exist.
                This should cause generation & submission of models that need running.
                Else return array full of nans.
        :param ensemble_average -- If True average the ensemble members.

        The four  parameters below they are applied in the order: scale, residual, transform, sumSquare.
        The first three are handled in comp_logical_info.obs and occur after any ensemble averaging.
        :param scale   -- if True scale obs  by self.scales()

        :param residual  -- if True remove target (self.target()) from obs

        :param transform  -- if provided transform the model obs by matrix multiplying them by this matrix.
                  It should be  N*nobs where nobs are the number of sim ons and  0 <= N <= nobs.
                  One application of this is to transform the data into a basis of eigenvectors of an
                    error covariance matrix. The transform matrix should be provided as a pandas datarray.
                  Column names being the obs names. Index being sensible labels for rows which will be new obs names.
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
        nEns = self.config.ensembleSize()  # how many ensemble members do we want to run.
        empty = pd.Series(np.repeat(np.nan, nObs), index=obsNames)
        multi_config_fn = self.config.fixed_param_function()
        fixed_params = self.config.fixedParams()  # get fixed parameters
        for indx in range(0, nsim):  # iterate over the simulations.
            pDict = dict(zip(paramNames, use_params[indx, :]))  # create dict with names and values.
            if ensemble_average or (nEns == 1):  # can do ensemble average in comp_logical_obs
                obs = self.comp_logical_obs(pDict, fixed_params, n_ensemble=nEns, multi_config_fn=multi_config_fn,
                                            transform=transform, scale=scale, residual=residual)
                result.append(obs)
            else:  # want all ensemble members separately but only if want more than one ensemble member.
                for ens_member in range(0, nEns):
                    params_ens = pDict | dict(ensembleMember=ens_member)
                    obs = self.comp_logical_obs(params_ens, fixed_params, multi_config_fn=multi_config_fn,
                                                transform=transform, scale=scale, residual=residual)
                    result.append(obs)

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

    def reset_logical_info(self):
        """
        Reset the logical info and model_status for unknown/called models to not_called. Needed when running algorithms.
        :return: None
        """
        # iterate over a static list of keys to avoid runtime mutation issues
        for key in list(self.model_status.keys()):
            if self.model_status.get(key) in ['unknown', 'called']:
                self.model_status[key] = 'not_called'
        # reset logical info to empty.
        self._logical_info = LogicalInfo()

    def set_model_status(self, model: Model, status: type_model_status):
        """
        Set  model_status
        :param model: A model object
        :param status: One of 'initial','called','not_called','unknown','read'
        :return: Nada
        """
        allowed_status = ['initial', 'called', 'not_called', 'unknown', 'read']
        if status not in allowed_status:
            raise ValueError(f"Invalid status {status}. Must be one of {' '.join(allowed_status)}")
        key = self.key_for_model(model)
        self.model_status[key] = status

    def check_deterministic(self, error: genericLib.error_handle_types = 'warn') -> bool:
        """
        Check that the model runs are deterministic. Done by checking that no value in model_status is not_called
        If not then suggests that some models that were run were not used which suggests non-determinism in the algorithm.

        Also checks that model_status and model_index have the same keys
          This will trigger an ValueError if they don't have the same keys regardless of value of error

        :param error: Used to in call to genericLib.error_handle to determine whether to raise an error, warn or ignore.
          See genericLib.error_handle for allowed values and what is done.
        :return: True if deterministic, False otherwise.
        """
        # want to check everything in self.model_status is in model_index (and vice versa)
        keys_model_status = set(list(self.model_status.keys()))
        keys_model_index = set(list(self.model_index.keys()))
        if keys_model_index != keys_model_status:
            raise ValueError("Problem with model_status and model_index. Fix code as keys differ.")

        # check no keys in model_status are not_called
        not_called = [k for k, v in self.model_status.items() if v == 'not_called']  # list of uncalled keys
        if not_called:
            uncalled_models = [str(self.model_index[k]) for k in not_called]
            message = f'Have {len(uncalled_models)} models not called. Unused models are:\n' \
                      + "\n".join(uncalled_models)
            genericLib.error_handle(message, error)

        return len(not_called) == 0  # return True if no missing keys, False otherwise.

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
            obs = self.logical_obs(scale=True, normalize=True).dropna()
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
        obs = modelFn(params.values)  # compute the obs where we need to. This may generate model simulations
        dobs = obs.iloc[1:, :] - obs.iloc[0, :]
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
         simulated_observations: Path to csv file of previous simulated obs. Header should be obs names, col 0 names
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
        obs = pd.read_csv(self.expand(eval_config['simulated_observations']), index_col=0).reindex(obs_names, axis=1)
        if obs.isnull().any().any():  # check obs
            raise ValueError("Some observations in evaluation database are missing. Check observation names.")
        my_logger.debug(f"Simulated_obs shape: {obs.shape} ")
        trans_obs = [self.transform_check(o, transform=transform, scale=scale, residual=True) for name, o in
                     obs.iterrows()]
        trans_obs = pd.DataFrame(trans_obs)
        if trans_obs.isnull().any().any():  # check transformed obs
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

        :param scale (default True). If True scale data for transform matrix and in calculations of obs
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
            # +1 allows cases that have been run but not added to logical obs/cost
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

        except np.linalg.linalg.LinAlgError:
            n_inst_models = len(self.models_to_instantiate())
            my_logger.info(f"Have just generated {n_inst_models} to instantiate")
            neval = len(self.logical_cost())
            if (neval > 1):  # got some evaluations.
                # Run DFOLS again with reduced number of fn evals to provide some diagnostic info.
                my_logger.debug('Running DFOLS again with reduced number of fn evals to get diagnostic info')
                random.seed(rng_seed)  # reset rng seed back to first value.
                with warnings.catch_warnings():  # catch the complaints from DFOLS about NaNs encountered...
                    warnings.filterwarnings('ignore')  # Ignore all warnings...
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
            raise optclim_exceptions.submitModel("dfols failed with lin alg error")
            # this is how DFOLS tells us it got NaN which then triggers running the next set of simulations.

        # Code here will be run when DFOLS has completed.
        # It mostly puts stuff in the final JSON file so can easily be looked at for subsequent analysis.
        if solution.flag not in (solution.EXIT_SUCCESS, solution.EXIT_MAXFUN_WARNING):
            print("dfols failed with flag %i error : %s" % (solution.flag, solution.msg))
            raise Exception("Problem with dfols")

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

        It does not do any scaling, residual or transform. It just runs the model(s) and returns the simulated obs.
        The parameters to use are specified in the configuration file.
        The fixed parameters are also specified in the configuration file.
        :param ensemble_average -- if True (default) and if ensemble size > 1 then average the ensemble members.
        :return: finalConfig -- a studyConfig. The following methods should give you useful data:
                finalConfig.obs() -- the observations

        """
        if stop:
            raise NotImplementedError("run_params stop not implemented as have no use case")

        params_dir = self.config.optimise()  # get the parameters to run
        params = self.get_parameters(params_dir)  # convert to dataframe

        obs = self.stdFunction(params.values, df=True, raiseError=True, ensemble_average=ensemble_average,
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

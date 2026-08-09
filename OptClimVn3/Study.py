"""
Class to provide ways of looking at a study directory.
This is a readOnly view -- unless you hack the Study object directly.
Note no methods are provided to save Study objects
SubmitStudy inherits from this and that has methods to submit models and modify state.
"""
from __future__ import annotations
import matplotlib.pyplot as plt  # so we can plot
import datetime
import copy
import logging
import pathlib  # needs python 3.6+
import typing

import numpy as np
import pandas as pd

from Models import Model
from model_base import model_base
from Model  import Model # root class for all models.
#from StudyConfig import OptClimConfigVn3
my_logger = logging.getLogger(f"OPTCLIM.{__name__}")
#TOMAYBEDO: Consider removing keeping the config. Instead, just parse bits of it that we need and store them in the Study.

class Study:
    # class attribute type information.
    config: "OptClimConfigVn3"
    name: str
    rootDir: pathlib.Path
    model_index: dict
    """
    Class to support a study.  This class provides support for reading info
    from a study -- both in progress or completed, and displaying them. 
    Does not handle submitting models or realizing that a new one needs to be generated.
    An instance has the following attributes:
    config -- configuration
    name -- name of the study.
    rootDir -- path to where stuff is
    model_index -- dict containing models indexed by model keys. 
    """

    def __init__(self, config: "OptClimConfigVn3",
                 name: typing.Optional[str] = None,
                 rootDir: typing.Optional[pathlib.Path] = None,
                 models: typing.Optional[list[Model]] = None):
        """
        Create read-only study instance.
        :param config: Configuration information.
        :param name: Name of the study. If None name of config is used.
        :param rootDir : Root dir where, by default, config file will be created and
            where model configurations will be searched for. Will be converted to an absolute path if not already by
            prepending the current working directory.
          If None will be current dir/config.name().
        :param models : Lst of models.
        """

        self.config = copy.deepcopy(config)

        if name is None: # NB once object exists its name cannot be changed.
            name = config.name()
        if name is None:  # still none as not defined in config
            name = 'Unknown'
        if name is not None:
            self.name = name

        if rootDir is None:  # No rootDir defined. Use cwd.
            self.rootDir = pathlib.Path.cwd() / self.name  # default path
        else:
            self.rootDir = rootDir
        # make sure rootDir is an absolute path rather than relative.
        if not self.rootDir.is_absolute():
            self.rootDir = pathlib.Path.cwd() / self.rootDir




        self.model_index = dict()
        if models is not None:
            for model in models:
                key = self.key_for_model(model)
                if key in self.model_index.keys():
                    raise ValueError(f"Got duplicate key {key}")
                self.model_index[key] = model

    def update_config(self, config: "OptClimConfigVn3"):
        """
        Add config to self and derive what needed from config to include in self.

        :param config: A study config used to set. This is deep copied into self
        :return: Nothing
        """
        my_logger.warning("Updating self.config. Be very very careful when you do this.")
        self.config = copy.deepcopy(config)


    def __repr__(self):
        """
        Returns a string representation of Study
        :return: String containing the name of the study, number of models and model types/states -- grouped.
        """

        def clean_series(series):
            """
            Clean repr of a series
            :param series: series to be represented
            :return: clean string!
            """
            return repr(series.to_dict()).replace("'", "").replace("{", "").replace("}", "")

        nmodels = len(self.model_index)
        if nmodels > 0:
            model_types = pd.Series({m.name: m.class_name() for m in self.model_index.values()})
            # pre 2.0 code. where series needs to be passed to by. 2.0+ by=None will do this.
            model_types = clean_series(model_types.groupby(by=model_types).count())
            status = self.status()
            status = clean_series(status.groupby(by=status).count())
        else:
            model_types = ""
            status = ""

        s = f"Name: {self.name} Nmodels:{nmodels}" \
            f" Status: {status} Model_Types:{model_types}"
        return s

    @classmethod
    def key_for_model(cls,model: Model, fpFmt: str = '%.4g') -> str:
        """
        Generate key from model
        :param model: model for which key gets generated. Uses parameters to generate the key.
        :param fpFmt: floating point format for floats
        :return: key
        """
        key = cls.key(model.attrs_for_key(), fpFmt=fpFmt)  # Generate key.
        return key

    @staticmethod
    def key(parameters: typing.Mapping, fpFmt: str = '%.4g') -> str:
        """
        Generate key from keys and values in parameters. There should include all fixed and variable parameters.
        This should be unique (to some rounding on float parameters)
        :param parameters -- a dictionary (or something that behaves likes a dict) of variable parameters.
        :param fpFmt -- format to convert float to string. (Default is %.4g)
        :return: a tuple as an index. tuple is key_name, value in sorted order of key_name.
        """
        params_to_ignore = ['reference_name'] # parameters to ignore when constructing the key
        keys = []
        param_keys = sorted(parameters.keys())  # fixed ordering
        param_keys = [ p for p in param_keys if p not in params_to_ignore ] # remove params_to_ignore from key construction
        # params to ignore.

        # deal with variable parameters -- produced by optimisation so have names and values.
        for k in param_keys:  # iterate over keys in sorted order.
            keys.append(k)
            v = parameters[k]
            if isinstance(v, float):
                keys.append(fpFmt % v)  # float point number so use formatter.
            elif isinstance(v,str):
                keys.append(v)  # string so just append
            elif isinstance(v,pathlib.PurePath): # pathlib object # Path inherits from PurePath
                keys.append(v.as_posix())
            else:  # just append the value.
                keys.append(repr(v))  # use the object repr method.

        keys = tuple(keys)  # convert to tuple
        return str(keys)  # and then to a string.
    @staticmethod
    def key_to_dict(key:str) -> dict[str,str]:
        """
        Convert key back to dict though values will be strings.
         A bit of a hack and really only there to support updating keys from old versions...
        :param key: string to be converted back to dict
        :return: dict of parameters & str repr of values.
        """
        # convert key back to dict.

        values = [k[1:-1] for k in key[1:-1].split(', ')]
        # remove any ' in values
        values = [v.replace("'","") for v in values]

        if len(values) % 2 != 0:
            raise ValueError(f'Expected even number of keys, got {len(values)}')
        result = dict(zip(values[0::2], values[1::2]))
        return result


    def get_model(self, parameters: typing.Mapping, fpFmt: str = '%.4g') -> typing.Optional[Model]:
        """
        Return model  that matches key generated from parameters or None if not match.
        :param parameters: parameters as a dict
        :param fpFmt: float format passed into genKey
        :return: model that has parameters.
        """

        key = self.key(parameters, fpFmt=fpFmt)
        my_logger.debug(f"Key is: {key}")
        model = self.model_index.get(key, None)

        return model

    def read_dir(self, direct: typing.Optional[pathlib.Path] = None, pattern: str = '*.mcfg'):
        """
        Read all files that look like model config files.
        :param pattern: glob pattern to match for model config
        :param direct: directory to look in -- all subdirectories will be looked for
        :return:
        """
        if direct is None:
            direct = self.rootDir
        if not direct.is_dir():
            raise ValueError(f"Directory {direct} is not a directory")
        files = direct.glob("**/" + pattern)
        self.read_model_configs(files)

    def read_model_configs(self, path_list: list[pathlib.Path]) -> typing.list[Model]:
        """
        Read model configurations from path_list and store them in self.model_index
          key will be generated from the model parameters and value will be the model.
        :param path_list: list of paths to read model configurations from.
        :return: models read in

        """
        models = []
        for f in path_list:
            try:
                my_logger.debug(f"Trying to load from {f}")
                m = Model.load_model(f)  # read the model.
                key = self.key_for_model(m)  # work out the key
                if key in self.model_index.keys():
                    my_logger.warning(f"Have duplicate model/key {key} {m}")
                self.model_index[key] = m
                models.append(m)
            except (IOError, EOFError):
                my_logger.warning(f"Failed to load_model from {f}. Ignoring.")

        return models
    

    def status(self) -> pd.Series:
        """
        :return: pandas series of model status
        """
        dct = {model.name: model.status for model in self.model_index.values()}
        return pd.Series(dct, dtype=str).rename(self.name)

    def params(self, normalize: bool = False,
               numeric:bool = False,
               model:typing.Optional[Model]=None,
               keys:typing.Optional[list[typing.Hashable]]=None) -> pd.DataFrame|pd.Series:
        """
        Extract the parameters used in the simulations. Will include ensembleMember -- as a "fake" parameter
        :param numeric -- if True convert all parameters to numeric values using errors='coerce'.
        :param normalize -- If True normalize parameters by min and max values
        :return: pandas dataframe of parameters
        """
        param_names = self.config.paramNames()  # parameter names we want
        if keys is None:
            keys = self.model_index.keys()

        models = [self.model_index[key] for key in keys]
        if model is None: # extract from existing list of models
            p = [pd.Series(model.parameters).rename(model.name).reindex(param_names)
                 for model in models]
            paramsDF = pd.DataFrame(p)
        else:
            paramsDF=pd.Series(model.parameters).rename(model.name).reindex(param_names)
        if numeric: # make the result numeric.
            paramsDF = paramsDF.apply(pd.to_numeric,errors='coerce')
        if normalize:  # want normalised values
            rng = self.config.paramRanges(paramNames=param_names)
            paramsDF = (paramsDF - rng.loc['minParam', :]) / rng.loc['rangeParam', :]

        return paramsDF


    def processed_models(self) -> list[Model]:
        """

        :return: List of models that have processed
        """
        return [model for model in self.model_index.values() if model.is_processed()]

    def simulated_observations(self,use_cache:bool = True) -> pd.DataFrame:
        """
        Get the simulated observations for all processed models.
        :param use_cache: Whether to use cached data if available. Passed through to model.simulated_obs().
         Set to False to force reload of observations
        :return: dataframe with all simulated observations. Empty dataframe if nothing available.
        """

        observations = [model.compute_simulated_observations(use_cache=use_cache) for model in self.model_index.values()
               if model.is_processed()]



        return pd.DataFrame(observations) if observations else pd.DataFrame()


    def obs(self, scale: bool = True,
            normalize: bool = False,
            obsNames:typing.Optional[list[str]]=None) -> typing.Optional[pd.DataFrame]:
        """
        Extract the Obs used in the *individual* simulations. If simulation has no observations then it is ignored.
        :param scale If True data will be scaled.
        :param normalize If True data will be normalized -- distance in SD's from tgt
        :param obsNames list of names of observations. If not provided will be extracted from model obs and target.
        :return: pandas dataframe of observations possibly scaled and normalized.
           None will be returned if there are no obs
        """

        obsDF = self.simulated_observations()  # get obs for all processed models.
        if obsDF.empty: # got an empty dataframe
            return None

        if obsNames is None:
            obsNames = self.config.obsNames()

        obsDF=obsDF.reindex(columns=obsNames).dropna(axis=1)

        if scale:  # scale ?
            obsDF *= self.config.scales(obsNames=obsNames)

        if normalize:  # normalize
            tgt = self.config.targets(scale=scale, obsNames=obsNames)
            obsDF -= tgt  # difference from tgt.
            # drop any nana which might have come from tgt
            obsDF = obsDF.dropna(axis=1)
            cov = self.config.Covariances(scale=scale)  # get covariances.
            errCov = cov['CovTotal']  # just want the total
            sd = pd.Series(np.sqrt(np.diag(errCov)),
                           index=errCov.index)  # square root of diagonal elements. Need to reindex.
            obsDF /= sd  # normalise by SD

        return obsDF

    def cost(self, scale: bool = True,
             obsNames:typing.Optional[list[str]]=None) -> pd.Series | None:
        """
        compute cost from data.
        :param: scale -- scale data.
        :return pandas series of costs.
        """
        obs = self.obs(scale=scale,obsNames=obsNames)  # get obs
        if obs is None:  # no data
            return None

        # which puts us into space where totalError is Identity matrix.

        target = self.config.targets(scale=scale,obsNames=obsNames)  # get targets
        obs = obs.reindex(columns=target.index)  # reindex obs to match targets.
        obs = obs.dropna(axis=1) # drop any missing data
        target = target.reindex(index=obs.columns)
        tMat = self.config.transform_matrix(scale=scale, obs_names=obs.columns)
        # extract just what we have in obs.
        tMat = tMat.reindex(columns=obs.columns)
        resid = (obs - target) @ tMat.T
        nobs = resid.shape[1]
        if nobs == 0:
            raise ValueError(
                "No usable observation columns remain after dropping missing data"
            )
        cost = np.sqrt(
            (resid ** 2).sum(1).astype(float) /nobs)
        cost = pd.Series(cost, index=obs.index).rename('cost ' + self.name)
        return cost

    def runConfig(self, filename: typing.Optional[pathlib.Path] = None,
                  scale: bool = True,
                  add_cost: bool = True,
                  best:typing.Optional[pd.Series]=None,
                  transJacobian:typing.Optional[pd.DataFrame] = None,
                  transJacobian_comment:typing.Optional[str] = None,
                  jacobian:typing.Optional[pd.DataFrame] = None,
                  jacobian_comment:typing.Optional[str] = None,
                  hessian:typing.Optional[pd.DataFrame] = None,
                  hessian_comment:typing.Optional[str] = None) -> "OptClimConfigVn3":
        """
        **copy** self.config and add parameters and obs to it. Modified config is returned
        :param filename - pathlib to file (or None). Will override filepath in new config
        :param scale -- passed to self.cost to compute cost when add_cost is True and also used to do inverse transform matrix
        :param add_cost -- add cost to the returned configuration
        :param best -- best evaluation parameter set
        :param transJacobian -- transformed jacobian matrix. Rows are parameters, columns are observations.
        :param jacobian -- jacobian matrix in initial original  space. Rows are parameters, columns are observations.
          If not provided then transJacobian is used to compute jacobian
        :param hessian -- hessian matrix.
        If not proved but jacobian is then hessian is computed as 2*jacobian.T@jacobian -- min sum of squares.
        :param transJacobian_comment -- comment for transJacobian
        :param jacobian_comment -- comment for jacobian
        :param hessian_comment -- comment for hessian
        :return: modified config. The following methods may  work if appropriate data passed in:
            finalConfig.parameters() -- returns the parameters for each model simulation
            finalConfig.simObs() -- returns the simulated observations for each model simulation.
            finalConfig.cost() -- returns the cost for each model simulation.
            finalConfig.bestEval() -- returns the best evaluation.
            finalConfig.optimumParams() -- returns the optimum parameters.
            finalConfig.transJacobian() -- returns the transformed jacobian.
            finalConfig.jacobian() -- returns the jacobian.
            finalConfig.hessian() -- returns the hessian.
            finalConfig.alg_info().get('jacobian_comment') -- returns the comment for the jacobian. Sim for other comments

        """
        newConfig = self.config.copy(filename=filename)  # copy the config.

        # But function wants models. So suggests including some meta-data in the model
        # when we do this,
        params = self.params()  # get params & obs
        obs = self.obs()
        # update newConfig with obs & params. As normal all are unscaled.

        newConfig.parameters(params)
        newConfig.simObs(obs)
        paramNames = set(newConfig.paramNames())
        if add_cost:  # want to include cost. Which might be computed with scaling
            cost = self.cost(scale=scale)
            # update newConfig
            cost = newConfig.cost(cost)
            # work out best directory by finding minimum cost.
            bestEval = None
            if len(cost) > 0:
                bestEval = cost.idxmin()
            newConfig.setv('bestEval', bestEval)  # best evaluation
        # add in optimum param set if provided. May be different from best eval based on cost.
        if best is not None:
            newConfig.optimumParams(optimum=best)  # store the optimum params.
        if transJacobian is not None:
            # check columns are obs names
            cols = set(transJacobian.columns)
            if cols != paramNames:
                raise ValueError(f"transJacobian columns {cols} do not match param names {paramNames}")
            newConfig.transJacobian(transJacobian,comment=transJacobian_comment)
            if jacobian is  None: # compute jacobian
                my_logger.info('Computing jacobian from transJacobian')
                inv_transM = newConfig.transform_matrix(scale=scale, inverse=True)  # get the inverse transformation matrix
                jacobian = inv_transM @ transJacobian  # transform jacobian to parameter space
                jacobian_comment = 'Jacobian computed from transJacobian'
            if hessian is None:
                my_logger.info('Computing hessian from transJacobian')
                hessian = 2 * transJacobian.T @ transJacobian # sum of squares
                hessian_comment = 'Hessian computed from transJacobian assuming sum of squares'
        # add jacobian  if provided  (or computed)
        if jacobian is not None:
            # check columns are obs names
            cols=set(jacobian.columns)
            paramNames = set(newConfig.paramNames())
            if cols != paramNames:
                raise ValueError(f"Jacobian columns {cols} do not match param names {paramNames}")
            newConfig.jacobian(jacobian,comment=jacobian_comment)
        if hessian is not None: #
            # check columns and index are param names
            if set(hessian.columns) != paramNames:
                raise ValueError(f"Hessian columns {hessian.columns} do not match param names {paramNames}")
            if set(hessian.index) != paramNames:
                raise ValueError(f"Hessian index {hessian.index} do not match param names {paramNames}")
            newConfig.hessian(hessian,comment=hessian_comment)


        return newConfig

        # do some plotting

    def plot(self,
             fname:typing.Optional[pathlib.Path]=None,
             fig_name: str = 'monitor',
             cost:typing.Optional[pd.Series]=None,
             obs:typing.Optional[pd.DataFrame]=None,
             params:typing.Optional[pd.DataFrame]=None,
             savefig_kwargs:typing.Optional[dict]=None,) -> \
            typing.Optional[tuple[plt.Figure,tuple[plt.Axes,plt.Axes,plt.Axes]]]:
        """
        plot cost, normalised parameter & obs values for runs.
        :param fig_name: name of figure to make -- default is monitor
        :param fname: path to save figure to if not None. Default is None
        :param cost -- cost values to plot. If None then will use self.cost
        :param obs -- obs values to plot. If None then will use self.obs(normalise=True)
        :param params - param values to plot. If None then will self.params(normalise=True)
        :param savefig_kwargs: dict of kwargs to pass to fig.savefig. Default is empty dict.
        :return: figure, (costAxis, paramAxis, obsAxis)

        Needs matplotlib
        """
        # get a bunch of annoying messages from matplotlib so turn them off...
        logging.getLogger('matplotlib.font_manager').disabled = True
        obsNames = self.config.obsNames()
        if savefig_kwargs is None:
            savefig_kwargs = {}
        if params is None:
            params = self.params(normalize=True,numeric=True)
        if obs is None:
            obs = self.obs(scale=True,normalize=True,obsNames=obsNames).dropna()
            obsNames = obs.columns #
        if cost is None:
            cost = self.cost(obsNames=obsNames)
        if (cost is None) or (len(cost) == 0):
            my_logger.warning("Nothing to plot")
            return  None # nothing to plot
        fig, ax = plt.subplots(3, 1, num=fig_name, figsize=[8.3, 11.7],
                               sharex='col', clear=True,layout='constrained')
        (costAx, paramAx, obsAx) = ax  # name the axis .
        cmap = copy.copy(plt.cm.get_cmap('RdYlGn'))
        cmap.set_under('skyblue')
        cmap.set_over('black')
        nx = len(cost)
        costAx.plot(np.arange(0, nx), cost.values)
        a = costAx.set_xlim(-0.5, nx)
        minv = cost.min()
        minp = cost.values.argmin()  # use location in array (as that is what we plot)
        costAx.set_title("Cost", fontsize='small')
        a = costAx.plot(minp, minv, marker='o', ms=12, alpha=0.5)
        costAx.axhline(minv, linestyle='dotted')
        a = costAx.set_yscale('log')
        yticks = [1, 2, 5, 10, 20, 50]
        a = costAx.set_yticks(yticks)
        a = costAx.set_yticklabels([str(y) for y in yticks])
        # plot params
        params = params.reindex(index=cost.index)  # reorder
        X = np.arange(-0.5, params.shape[1])
        Y = np.arange(-0.5, params.shape[0])  # want first iteration at 0.0
        cm = paramAx.pcolormesh(Y, X, params.T.values, cmap=cmap, vmin=0.0, vmax=1.)  # make a colormesh
        a = paramAx.set_yticks(np.arange(0, len(params.columns)))
        a = paramAx.set_yticklabels(params.columns)
        a = paramAx.set_title("Normalised Parameter")
        a = paramAx.axvline(minp, linestyle='dashed', linewidth=2, color='gray')

        # plot norm obs
        X = np.arange(-0.5, obs.shape[1])
        Y = np.arange(-0.5, obs.shape[0])
        cmO = obsAx.pcolormesh(Y, X, obs.T.values, vmin=-4, vmax=4, cmap=cmap)
        a = obsAx.set_yticks(np.arange(0, len(obs.columns)))
        a = obsAx.set_yticklabels(obs.columns, fontsize='x-small')
        obsAx.set_xlabel("Iteration")
        xticks = np.arange(0, nx // 5 + 1) * 5
        a = obsAx.set_xticks(xticks)
        a = obsAx.set_xticklabels(xticks)
        obsAx.axvline(minp, linestyle='dashed', linewidth=2, color='gray')

        obsAx.set_title("Normalised Observations")
        # plot the color bars.
        for cmm, ax in zip([cmO, cm], [obsAx,paramAx]):
            cb = fig.colorbar(cmm, ax=ax, orientation='horizontal', fraction=0.05, extend='both')



        fig.suptitle(self.name + " " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), fontsize='small',
                     y=0.99)
        fig.show()
        if fname is not None:
            fig.savefig(fname,**savefig_kwargs)  # save the figure
        return fig, (costAx, paramAx, obsAx)

    def reload(self):
        """
        reload in place the study.
        """
        new_study_dict=vars(self.load(self.config_path))
        self.fill_attrs(new_study_dict)

    # end of Study


# use model_base.__eq__ for equality. Real hack. Sure there are better ways.
Study.__eq__ = model_base.__eq__

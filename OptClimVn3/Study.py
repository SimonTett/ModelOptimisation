"""
Class to provide ways of looking at a study directory.
Goal is to merge this into ModelSubmit (eventually).
Problem is that Study wants an ordered list while ModelSubmit wants a dict keyed on parameters.
Will privd e a
Rule#1 -- only have one copy of anything.
"""
from __future__ import annotations
import matplotlib.pyplot as plt # so we can plot
import datetime
import copy
import logging
import pathlib  # needs python 3.6+
import typing

import numpy as np
import pandas as pd

import StudyConfig
from Models.model_base import model_base
from Models.Model import Model
import generic_json
from StudyConfig import OptClimConfigVn2, OptClimConfigVn3



class Study(model_base):
    """
    Class to support a study.  This class provides support for reading info
    from a study -- both in progress or completed, and displaying them.  But does not
    handle submitting models or realising that a new one needs to be generated.
    """

    def __init__(self, config: OptClimConfigVn2|OptClimConfigVn3,
                 name: typing.Optional[str] = None,
                 rootDir: typing.Optional[pathlib.Path] = None,
                 models: typing.Optional[typing.List[Model]] = None):
        """
        Create read-only study instance.
        :type rootDir:
        :param config: configuration information
        :param name: name of the study. If None name of config is used.
        :param rootDir : root dir where config file will be created and where model configurations will be searched for.
          If None will be current dir/config.name()
        :param models -- list of models.
        """
        if config is None:
            self.config = None
        else:
            self.config = copy.deepcopy(config)

        if name is None:
            name = config.name()
        if name is None: # still none as not defined in config
            name= 'Unknown'

        self.name = name

        if rootDir is None:  # no rootDir defined. Use cwd.
            self.rootDir = pathlib.Path.cwd() / self.name  # default path
        else:
            self.rootDir = rootDir
        self.rootDir.mkdir(parents=True, exist_ok=True)  # create it if need be.

        self.config_path = self.rootDir / (self.name + '.scfg')

        self.model_index = dict()

        if models is not None:
            for model in models:
                key = self.key_for_model(model)
                if key in self.model_index.keys():
                    raise ValueError(f"Got duplicate key {key}")
                self.model_index[key] = model

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
            model_types = pd.Series({m.name:m.class_name() for m in self.model_index.values()})
            # pre 2.0 code. where series needs to be passed to by. 2.0+ by=None will do this.
            model_types =  clean_series(model_types.groupby(by=model_types).count())
            status =self.status()
            status = clean_series(status.groupby(by=status).count())
        else:
            model_types= ""
            status = ""

        s = f"Name: {self.name} Nmodels:{nmodels}" \
            f" Status: {status} Model_Types:{model_types}"
        return s

    def key_for_model(self, model:Model, fpFmt:str = '%.4g'):
        """
        Generate key from model
        :param model: model for which key gets generated. Uses parameters to generate the key.
        :param fpFmt: floating point format for floats
        :return: key
        """
        key = self.key(model.parameters, fpFmt=fpFmt)  # Generate key.
        return key

    @staticmethod
    def key(parameters: typing.Mapping, fpFmt: str = '%.4g') -> str:
        """
        Generate key from keys and values in paramDict. There should include all fixed and variable parameters.
        This should be unique (to some rounding on float parameters)
        :param parameters -- a dictionary (or something that behaves likes a dict) of variable parameters.
        :param fpFmt -- format to convert float to string. (Default is %.4g)
        :return: a tuple as an index. tuple is key_name, value in sorted order of key_name.
        """

        keys = []
        paramKeys = sorted(parameters.keys())  # fixed ordering

        # deal with variable parameters -- produced by optimisation so have names and values.
        for k in paramKeys:  # iterate over keys in sorted order.
            keys.append(k)
            v = parameters[k]
            if isinstance(v, float):
                keys.append(fpFmt % v)  # float point number so use formatter.
            else:  # just append the value.
                keys.append(repr(v))  # use the object repr method.
        keys = tuple(keys)  # convert to tuple
        return str(keys)

    def get_model(self, parameters: typing.Mapping, fpFmt: str = '%.4g') -> Model:
        """
        Return model  that matches key generated from parameters or None if not match.
        :param parameters: parameters as a dict
        :param fpFmt: float format passed into genKey
        :return: model that has parameters.
        """

        key = self.key(parameters,fpFmt=fpFmt)
        logging.debug("Key is\n", repr(key), '\n', '-' * 60)
        model = self.model_index.get(key, None)
        return model

    def read_dir(self, direct: typing.Optional[pathlib.Path]=None, pattern:str ='*.mcfg'):
        """
        Read all files that look like model config files.
        :param pattern: glob pattern to match for model config
        :param direct: directory to look in -- all subdirectories will be looked for
        :return:
        """
        if direct is None:
            direct=self.rootDir
        if not direct.is_dir():
            raise ValueError(f"Directory {direct} is not a directory")
        files = direct.glob("**/" + pattern)
        self.read_model_configs(files)

    def read_model_configs(self, path_list: iter):
        """
        Read model configurations from path_list and store them in self.model_index
          key will be generated from the model parameters and value will be the model.
        :return: models read in
        """
        models = []
        for f in path_list:
            try:
                logging.debug(f"Trying to load from {f}")
                m = Model.load_model(f)  # read the model.
                key = self.key_for_model(m)  # work out the key
                if key in self.model_index.keys():
                    logging.warning(f"Have duplicate model/key {key} {m}")
                self.model_index[key] = m
                models.append(m)
            except (IOError, EOFError):
                logging.warning(f"Failed to load_model from {f}. Ignoring.")

        return models

    def status(self) -> pd.Series:
        """
        :return: pandas series of model status
        """
        dct = {model.name: model.status for model in self.model_index.values()}
        return pd.Series(dct).rename(self.name)

    def params(self, normalize: bool = False) -> pd.DataFrame:
        """
        Extract the parameters used in the simulations. Will include ensembleMember -- as a "fake" parameter
        :return: pandas dataframe of parameters
        """
        param_names = self.config.paramNames()  # parameter names we want
        p = [pd.Series(model.parameters).rename(model.name).reindex(param_names)
             for model in self.model_index.values()]
        paramsDF = pd.DataFrame(p)

        if normalize:  # want normalised values
            rng = self.config.paramRanges(paramNames=param_names)
            paramsDF = (paramsDF - rng.loc['minParam', :]) / rng.loc['rangeParam', :]

        return paramsDF

    def obs(self, scale: bool = True, normalize: bool = False) -> pd.DataFrame | None:
        """
        Extract the Obs used in the *individual* simulations. If simulation has no observations then it is ignored.
        :param scale If True data will be scaled.
        :param normalize If True data will be normalized -- distance in SD's from tgt
        :return: pandas dataframe of observations possibly scaled and normalized.
           None will be returned if there are no obs
        """
        obs = [model.simulated_obs.rename(model.name)
               for model in self.model_index.values() if model.simulated_obs is not None]
        if len(obs) == 0:  # empty list
            return None
        obsDF = pd.DataFrame(obs)

        if scale:  # scale ?
            obsDF *= self.config.scales(obsNames=obsDF.columns)

        if normalize:  # normalize
            tgt = self.config.targets(scale=scale, obsNames=obsDF.columns)
            obsDF -= tgt  # difference from tgt.
            cov = self.config.Covariances(scale=scale)  # get covariances.
            errCov = cov['CovTotal']  # just want the total
            sd = pd.Series(np.sqrt(np.diag(errCov)),
                           index=errCov.index)  # square root of diagonal elements. Need to reindex.
            obsDF /= sd  # normalise by SD

        return obsDF

    def cost(self, scale: bool = True) -> pd.Series | None:
        """
        compute cost from data.
        :param: scale -- scale data.
        :return pandas series of costs.
        """
        obs = self.obs(scale=scale)  # get obs
        if obs is None:  # no data
            return None
        tMat = self.config.transMatrix(scale=scale,
                                       dataFrame=True)  # which puts us into space where totalError is Identity matrix.
        nObs = len(obs.columns)
        resid = (obs - self.config.targets(scale=scale)) @ tMat.T
        cost = np.sqrt(
            (resid ** 2).sum(1).astype(float) / nObs)  # TODO -- make nObs the number of indep matrices -- len(resid)
        cost = pd.Series(cost, index=obs.index).rename('cost ' + self.name)
        return cost

    def to_dict(self):
        """
        Convert Study to a dict. Uses super class method to convert but then replaces
          models with config_path and key with str(key), and replaces config with config.to_dict()
        :return: dict
        """
        dct = super().to_dict()
        logging.debug(f"Replacing models in model_index with config_path")
        m2 = dict()
        for key, model in dct['model_index'].items():
            m2[key] = model.config_path
        dct['model_index'] = m2
        logging.debug(f"Running to_dict on config. FIXME")
        dct['config']= self.config.to_dict()
        return dct

    @classmethod
    def from_dict(cls, dct: dict):
        """
        :param dct: dict to be converted to a Study.
        As config is a positional parameter which must be present this is poped out of the dct and used in the initialization
        Loads any models using the paths in model_index.
         If path does not exist then the that model is deleted with a warning given.
        :return: a study
        """
        # largely copy-paste from superclass from_dict
        # need to deal with config..which is a right mess. TODO fix StudyConfig so it is much clearer..
        known_versions = {2:StudyConfig.OptClimConfigVn2,3:StudyConfig.OptClimConfigVn3}


        config = dct.pop('config')
        version = config['Config']['version']
        c = known_versions[version](StudyConfig.dictFile())
        for name, value in config.items():
            setattr(c, name, value)
        study = cls(config=c)  # create an default instance
        for name, value in dct.items():
            if hasattr(study, name):
                setattr(study, name, value)
            else:
                logging.warning(f"Did not setattr for {name} as not in obj")


        model_index = dict()
        for key, path in study.model_index.items():  # iterate over the paths (which is how we represent the models)
            if path.exists():
                logging.debug(f"Loading model from {path}")
                # verify key is as expected.
                model = Model.load_model(path)  # load the model.
                got_key = study.key_for_model(model)
                if key != got_key: # key changed.
                    logging.warning(f"Key has changed from {key} to {got_key} for model {model}")
                model_index[got_key] = model
            else:
                logging.warning(f"Failed to find {path}.")

        study.model_index = model_index  # overwrite the index

        return study

    def runConfig(self, filename:typing.Optional[pathlib.Path]=None,
                  scale:bool=True,add_cost:bool=True) -> OptClimConfigVn2|OptClimConfigVn3:
        """
        **copy** self.config and add parameters and obs to it. Modified config is returned
        :param filename - pathlib to file (or None). Will override filepath in new config
        :param scale -- passed to self.cost to compute cost when add_cost is True
        :param add_cost -- add cost to the returned configuration
        :return: modified config. The following methods will work:
            finalConfig.parameters() -- returns the parameters for each model simulation
            finalConfig.simObs() -- returns the simulated observations for each model simulation.
        """
        newConfig = self.config.copy(filename=filename) # copy the config.

        params = self.params()  # get params & obs
        obs = self.obs()
        # update newConfig with obs & params. As normal all are unscaled.

        newConfig.parameters(params)
        newConfig.simObs(obs)

        if add_cost: # want to include cost. Which might be computed with scaling
            cost = self.cost(scale=scale)
            # update newConfig
            cost = newConfig.cost(cost)
            # work out best directory by finding minimum cost.
            bestEval = None
            if len(cost) > 0:
                bestEval = cost.idxmin()
            newConfig.setv('bestEval', bestEval)  # best evaluation

        return newConfig

        # do some plotting

    def plot(self, figName='monitor', monitorFile=None):
        """
        plot cost, normalised parameter & obs values for runs.
          Could do with a clean up and make better use of pandas plotting
        :param figName: name of figure to make -- default is monitor
        :param monitorFile: name of file to save figure to if not None. Default is None
        :return: figure, (costAxis, paramAxis, obsAxis)

        Note needs matplotlib
        """
        # get a bunch of annoying messages from matplotlib so turn them off...
        logging.getLogger('matplotlib.font_manager').disabled = True
        cost = self.cost()
        if len(cost) == 0:
            return  # nothing to plot
        fig, ax = plt.subplots(3, 1, num=figName, figsize=[8.3, 11.7], sharex='col', clear=True)
        (costAx, paramAx, obsAx) = ax  # name the axis .
        cmap = copy.copy(plt.cm.get_cmap('RdYlGn'))
        cmap.set_under('skyblue')
        cmap.set_over('black')
        try:  # now to plot
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

            parm = self.params(normalize=True)
            parm = parm.reindex(index=cost.index)  # reorder
            X = np.arange(-0.5, parm.shape[1])
            Y = np.arange(-0.5, parm.shape[0])  # want first iteration at 0.0
            cm = paramAx.pcolormesh(Y, X, parm.T.values, cmap=cmap, vmin=0.0, vmax=1.)  # make a colormesh
            a = paramAx.set_yticks(np.arange(0, len(parm.columns)))
            a = paramAx.set_yticklabels(parm.columns)
            a = paramAx.set_title("Normalised Parameter")
            a = paramAx.axvline(minp, linestyle='dashed', linewidth=2, color='gray')

            # plot norm obs
            obs = self.obs(scale=True,normalize=True)
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
            for cmm, title in zip([cmO, cm], ['Obs', 'Param']):
                cb = fig.colorbar(cmm, ax=costAx, orientation='horizontal', fraction=0.05, extend='both')
                cb.ax.set_xlabel(title)
            # fig.colorbar(cm, ax=costAx, orientation='horizontal', fraction=0.05,extend='both')
        except  TypeError:  # get this when nothing to plot
            print("Nothing to plot")
            pass

        fig.suptitle(self.name + " " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), fontsize='small',
                     y=0.99)
        fig.tight_layout()
        fig.show()
        if monitorFile is not None:
            fig.savefig(str(monitorFile))  # save the figure
        return fig, (costAx, paramAx, obsAx)

    # end of Study
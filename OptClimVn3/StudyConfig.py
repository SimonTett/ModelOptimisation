"""
Provides classes and methods suitable for manipulating study configurations.  Includes two useful classes:
    fileDict which designed to provide some kind of permanent store across  invocations of the framework. 

    studyConfig which inherits from fileDict but "knows" about configuration for study and provides methods to
       manipulate it. Idea being to insulate rest of code from details of configuration file and allow complex processing 
       if necessary.

    TODO: Sort out constraint use in studyConfig. Current implementation is problematic as constraint turns up
      from time to time without any obvious reason why

    TODO: Consider major re-factorisation of studyConfig -- it has been accreting functionality in an unplanned way.
     Perhaps splitting into core and derived values might be the way to go.



"""
from __future__ import annotations

import typing
import copy
import datetime
import json
import logging
import os
import pathlib
import re

import generic_json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray  # TODO -- consider removing dependence on xarray

__version__ = '3.0.0'
my_logger = logging.getLogger(f"OPTCLIM.{__name__}")
type_fixed_param_function: typing.TypeAlias = typing.Callable[
    [dict[typing.Hashable, 'Model.Model']], typing.Optional[pd.Series]]


# functions available to everything.

def readConfig(filename, **kwargs):
    """
    Read a configuration and return object of the appropriate version.
    :param filename: name of file (or filepath) to read.
     all kw args passes onto the creation
    :return: Configuration of appropriate type
    """
    path = pathlib.Path(os.path.expandvars(filename)).expanduser()

    if os.path.isfile(path) is False:
        raise IOError("File %s not found" % filename)
    config = dictFile(filename=path)  # read configuration using rather dumb object.
    config = config.to_StudyConfig(**kwargs)  # convert dictFile to appropriate StudyConfig.
    return config


def getDefault(dct, key, default):
    """
    :param dct: dictionary to read value from
    :param key: key to use
    :param default: default value to use if not set or None
    :return: value from dct if provided, default if not provided or None
    """
    value = dct.get(key, default)
    if value is None:  # value is none then use default
        value = default
    return value  # and return the value


# class to deal with numpy in JSON.

class NumpyEncoder(json.JSONEncoder):
    # this code a fairly straight import from Mike's numpyJSON.
    # which provides numpy aware JSON encoder and decoder.
    # TODO extend to work with pandas datarray and series
    # which can be done using .to_json() method then adding info about type
    # in json output.
    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a OrderedDict
        holding dtype, shape and the data
        """
        if isinstance(obj, np.ndarray):
            data_list = obj.tolist()
            return dict(__ndarray__=data_list, dtype=str(obj.dtype), shape=obj.shape)
        elif 'dtype' in dir(obj):
            return dict(__npdatum__=str(obj), dtype=str(obj.dtype))
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(obj)


def decode(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict):
        if '__ndarray__' in dct:
            data = np.array(dct['__ndarray__']).reshape(dct['shape'])  ## extract data
            return data  # already been converted etc so just return it
    elif '__npdatum__' in dct:
        data = dct['__npdatum__']
        return data

    return dict(dct)


class dictFile(dict):
    """
    extends dict to prove save and load methods.
    """
    _filename: pathlib.Path
    Config: typing.Dict

    def __init__(self, filename: typing.Optional[typing.Union[str, pathlib.Path]] = None,
                 Config_dct: typing.Optional[dict] = None):
        """
        Initialise dictFile object from file or from dict (which is fairly awful way)
        :param filename -- name of file to load
        :param Config_dct used to initialise Config
        :return: just call dict init and returns that.
        """

        if filename is not None:
            if Config_dct is not None:
                raise ValueError("Do not specify Config_dct and file")
            path = pathlib.Path(os.path.expandvars(filename)).expanduser()
            try:
                with path.open(mode='r') as fp:
                    Config_dct = json.load(fp, object_hook=decode)  ##,object_pairs_hook=collections.OrderedDict)
            except IOError:  # I/O problem
                Config_dct = dict()  # make it empty

        else:
            path = None

        self.Config = Config_dct
        self._filename = path

    def print_keys(self):
        """
        Print out top-level keys
        :return: None
        """

        for k, v in self.Config.items():
            print("%s: %s" % (k, v))

    def save(self, filename=None, verbose=False):
        """
        saves dict to specified filename.
        :param filename to save file to. Optional and if not provided will use
           private variable filename in object. If provided filename that self uses
           subsequently will be this filename
        :param verbose (optional, default False). If true be verbose

        :return: status of save.
        """
        if filename is not None:
            self._filename = filename  # overwrite filename
        if filename is None and hasattr(self, "_filename"):
            filename = self._filename
        if filename is None: raise ValueError("Provide a filename")
        # simply write object to JSON file. Note no error trapping
        if os.path.isfile(filename): os.remove(filename)

        with open(filename, 'wt') as fp:
            json.dump(self.Config, fp, cls=NumpyEncoder, indent=4)
        if verbose:
            print("Wrote to %s" % filename)
        # return None

    def getv(self, key: typing.Hashable, default=None):
        """
        Return values from Config component
        :return: value or None if key not defined
        """
        return self.Config.get(key, default)

    def setv(self, key, var):
        """

        :param key: Key to be set
        :param var: value to be set
        :return: modifies self
        """
        self.Config[key] = var

    def increment(self, key, increment):
        """

        :param key: key for which value we are incrementing
        :param increment: how much we increment key by. Be careful what you increment..
        :return: incremented value
        """
        default = 0  # default value
        if isinstance(increment, (list, tuple)):  # incrementing using a list or tuple so default becomes a list
            default = []  # empty list
        value = self.getv(key, default=default)  # get the value with default
        value += increment  # increment the value
        self.setv(key, value)  # set it
        return value  # return it

    def fileName(self):
        """
        :return: path to filename
        """
        return self._filename

    def to_StudyConfig(self, **kwargs) -> OptClimConfigVn3:
        """
        Convert a dictfile to an OptClimVn3
        :param config_dir: dictfile rep of config.
        :return: OptClimConfigVn3 (or with development later version)
        """

        vn = self.getv('version', default=None)
        if isinstance(vn, str):
            vn = float(vn)  # convert to float as version stored as string

        # use appropriate generator fn
        if (vn is None) or (vn < 3):
            raise ValueError(
                f"Version = {vn} in {self._filename}. Update to version 3 or greater to work with current software...")
        elif vn < 4:  # version 3 config
            config = OptClimConfigVn3(self, **kwargs)
        else:
            raise Exception(f"Version must be < 4. Write new code for {vn}!")
        return config


class OptClimConfig(dictFile):
    """
    Class to provide methods for working with OptclimConfig file.
    Inherits from dictFile which provides methods to load and save configurations
    All methods, where sensible, return a pandas dataframe or series.

    When extending this care needs to be taken that data in DataFrames/Series is float rather than objects.
    Objects rather than floats will be created if mixed strings/floats are made into dataframes/series.
    Most of the code avoids this by extracting only the required parameters and observations.

    """

    def __init__(self, config):
        """
        Return OptClimConfig object -- likely called using readConfig
        :param config: -- a dictFile configuration. Information will be copied from here. 
        """
        self.bestEval = None
        self._covariances = None  # where we store the covariances. #TODO -- include this in the covariances.
        self.__dict__.update(config.__dict__)  # just copy the values across!

    def version(self):
        """

        :return: the current version of the configuration file.
        """
        return self.getv('version', None)

    def name(self, name=None):
        """
        :param name -- if not None then set name in self to name
        :return: a short name for the configuration file
        """
        if name is not None:
            self.setv("Name", name)
        name = self.getv("Name")
        if name is None:
            # use filename to work out name
            name = self.fileName().stem
            # name = 'Unknown'
        return name

    def paramNames(self, paramNames=None):
        """
        :param paramNames (default None). If not none set ParamList in config to this. paramNames should be a list.
        :return: a list of parameter names from the configuration files
        """
        if paramNames is not None:
            self.Config['study']['ParamList'] = paramNames[:]  # copy paramnames in
        return self.Config['study']['ParamList'][:]  # return a copy of the list.

    def obsNames(self, obsNames=None, add_constraint=True):
        """
        :param (optional default is None). If not None set obsNames to values
        :param (optional default = False) add_constraint -- if True add the constraint name to the list of obs
        :return: a list  of observation names from the configuration files
        """
        if obsNames is None:
            obs = self.getv('study', {}).get('ObsList', [])[:]  # return a copy of the array.
        else:
            self.getv('study', {})['ObsList'] = list(obsNames)[:]  # need to copy not have a reference.
            # TODO -- decide what to do with scales at this point
            obs = list(obsNames)

        if add_constraint and self.constraint():  # adding constraint and its defined.
            if self.constraintName() not in obs:
                obs.append(self.constraintName())

        # check for duplicates
        dup_obs = set([ob for ob in obs if obs.count(ob) > 1])

        if len(dup_obs) > 0:
            msg = "Have duplicate observations for :" + " ".join(dup_obs)
            raise ValueError(msg)

        return obs

    def paramRanges(self, paramNames=None):
        """
        :param paramNames -- a list of the parameters to extract ranges for.
        If not supplied the paramNames method will be used.
        :return: a pandas array with rows names minParam, maxParam, rangeParam
        """
        if paramNames is None:
            paramNames = self.paramNames()
        param = pd.DataFrame(self.Config['minmax'],
                             index=['minParam', 'maxParam'])
        param = param.loc[:, paramNames]
        # just keep the parameters we want getting rid of comments etc in the JSON file
        param = param.astype(float)
        param.loc['rangeParam', :] = param.loc['maxParam', :] - param.loc['minParam', :]  # compute range
        return param

    def standardParam(self, paramNames: typing.List[str] = None,
                      values: typing.Optional[typing.Dict] = None,
                      all: bool = False,
                      scale: bool = False):
        """
        Extract standard parameter values for study
        :param paramNames: Optional names of parameters to use.
        :param values (default None). If values is not None set standardParam to values passed in and return them
        :param all (default False). Return the whole default parameters dict.
        :param scale (default False). If True scale the values by the range
        :return: pandas series (unless all is set)
        """
        if paramNames is None:
            paramNames = self.paramNames()
        if values is not None:
            # check have Parameters and if not create it.
            if 'standardModel' not in self.Config:
                self.Config["standardModel"] = dict()  # create it as an empty ordered dict.

            self.Config["standardModel"]['paramValues'] = values
        else:
            values = self.Config['standardModel']['paramValues']
        if all:
            return values

        svalues = pd.Series([values.get(k, np.nan) for k in paramNames], index=paramNames)
        if scale:
            range = self.paramRanges(paramNames=paramNames)  # get param range
            svalues = (svalues - range.loc['minParam', :]) / range.loc['rangeParam', :]
        return svalues.rename(self.name())

    def standardObs(self, obsNames=None, scale=False):
        """
        Extract standard obs values for study
        :param obsNames: Optional names of observations to use.
        :param scale (default False). If true scale values
        :return: pandas series
        """
        if obsNames is None:
            obsNames = self.obsNames()
        svalues = pd.Series([self.Config['standardModel']['SimulatedValues'].get(k, np.nan) for k in obsNames],
                            index=obsNames, )
        if scale:
            svalues = svalues * self.scales(obsNames=obsNames)  # maybe scale it.

        return svalues.rename(self.name())

    def standardConstraint(self, constraintName=None):
        """
        Extract constraint value from standard model values.
        :param constraintName (default None): value to extract
        :return: constrain as a pandas Series
        """
        if constraintName is None:
            constraintName = [self.constraintName()]

        series = pd.Series([self.Config['standardModel']['SimulatedValues'].get(k, np.nan) for k in constraintName],
                           index=constraintName)
        return series.rename(self.name())

    def beginParam(self, paramNames=None, scale=False):
        """
        get the begin parameter values for the study. These are specified in the JSON file in begin block
        Any values not specified use the standard values
        :param paramNames: Optional names of parameters to use.
        :param scale (default False). If True scale parameters by their range so 0 is minimum and 1 is maximum
        :return: pandas series of begin parameter values.
        """
        if paramNames is None:  paramNames = self.paramNames()
        begin = {}  # empty dict
        standard = self.standardParam(paramNames=paramNames, scale=scale)
        scaleRange = self.Config['begin'].get("paramScale")  # want to scale ranges?
        range = self.paramRanges(paramNames=paramNames)  # get param range

        for p in paramNames:  # list below is probably rather slow and could be sped up!
            begin[p] = self.Config['begin']['paramValues'].get(p)
            if begin[p] is not None:
                if scaleRange:  # values are specified as 0-1
                    begin[p] = begin[p] * range.loc['rangeParam', p] + range.loc['minParam', p]
                if scale:  # want to return them in range 0-1
                    begin[p] = (begin[p] - range.loc['minParam', p]) / range.loc['rangeParam', p]

            if begin[p] is None: begin[p] = standard[p]

        begin = pd.Series(begin, dtype=float)[paramNames]  # order in the same way for everything.

        # verify values are within range
        if scale:
            L = begin.gt(1.0) | begin.lt(0.0)
        else:
            L = range.loc['maxParam', :].lt(begin) | begin.lt(range.loc['minParam', :])

        if np.any(L):
            print(1)
            print("begin: \n", begin)
            print("range: \n", range)
            print("Parameters out of range", begin[L].index)
            raise ValueError("Parameters out of range: ")

        return begin.astype(float).rename(self.name())

    def firstOptimise(self):

        firstOptimise = self.Config['begin'].get('firstOptimiseStep', 'GN')
        if firstOptimise is None:
            firstOptimise = 'GN'
        return firstOptimise

    def studyVars(self, studyDir):
        """
        :param studyDir the study directory
        :return: a studyVars object for the current study
        """
        file = self.studyFileStore()
        file = os.path.join(studyDir, file)
        return dictFile(filename=file)

    def studyFileStore(self):
        """
        Get the name of the file used to store and update information for the whole study.

        :return: filename relative to studyDir
        """
        fileStore = self.Config['begin'].get('studyFileStore', 'study_vars.json')
        if fileStore is None:
            fileStore = 'study_vars.json'
        return fileStore

    def cacheFile(self):
        """
        Get the pathname relative to the study directory of the cache file.
        This file holds information (directories in current design) on model simulation directories.
        :return:  filename relative to studyDir
        """

        fileStore = self.Config['begin'].get('studyCacheFile', 'cache_file.json')
        if fileStore is None:
            fileStore = 'cache_file.json'
        return fileStore

    def targets(self, targets: None | pd.Series = None, obsNames=None, scale=False):
        """
        Get (or set) the target values for specific obs names. If some obsNames not present in target then a ValueError is raised
        :param targets -- tgt values as pandas array or None.
         Note that this does not change the obsNames which should be set using obsNames()
        :param obsNames: list of observations to use. If not None the self.obsNames() will be used.
        :param scale: if True  scale target values by scaling
        :return: target values as a pandas series
        """
        if obsNames is None:
            obsNames = self.obsNames()
        if targets is None:
            tgt = self.getv('targets')
        else:
            tgt = targets.to_dict().copy()  # make sure we make a copy.
            self.setv('targets', tgt)
        missing = set(obsNames) - set(tgt.keys())
        if len(missing):
            raise ValueError("Missing some obs = " + " ".join(missing))
        tvalues = pd.Series({obs: tgt[obs] for obs in obsNames})  # extract the required obsNames
        if scale:
            tvalues = tvalues * self.scales(obsNames=obsNames)
        return tvalues.rename(self.name())

    def constraint(self, value=None):
        """
        Work out if doing constrained optimisation or not.
        :return: True if constraint active.
       
        """
        # TODO: Consider just using mu -- if it is set then use it.
        opt = self.optimise()  # get optimisation block
        if value is not None:  # want to set it
            opt = self.optimise(sigma=value)

        constraint = opt.get('sigma', False)
        return constraint

    def constraintName(self, constraintName=None):
        """
        Extract the name of the constraint (if any) variable from the configuration file

        :return: the name of the constraint variable
        """
        if constraintName is not None:
            self.Config['study']['constraintName'] = constraintName

        return self.Config['study'].get("constraintName")

    def constraintTarget(self, constraintName=None, scale=False):
        """
        extract the value for the constraint variable target returning it as a pandas series.
        :param constraintName (optional) if not specified then constraintName() will be used
        :param scale (optional; default False) if True then constraint value will be appropriately scaled.
        :return: constraint target as a pandas series
        """

        if constraintName is None:
            constraintName = self.constraintName()  # get constraint name
        return self.targets(obsNames=[constraintName],
                            scale=scale)  # wrap name as list and use targets method to get value

    def scales(self, scalings=None, obsNames=None):
        """
        Get the scales for specified obsNames
        :param scalings If not None then set scales to these values. Should be dict like.
            Any value of 1 will not be stored.
        :param obsNames: list of observations to use. If None then self.obsNames will be used
        :return: scales as a pandas series
        """
        if scalings is not None:
            self.Config['scalings'] = {key: value for key, value in scalings.items() if value != 1.0}
        scalings = self.Config.get('scalings', {})
        if obsNames is None:
            obsNames = self.obsNames()
        # TODO raise error if any of the scaling names are not in obsNames as a consistency check.
        missing = {k for k in scalings.keys() if not k.endswith("comment")} - set(obsNames)
        # removing any keys that end with "comment"

        if missing:
            raise ValueError("Following scaling keys are not in obsNames: " + " ".join(missing))
        scales = pd.Series([scalings.get(k, 1.0) for k in obsNames], index=obsNames).rename(self.name())
        # get scalings -- if not defined set to 1.

        return scales

    def maxFails(self, value=None):
        """
        :param value. If not None set maxFails to this.

        :return: the maximum number of fails allowed. If nothing set then return 0.
        """
        if value is not None:
            optimise = self.optimise(maxFails=value)
        else:
            optimise = self.optimise()

        maxFails = optimise.get('maxFails', 0)

        return maxFails

    def Covariances(self, obsNames=None, trace=False, dirRewrite=None, scale=False, constraint=None, read=False,
                    CovTotal: typing.Optional[pd.DataFrame] = None,
                    CovIntVar: typing.Optional[pd.DataFrame] = None,
                    CovObsErr: typing.Optional[pd.DataFrame] = None):
        """
        If CovObsErr and CovIntVar are both specified then CovTotal will be computed from
        CovObsErr+2*CovIntVar overwriting the value of CovTotal that may have been specified.
        Unspecified values will be set equal to None.
        If CovIntVar is not present it will be set to diag(1e-12)
        If CovTotal is not present it will be set to the identity matrix

        :param obsNames: Optional List of observations wanted and in order expected. :param trace: optional with
        default False. If True then additional output will be generated. :param dirRewrite: optional with default
        None. If set then rewrite directory names used in readCovariances. :param scale: if set true (default is
        false) then covariances are scaled by scaling factors derived from self.scales() :param constraint: is set to
        True  (default is None) then add constraint weighting into Covariances. If set to None then if configuration
        asks for constraint (study.sigma set True) then will be set True. If set False then no constraint will be
        set. Total and ObsErr covariances for constraint will be set to 1/(2*mu) while IntVar covariance will be set
        to 1/(100*2*mu) This is applied when data is returned. If you don't want constraint set then see
        StudyConfig.constraint method.

        :param CovTotal -- if not None set CovTotal to  value overwriting any existing values.
           Should be a pandas datarrray
        :param CovIntVar -- if not None set CovIntVar to value overwriting any existing values.
        :param CovObsErr -- if not None set CovObsErr to value overwriting any existing values.
         In setting values you can make CovTotal inconsistent with CovIntVar and CovObsErr.
         This method does not check this. You should also pass in unscaled values as scaling is applied on data
        No diagonalisation  is done to these value. Constraint, if requested, added on.
         Scaling is then applied to these values (or original values)
        :param read (default True)-- if True use readCovariances to read in the data in essence resetting covariances
        :return: a dictionary containing CovTotal,CovIntVar, CovObsErr-  the covariance matrices and ancillary data.
         None if not present.

        TODO: Modify internal var covariance matrix as depends on ensemble size.
        """

        keys = ['CovTotal', 'CovIntVar', 'CovObsErr']  # names of covariance matrices
        useConstraint = constraint

        if constraint is None:
            useConstraint = self.constraint()  # work out if we have a constraint or not.

        if obsNames is None:
            obsNames = self.obsNames(add_constraint=False)  # don't want constraint here. Included later
        cov = {}  # empty dict to return things in
        covInfo = self.getv('study', {}).get('covariance')
        # extract the covariance matrix and optionally diagonalise it.
        readData = (self._covariances is None) or read
        bad_reads = []
        if readData:
            for k in keys:
                fname = covInfo.get(k, None)
                if fname is not None:  # specified in the configuration file so read it
                    try:
                        cov[k] = self.readCovariances(fname, obsNames=obsNames, trace=trace, dirRewrite=dirRewrite)
                        cov[k + "File"] = fname  # store the filename
                        if cov[k] is not None:  # got some thing to further process
                            if covInfo.get(k + "Diagonalise", False):  # want to diagonalise the covariance
                                # minor pain is that np.diag returns a numpy array so we have to remake the DataFrame
                                cov[k] = pd.DataFrame(np.diag(np.diag(cov[k])), index=obsNames, columns=obsNames,
                                                      dtype=float)
                                my_logger.info("Diagonalising " + k)
                    except ValueError as exception:  # error in readCovariance
                        bad_reads += [str(exception)]
            if len(bad_reads) > 0:  # failed somehow. Raie ValueError.
                raise ValueError("\n".join(bad_reads))
            # make total covariance from CovIntVar and CovObsErr if both are defined.
            if cov.get('CovIntVar') is not None and cov.get(
                    'CovObsErr') is not None:  # if key not defined will "get" None
                k = 'CovTotal'
                cov[k] = cov['CovObsErr'] + 2.0 * cov['CovIntVar']
                cov[k + '_info'] = 'CovTotal generated from CovObsErr and CovIntVar'
                if trace:
                    print("Computing CovTotal from CovObsErr and CovIntVar")
                if covInfo.get(k + "Diagonalise", False):  # diagonalise total covariance if requested.
                    if trace: print("Diagonalising " + k)
                    cov[k] = pd.DataFrame(np.diag(np.diag(cov['CovTotal'])), index=obsNames, columns=obsNames)

            if cov.get('CovIntVar') is None:  # make it very small!
                cov['CovIntVar'] = pd.DataFrame(1.0e-12 * np.identity(len(obsNames)),
                                                index=obsNames, columns=obsNames,
                                                dtype=float)
            if cov.get('CovTotal') is None:
                cov['CovTotal'] = pd.DataFrame(np.identity(len(obsNames)),
                                               index=obsNames, columns=obsNames,
                                               dtype=float)

                print("Warning: No covariance defined. Assuming Identity")

            self._covariances = copy.deepcopy(cov)  # store the covariances as we have read them in.
            # Need a deep copy as cov is a dict pointing to datarrays. If the dataarrays get modified then
            # they get modifed here which we don't want...
        # end of reading in data.

        # set up values from values passed in  overwriting values if necessary
        if CovTotal is not None:
            self._covariances['CovTotal'] = CovTotal.copy()
            self._covariances['CovTotal' + 'File'] = 'Overwritten '
        if CovIntVar is not None:
            self._covariances['CovIntVar'] = CovIntVar.copy()
            self._covariances['CovIntVar' + 'File'] = 'Overwritten '
        if CovObsErr is not None:
            self._covariances['CovObsErr'] = CovObsErr.copy()
            self._covariances['CovObsErr' + 'File'] = 'Overwritten '

        cov = copy.deepcopy(self._covariances)  # copy from stored covariances.
        # Need a deep copy as cov is a dict pointing to datarrays. If the dataarrays get modified then
        # they get modified here which we don't want...

        # apply constraint.
        if useConstraint:  # want to have constraint wrapped in to covariance matrices. Rather arbitrary for all but Total!
            consValue = 2.0 * self.optimise()['mu']
            consName = self.constraintName()
            for k, v in zip(keys, (consValue, consValue / 100., consValue)):
                # Include the constraint value. Rather arbitrary choice for internal variability
                if k in cov:
                    cov[k].loc[consName, :] = 0.0
                    cov[k].loc[:, consName] = 0.0
                    cov[k].loc[consName, consName] = v
        # scale data
        if scale:
            obsNames = self.obsNames(
                add_constraint=useConstraint)  # make sure we have included the constraint (if wanted) in obs
            scales = self.scales(obsNames=obsNames)

            cov_scale = pd.DataFrame(np.outer(scales, scales), index=scales.index, columns=scales.index)
            for k in keys:
                if k in cov and cov[k] is not None:
                    cov[k] = cov[k] * cov_scale
                    if trace: print("Scaling " + k)

        return cov

    def transMatrix(self, scale=False, verbose=False, minEvalue=1e-6, dataFrame=True):
        """
        Return matrix that projects data onto eigenvectors of total covariance matrix
        :param scale: (Default False) Scale covariance.
        :param verbose: (default False) Be verbose.
        :param dataFrame: wrap result up as a dataframe
        :param minEvalue: evalues less than minEvalue * max(eigenvalues) are removed. Meaning a non-square transMatrix
        :return: Transformation matrix that makes Total covariance matrix I.
        """

        # compute the matrix that diagonalises total covariance.
        cov = self.Covariances(trace=verbose, scale=scale)  # get covariances.
        errCov = cov['CovTotal']
        # compute eigenvector and eigenvalues of covariances so we can transform residual into diagonal space.
        evalue, evect = np.linalg.eigh(errCov)
        # deal with small evalues.
        crit = evalue.max() * minEvalue
        indx = evalue > crit

        transMatrix = (np.diag(evalue[indx] ** (-0.5)).dot(evect[:, indx].T))  # what we need to do to transform to
        if dataFrame:
            transMatrix = pd.DataFrame(transMatrix, index=np.arange(0, np.sum(indx)), columns=errCov.columns)
        return transMatrix

    def steps(self, steps=None, paramNames=None):
        """
        Compute perturbation  for all parameters supplied. If value specified use that. If not use 10% of the range.
        Quasi-scientific in that 10% of range is science choice but needs knowledge of the structure of the JSON file
             so in this module.
        :param steps (optional). If not None then set the steps.
        :param paramNames -- optional the parameter names for step sizes. If not defined uses self.paramNames() to work
                them out
        :return: the step sizes for the parameters as a pandas Series.
        """

        if steps is not None:
            self.Config['steps'] = steps
        if paramNames is None:
            paramNames = self.paramNames()

        param = self.paramRanges(paramNames=paramNames)
        defaultStep = 0.1 * (param.loc['maxParam', :] - param.loc['minParam', :])  # 10% of range and default cases
        pert = {}
        for p in paramNames:
            pert[p] = self.Config['steps'].get(p, defaultStep.loc[p])

        perturbation = pd.Series(pert)[paramNames]  # make sure in the same order.
        return perturbation.astype(float).rename(self.name())

    def rewriteDir(dir, dir_rewrite):
        """
         rewrite dir using keys in dir_rewrite
        :param dir: list or string of directories to be rewritten
        :param dir_rewrite:  dictionary of files to be rewritten
        :return: rewritten dict
        """
        if isinstance(dir, str):  # this is a string
            result = dir
            for k in dir_rewrite.keys():  # iterate over keys
                if k in dir:  # got some text to rewrite.
                    result = dir.replace(k, dir_rewrite[k])
                    continue  # exit all processing
        else:  # it is a list..
            result = []  # output list
            for s in dir:  # iterate over list of strings to rewrite
                rs = s
                for k in dir_rewrite.keys():  # iterate over keys
                    if k in s:  # got it
                        rs = s.replace(k, dir_rewrite[k])
                        next  # no more replacement.
                result.append(rs)  # append the rewritten text.

        return result  # return the result escaping the loop

    def readCovariances(self, covFile, obsNames=None, trace=False, dirRewrite=None):
        """
        :param covFile: Filename for the covariance matrix. Env variables and ~ expanded
        :param obsNames: List of Observations wanted from covariance file.
           If None then self.obsNames() will be used though constraint name will be omitted.
        :param trace: (optional) if set True then some handy trace info will be printed out.
        :param dirRewrite (optional) if set to something then the first key in dirRewrite that matches in covFile
              will be replaced with the element.
        :return: cov -- a covariance matrix sub-sampled to the observations

        Returns a covariance matrix from file optionally sub-sampling to named observations.
        Note if obsName is not specified ordering will be as in the file.
        """
        if obsNames is None:
            obsNames = self.obsNames(add_constraint=False)  # do not include constraint here. It gets added on later.
        use_covFile = os.path.expanduser(os.path.expandvars(covFile))
        if dirRewrite is not None:  # TODO consider removing this.
            use_covFile = self.rewriteDir(use_covFile, dirRewrite)
            if trace: print("use_covFile is ", use_covFile)

        if not (os.path.isfile(use_covFile) and os.access(use_covFile, os.R_OK)):
            print("Failed to read ", use_covFile)
            raise IOError
        # now read in covariance file
        try:
            cov = pd.read_csv(use_covFile)  # read the covariance
            cov.set_index(cov.columns, drop=False, inplace=True,
                          verify_integrity=True)  # provide index
        except ValueError:  # now likely have index
            cov = pd.read_csv(use_covFile, index_col=0)

        # verify covariance is sensible. Should not have any missing data
        if cov.isnull().sum().sum() > 0:  # got some missing
            raise ValueError(f'cov {use_covFile} has missing data. Do fix')
        # check have required obsNames.
        missing = set(obsNames) - set(cov.index)
        if len(missing):
            raise ValueError(f"Covar {covFile} Missing some obs = " + " ".join(missing))

        cov = cov.reindex(index=obsNames, columns=obsNames)  # extract the values comparing to olist

        return cov

    def postProcessScript(self, postProcessScript=None):
        """
        :param postProcessScript (defaultNone). If not None set the postProcessScript to this value
        :return: the full path for the postprocessing script
        """
        if postProcessScript is not None:
            self.Config['postProcess']["script"] = postProcessScript

        ppScript = self.Config['postProcess'].get("script",
                                                  "$OPTCLIMTOP/obs_in_nc/comp_obs.py")  # get PostProcessScript
        ppScript = os.path.expanduser(os.path.expandvars(ppScript))  # expand shell variables and home
        return ppScript

    def postProcessOutput(self, postProcessOutput=None):
        """
            :param (optional) postProcessOutput -- if not None (which is default) value will be used to set
        :return: relative  path for output from post processing script. Path is taken relative to model directory

        """
        ppProcess = self.getv('postProcess', {})
        if postProcessOutput is not None:
            ppProcess['outputPath'] = postProcessOutput
            self.setv('postProcess', ppProcess)

        ppOutput = self.getv('postProcess').get("outputPath", "observations.nc")

        return ppOutput

    def referenceConfig(self, referenceConfig=None):
        """
        :param referenceConfig  -- set referenceConfig if not None (Should be a string).
        :return: full path to the reference configuration of model being used
        """
        if referenceConfig is not None:
            self.getv('study')['referenceModelDirectory'] = referenceConfig
        modelConfigDir = self.getv('study').get('referenceModelDirectory')
        # and now expand home directories and env variables
        if modelConfigDir is not None:
            modelConfigDir = os.path.expanduser(os.path.expandvars(modelConfigDir))
        return modelConfigDir

    def optimise(self, **kwargs):
        """
        Extract and package all optimisation information into one directory
        Note this is not a copy
        :param kwargs -- added to optimise.
        :return: a dict
        """
        optimise = self.getv('optimise', {})
        if kwargs:  # Got some values. Update the optimise dict and put them back in
            optimise.update(kwargs)
            self.setv('optimise', optimise)
        return optimise

    def fixedParams(self):
        """
        :return: a dict of all the fixed parameters. All names ending _comment or called comment will be excluded 
        """

        fix = copy.copy(self.getv('Parameters').get('fixedParams', None))
        if fix is None: return {}  # nothing found so return empty dir
        # remove comment and _comment -- I suspect this would be good to do at higher level.
        for k in list(fix.keys()):
            if k == 'comment' or k.endswith('_comment'):
                fix.pop(k)  # delete the unwanted key
        #
        # deal with Nones and see if have default value. Code nicked from initParams
        paramNames = fix.keys()  # parameters we have.
        standard = self.standardParam(all=True)  # get all the standard values

        for p in paramNames:  # list below is probably rather slow and could be sped up!
            if fix[p] is None and standard.get(p) is not None:  # fixed param is None and standard is not None
                fix[p] = standard[p]  # set fixed value to standard valye

        return fix

    ## these methods to do with running the model.
    ## and so in next version should read from a "run" block.
    def runCode(self):
        """
        
        :return: the runCode (or None) 
        """
        return self.getv("runCode", None)

    def runTime(self):
        """
        
        :return: the run time (or None)
        """
        return self.getv("runTime", None)

    def modelFunction(self, fnTable):
        """
        lookup and return model creation function from value of modelName in fnTable.
        :return:the function that creates a model instance
        """

        name = self.getv("modelName", None)
        return fnTable.get(name)

    def machine_name(self):

        """
        Return name of Machine in config file.
        """

        return self.getv('machineName', None)

    def fakeFunction(self, fnTable):
        """
        Use value of fakeFunction (default is 'default') to look up and return function in fnTable

        :param: fnTable -- dict  of functions
        :return: fake function
    
        """

        name = getDefault(self.Config, 'fakeFunction', 'default')
        return fnTable.get(name)

    def GNgetset(self, name, variable=None):
        """
        Common method to set/get stuff
        :return:
        """

        GNinfo = self.getv('GNinfo', None)

        if GNinfo is None:  # no GNinfo so create it.
            GNinfo = dict()
            GNinfo['comment'] = 'Gauss-Newton Algorithm information'

        if variable is None:
            variable = GNinfo.get(name, None)
            if variable is None: return None  # no variable so return None
            if isinstance(variable, list):
                variable = np.array(variable)  # extract variable from GNinfo and convert to numpy array
        else:  # got variable so put it in the GNinfo
            try:
                GNinfo[name] = variable.tolist()
            except AttributeError:
                GNinfo[name] = variable
            self.setv('GNinfo', GNinfo)  # store it.

        return variable

    def GNjacobian(self, jacobian=None, normalise=False, constraint=False):
        """
        Set/Return the Jacobian array as an xarray DataArray
        :param jacobian: (default None). If not None should be a numpy 3D array which will be stored in the config
        :return: xarray version of the Jacobian array
        """

        jacobian = self.GNgetset('jacobian', jacobian)
        if jacobian is None:
            return None
        # jacobian by default includes constraint. if constraint is False remove it.
        if not constraint:
            jacobian = jacobian[..., 0:-1]

        paramNames = self.paramNames()
        obsNames = self.obsNames(add_constraint=constraint)
        itern = np.arange(0, jacobian.shape[0])
        name = self.name()
        # TODO -- fix this for random case.
        jacobian = xarray.DataArray(jacobian,
                                    coords={'Iteration': itern, 'Parameter': paramNames, 'Observation': obsNames},
                                    dims=['Iteration', 'Parameter', 'Observation'], name=name)

        # want to normalise ?
        if normalise:
            rng = self.paramRanges(paramNames=jacobian.Parameter.values)
            rng = xarray.DataArray(rng.loc['rangeParam'], {'Parameter': rng.columns}, dims=['Parameter'])
            jacobian = jacobian * rng

        return jacobian

    def GNhessian(self, hessian=None):
        """
        Return the Hessian array as an xarray DataArray
        :param hessian: (default None). If not None should be a numpy 3D array which will be stored in the config
        :return: xarray version of the Hessian array
        """

        hessian = self.GNgetset('hessian', hessian)
        if hessian is None:
            return None
        paramNames = self.paramNames()
        itern = np.arange(0, hessian.shape[0])
        name = self.name()
        hessian = xarray.DataArray(hessian,
                                   coords={'Iteration': itern, 'Parameter': paramNames, 'Parameter_2': paramNames},
                                   dims=['Iteration', 'Parameter', 'Parameter_2'], name=name)

        return hessian

    def GNparams(self, params=None):

        """
        Return (and optionally set) the best parameter values as an xarray from the GN optimisation.
        :param params: (default None). If not None should be a 2D numpy array which will be stored in the config.
        :return: parameter array as xarray.DataArray
        """

        params = self.GNgetset('params', params)
        if params is None: return None
        paramNames = self.paramNames()
        iterCount = np.arange(0, params.shape[0])
        name = self.name()
        params = xarray.DataArray(params, coords={'Iteration': iterCount, 'Parameter': paramNames},
                                  dims=['Iteration', 'Parameter'], name=name)

        return params

    def GNcost(self, cost=None):
        """
        Return (and optionally set) the cost values as a pandas Series from the GN optimisation.
        :param cost:  (default None)). If not None should a 1D numpy array which will be stored in the config.
        :return: cost as a pandas.Series
        """
        cost = self.GNgetset('cost', cost)
        if cost is None:
            return None
        iterCount = np.arange(0, cost.shape[0])
        name = self.name()
        cost = pd.Series(cost, index=iterCount, name=name)
        cost.index.rename('Iteration', inplace=True)

        return cost

    def GNstatus(self, status=None):
        """
        Return and optionally set the the status of the Gauss-Newton aglorithm
        :param status: if not None should be a string which will stored.
        """
        status = self.GNgetset('status', status)
        return status

    def GNalpha(self, alpha=None):
        """
        Return (and optionally set) the alpha values as a pandas Series from the N optimisation.
        :param alpha:  (default None)). If not None should a 1D numpy array which will be stored in the config.
        :return: alpha as a pandas.Series
        """
        alpha = self.GNgetset('alpha', alpha)
        if alpha is None: return None
        iterCount = np.arange(0, alpha.shape[0])
        name = self.name()
        alpha = pd.Series(alpha, index=iterCount, name=name)
        alpha.index.rename('Iteration', inplace=True)

        return alpha

    def GNparamErrCovar(self, normalise=False, constraint=True, Jac=None):
        r"""
        Compute the covariance for parameter error.
        Theory is that (to first order) $$\vect{\delta O}= \matrix{J}\vect{\delta p}$$
          where $$\delta O$$ are perturbations in observations,
          $$\delta p$$ are perturbations to parameters and J is the Jacobian.
          Then multiply by J^+ (the pseudo-inverse) to give $$\vect{\delta p} = \matrix{J}^+  \vect{\delta O}$$.
        Then the covariance is $$\matrix{J}^+ C {\matrix{J}^+}^T$$

         or alternatively with covariance...
         $$P=(J^TC^{-1}J)^{-1}J^TC^{-1}$$
        :param normalise (default = True) -- compute the normalised covariance error (fraction of range)
        :return: covariance of parameters
        """
        import numpy.linalg as linalg
        invFn = linalg.inv
        # need both the Jacobian and Covariance matrix.
        covar = self.Covariances(constraint=constraint)
        covar = covar['CovTotal'].values

        invCov = invFn(covar)
        if Jac is None:  # compute Jacobian
            Jac = self.GNjacobian(normalise=normalise, constraint=constraint).isel(
                Iteration=-1).T  # extract the last Jacobian
        P = invFn(Jac.values.T.dot(invCov).dot(Jac.values)).dot(Jac.values.T).dot(invCov)  # transform matrix.
        # paramCovar=linalg.inv(Jac.T.dot(invCov).dot(Jac))*Jac.T*

        # JacPinv = linalg.pinv(Jac.values)
        # paramCovar= JacPinv.dot(covar).dot(JacPinv.T)
        paramCovar = P.dot(covar).dot(P.T)
        # now wrap paramCovar up as a dataframe.
        paramCovar = pd.DataFrame(paramCovar, index=Jac.Parameter, columns=Jac.Parameter)

        return paramCovar

    def optimumParams(self, paramNames=None, normalise=False, **kwargs):
        """
        Set/get the optimum parameters.  (VN1 variant)
        :param values: default None -- if set then parameter values in configuration gets updated with values
        :param paramNames -- return specified parameters.
        :param scale (default False). If True scale parameters over range (0 is min, 1 is max)
        :return: values as pandas series.
        """
        raise NotImplementedError("vn1 optimum parameters no longer supported")
        # TODO merge this with vn2 code which could be done using  if self.version >= 2: etc
        if hasattr(kwargs, 'scale'):
            raise Exception("scale no longer supported use normalise")  # scale replaced with normalised 14/7/19
        if len(kwargs) > 0:  # set the values
            self.Config['study']['optimumParams'] = kwargs
        if paramNames is None:
            paramNames = self.paramNames()
        stdParams = self.standardParam(paramNames=paramNames)
        values = self.Config['study'].get('optimumParams', None)
        values = {k: values.get(k, stdParams[k]) for k in paramNames}
        if values is not None:
            values = pd.Series(values)  # wrap it as a pandas series.
        if normalise:
            range = self.paramRanges(paramNames=paramNames)  # get param range
            values = (values - range.loc['minParam', :]) / range.loc['rangeParam', :]

        return values

    def GNsimulatedObs(self, set_obs=None, obsNames=None):
        """
        Set/get the simulated observations from the best value in each iteration of the Gauss Newton algorithm.
        :param obsNames: Names of observations wanted
        :param set_obs: a pandas array or xarray (or anything that has a to_dict method) that will set values of the observations.
          Should only be the best values on each iteration. Perhaps this routne should have all values... 
        :return: Observations as a pandas array.

        """

        if set_obs is not None:  # set the value
            # check we've got GNinfo
            GNinfo = self.getv('GNinfo', {})
            # should modify to use GNgetset but that assumes numpy array.
            GNinfo['SimulatedObs'] = set_obs.to_dict()
            self.setv('GNinfo', GNinfo)  # and set it.
        # get the value.

        result = pd.DataFrame(self.getv('GNinfo')['SimulatedObs'])  # should trigger an error if the obs does not exist.
        name = self.name()
        # convert if needed from unicode.
        try:
            name = name.encode('utf8')
        except TypeError:
            pass

        # result.rename(name,inplace=True) # rename it.

        if obsNames is None:
            obsNames = self.obsNames()
        result = result.loc[:, obsNames]

        return result  # and return the result.

    def GNoptimumObs(self, obsNames=None):
        """
        :param obsNames -- the observations to extract
        :return: Optimum observations as a pandas series. 
        """

        obs = self.GNsimulatedObs(obsNames=obsNames)
        return obs.iloc[-1, :]  # return the last obs.

    # Generic stuff
    def alg_info(self, **argv):
        """
        : named arguments used to set algorithm information in config
           Information is stored in config information based on name of algorithm
        :return:  specific algorithm information.
        """

        alg_info_name = self.optimise()['algorithm'].upper() + "_information"
        info = self.getv(alg_info_name, {})
        if len(info) == 0:  # empty so set it.
            self.setv(alg_info_name, {})
            info = self.getv(alg_info_name)

        if argv:  # got some values to set
            for key, item in argv.items():  # iterate over any arguments we have
                info[key] = item
            self.setv(alg_info_name, info)  # set it

        return info

    def set_dataFrameInfo(self, **setargs):
        """
        Put pandas dataframe into alg_info
        :param **setargs -- arguments to set.
        example: config.set_dataFrameInfo(result=result,flaming=flaming,success=pd.Dataframe(data))
        """
        # raise NotImplementedError("Write self test")
        alg_info = self.alg_info()
        for k, v in setargs.items():
            alg_info['__' + k + '_type'] = str(type(v))  # store the type
            alg_info[k] = v.to_json(orient='split')  # set it

    def get_dataFrameInfo(self, input_keys: typing.Union[typing.List[str], str], dtype=None):

        """
        Return pandas dataframes from alg_info

        :param input_keys a *list* of variables to return or a string.
        If a string then this is interpreted as a key.
        If list then each element is a key.
        returns a tuple containing dataframes. If tuple has one element then a dataframe is returned.
        :param dtype If not None the type to convert the dataframe to using astype method.
        Example: config.get_dataFrameInfo(['transJacobian','Flames'])



        """
        alg_info = self.alg_info()
        result = []
        typ_lookup = {"<class 'pandas.core.frame.DataFrame'>": "frame",
                      "<class 'pandas.core.series.Series'>": "series"}

        if isinstance(input_keys, str):
            keys = [input_keys]
        else:
            keys = input_keys
        for arg in keys:  # get back the value
            try:
                typ = alg_info['__' + arg + '_type']  # get the type(series or dataframe)
                df = pd.read_json(alg_info[arg], orient='split', typ=typ_lookup[typ])
                if dtype is not None:
                    df = df.astype(dtype)
            except KeyError:  # failed to find so set to None
                df = None
            result.append(df)
        if len(keys) == 1:  # deal with single list.
            return result[0]
        else:
            return tuple(result)

    def diagnosticInfo(self, diagnostic=None):
        """

        :param diagnostic: A pandas dataframe. If set add this to the configuration
        :return: the values as a  dataframe
        """

        if diagnostic is not None:
            self.set_dataFrameInfo(diagnosticInfo=diagnostic)

        return self.get_dataFrameInfo(['diagnostic'])

    def jacobian(self, jacobian=None):
        """
        Set jacobian if set. Return jacobian as pandas dataframe
        :param jacobian: a pandas dataframe containing the jacobian information
        :return: jacobian -- converted to a float (the json conversion loses type info)
        """

        if jacobian is not None:
            self.set_dataFrameInfo(jacobian=jacobian)

        return self.get_dataFrameInfo('jacobian', dtype=float)

    def transJacobian(self, transJacobian=None):
        """
        Set transformed jacobian if set. Return transJacobian as pandas dataframe
        :param transJacobian: a pandas dataframe containing the transformed jacobian information.
           If not None will set values.
        :return: transformed jacobian (J.Trans^T) where Trans is the transform matrix (see transMatrix)
               converted to a float (the json conversion loses type info)
        """

        if transJacobian is not None:
            self.set_dataFrameInfo(transJacobian=transJacobian)

        return self.get_dataFrameInfo('transJacobian', dtype=float)

    def hessian(self, hessian=None):
        """
        :param hessian. If not None set hessian to this value. Should be a pandas dataframe.
        return -- returns the hessian matrix converted to a float (the json conversion loses type info).
        """

        if hessian is not None:
            self.set_dataFrameInfo(hessian=hessian)

        return self.get_dataFrameInfo('hessian', dtype=float)

    def paramErrCovar(self, normalise=True, useCov=False):
        r"""
        Compute the covariance for parameter error. TODO -- consider removing.
          to store in the final json file. Then regardless of algorithm we can get it out again...
        Theory is that (to first order) \vect{\delta O}= \matrix{J}\vect{\delta p}
          where \delta O are perturbations in observations, \delta p are perturbations to parameters and J is the Jacobian.
          Then multiple by J^+ (the pseudo-inverse) to give \vect{\delta p} = \matrix{J}^+  \vect{\delta O}.
        Then the covariance is \matrix{J}^+ C {\matrix{J}^+}^T

         or alternatively with covariance...
         P=(J^TC^{-1}J)^{-1}J^TC^{-1}
        :param normalise (default = True) -- compute the normalised covariance error (fraction of range)
        :param useCov
        :return: covariance of parameters
        """
        import numpy.linalg as linalg
        invFn = linalg.pinv
        if useCov:
            raise NotImplementedError("Not implemented inv covariance -- get them from GNparamErrCov")
        if not normalise:
            raise NotImplementedError("Normalise must be true")
        Jac = self.jacobian()
        Hess = Jac.T.dot(Jac)
        P = invFn(Hess).dot(Jac.T)
        paramCovar = P.dot(P.T)
        # now wrap paramCovar up as a dataframe.
        paramCovar = pd.DataFrame(paramCovar, index=Jac.columns, columns=Jac.columns)

        return paramCovar

    def best_obs(self, best_obs=None):
        """
        Set best_obs if set. Return best_obs as pandas series
        :param best_obs: a pandas series containing the best observations info
        :return: best_obs
        """

        if best_obs is not None:
            # self.set_dataFrameInfo(best_obs=best_obs.to_json(orient='split'))
            self.set_dataFrameInfo(best_obs=best_obs)

        b = self.get_dataFrameInfo('best_obs')
        # return pd.read_json(self.alg_info().get('best_obs'), orient='split',
        #                    typ='series')  # magic because it is a series not a dataframe.
        return b

    def DFOLSinfo(self, diagnosticInfo=None):
        """

        :param diagnosticInfo: (optional -- defult None). If set add ths to the confguration
        :return: the values as a  dataframe. Temp code for backward compatability
        """

        import warnings
        warnings.warn("Use diagnosticInfo not DFOLSinfo")
        return self.diagnosticInfo(diagnostic=diagnosticInfo)

    def DFOLS_config(self, dfolsConfig=None):
        """

        Extract (and optionally set) the DFOLS specific information
        :param dfolsConfig -- a dict of dfoLS config data which replaces existing values
        :return: the configuration which (like most python is a ptr tothe data. If you change
        """
        if dfolsConfig is not None:
            self.optimise()['dfols'] = copy.deepcopy(dfolsConfig)  # copy input.
        return self.optimise().get('dfols', {})

    def DFOLS_userParams(self, userParams=None, updateParams=None):
        """

        Extract the user parameters for DFOLS
        :param userParams (default None) -- default user params which are overwritten by config values.
        :param updateParams (default None) -- modify userparams in configuration.
        :return: user parameters
        """

        if userParams is None:
            result = {}.copy()  # make sure get different empty dict.
        else:
            result = copy.deepcopy(userParams)

        if updateParams is not None:
            # breaks
            self.DFOLS_config()['namedSettings'] = copy.deepcopy(
                updateParams)  # deep copy so stored dict not linked to var
        namedSettings = self.DFOLS_config().get('namedSettings', {})
        for k in namedSettings.keys():  # loop over keys
            if not re.search(r'_comment\s*$', k):  # not a comment
                result[k] = namedSettings[k]

        return result

    # stuff for BOBYQA - merged from Sophy's code

    def BOBYQAinfo(self, diagnosticInfo=None):
        """

        :param diagnosticInfo: (optional -- defult None). If set add ths to the confguration
        :return: the vlues as a  dataframe
        """
        BOBYQAinfo = self.getv('BOBYQA',
                               dict())  # should modify to use GNgetset but that assumes numpy array.
        if diagnosticInfo is not None:
            BOBYQAinfo['diagnostic'] = diagnosticInfo.to_json(orient='split')

        self.setv("BOBYQA", BOBYQAinfo)  # store the diagnostic info.

        return pd.read_json(BOBYQAinfo.get('diagnostic'), orient='split')

    def simObs(self, simObs=None, best=False):
        """

        :param simObs: optional -- default None and if passed should be a pandas dataframe
        :param best: optional-- defaultFalse. If true return the obs for the best iteration
        :return:  dataframe of simulated observations in order in which they were ran (which may not be
           the same as the algorithm sees them)

        TODO: Modify to use set/get_dataframe
        """
        if simObs is not None:
            # add fake index to preserve index.
            self.setv('simObs', simObs.to_dict())

        sObs = pd.DataFrame(self.getv('simObs'))
        if best:
            sObs = sObs.loc[self.bestEval, :]

        return sObs

    def parameters(self, parameters=None, best=False, normalise=False):
        """

        :param parameters: optional -- default None and if passed should a pandas dataframe
        : param (optional) normalise -- default False. If True the parameters 
           returned are normalised to 0 (min allowed) to 1 (max allowed)
        :return:  dataframe of simulated observations
        :TODO rewrite to use set/get_dataframe.
        """
        if parameters is not None:
            self.setv('parameters', parameters.to_dict())

        # params = pd.read_json(self.getv('parameters'), orient='index')
        params = pd.DataFrame(self.getv('parameters'))
        if best:
            params = params.loc[self.bestEval, :]

        if normalise:
            prange = self.paramRanges(paramNames=params.columns)  # get param range
            prange.fillna({'minParam': 0, 'rangeParam': 1}, inplace=True)
            params = (params - prange.loc['minParam', :]) / prange.loc['rangeParam', :]

        return params

    def cost(self, cost=None, best=False):
        """

        :param cost: (optional) the cost for each model evaluation as a pandas series.
        :param best (optional) -- if True return only the best value
        :return: cost as a pandas series.
        """

        if cost is not None:
            # for python prior to 3.5 ??? when dicts are unordered
            # shoudl also store as a list the index.
            # then get that and use to index things
            # probably need similar for parameters and obs to it is more generic.
            self.setv('costd', cost.to_dict())

        cost = pd.Series(self.getv('costd'), dtype=float)
        cost.name = 'Cost'
        bestEval = self.getv('bestEval')
        if best:
            cost = cost.loc[bestEval]

        return cost

    def directories(self, directories=None, best=False):
        """

        :param directories: (optional -- default = None). The directory (currently full path) where each simulation was run as a pandas series
        :param best (optional) -- if True return only the best value
        :return: directories as a pandas series
        """
        if directories is not None:
            self.setv('dirValues', directories.to_json())

        directories = self.getv('dirValues', [])
        if len(directories) == 0: return pd.Series()
        directories = pd.read_json(directories, typ='series')

        if best:
            directories = directories[self.bestEval]

        return directories

    def ensembleSize(self, value=None):
        """
        Get/set the ensemble size
        :param value: If not None (default is None) then set the value for number of ensemble members in each simln
        :return: ensemble size
        """

        study = self.getv('study', {})  # get study dict setting to empty dict if not set
        if value is not None:  # value provided? If so set and save it.
            study['ensembleSize'] = value
            self.setv('study', study)

        ensembleSize = study.get('ensembleSize')
        # default is ensemble size of 1.
        if ensembleSize is None:
            ensembleSize = 1  # do this way as JSON file can have null.

        return ensembleSize

    def maxRuns(self, value=None):
        """
        Get/set the maximum numbers of runs
        :param value: If not None (default is None) then set the value f
        :return: maximum number of runs to be done.
        """

        if value is not None:  # value provided? If so set and save it.
            self.setv('maxRuns', value)

        mx = self.getv('maxRuns')
        if mx < 1:
            raise ValueError(f"maxRuns {mx} < 1")
        # no default -- up to calling application to decide.
        return mx

    def runJacobian(self, jacobian=None, normalise=False, scale=True):
        """
        Store or return jacobian.
        :param jacobian: jacobian as dataset with mean (Jacobian) and variances (Jacobian_var)
        :return: jacobian as xaarry dataset.
        TODO : Consider removing dependance on xarray -- which means storing as dataframe.
        TODO: Remove -- probably not needed.
        """

        if jacobian is None:
            result = xarray.Dataset.from_dict(self.getv('runJacobian'))

        else:
            dict = jacobian.to_dict()  # convert to dict
            self.setv('runJacobian', dict)  # store it
            result = jacobian  # just return it as it came in

        if scale:  ## apply scalings
            scaling = self.scales(obsNames=result.Observation.values).to_xarray().rename({'index': 'Observation'})
            try:
                result['Jacobian'] = result.Jacobian * scaling
                result['Jacobian_var'] = result.Jacobian_var * (scaling ** 2)
            except AttributeError:
                pass

        if normalise:  # normalise by range.
            paramRange = self.paramRanges(paramNames=result.parameter.values)
            range = paramRange.loc['rangeParam', :].T.to_xarray().rename({'index': 'parameter'})
            try:
                result['Jacobian'] = result.Jacobian * range
                result['Jacobian_var'] = result.Jacobian_var * (range ** 2)
            except AttributeError:
                pass

        return result

    def baseRunID(self, value: typing.Optional[str] = None) -> str:
        """
        Return (and optionally set) the baseRunID.
        If not defined return 'aa'
        :return: baseRunId
        """

        if value is not None:
            self.setv('baseRunID', value)

        return self.getv('baseRunID', default='aa')  # nothing defined make it aa

    def maxDigits(self, value=None):
        """
        Return (and optionally set) the maxDigits for model run names

        If not defined return None -- up to the user of this to deal with it.

        :param value -- value to be set -- default is not to set it

        """

        if value is not None:
            self.setv('maxDigits', value)

        return self.getv('maxDigits', default=3)  # nothing defined -- make it 3.

    def copy(self, filename: typing.Optional[pathlib.Path] = None) -> OptClimConfig:
        """
        :param filename (optional default None): Name of filename to save to.
        :return: a copy of the configuration but set _fileName  to avoid accidental overwriting of file.
        """

        result = copy.deepcopy(self)  # copy the config

        if (filename is not None) and (filename.exists()) and (result._filename.samefile(filename)):
            raise ValueError("Copy has same filename as original.")
        result._filename = filename  # set filename
        return result


class OptClimConfigVn2(OptClimConfig):
    """
    Version 2 of OptClimConfig -- modify OptClimConfig methods and see OptClimConfig.__init__() for
    most of setup.
    """

    #

    def paramNames(self, paramNames=None):
        """
        :param paramNames -- a set of paramNames to overwrite existing values. Should be a list
        :return: a list of parameter names from the configuration files
        """

        keys = self.getv('Parameters', {})['initParams'].keys()  # return a copy of the list.
        # remove comment keys
        keys = [k for k in keys if 'comment' not in k]
        return keys

    def standardParam(self,
                      values: typing.Optional[typing.Dict] = None,
                      paramNames: typing.List[str] = None,
                      all: bool = False,
                      scale: bool = False):

        """
        Extract standard parameter values for study
        :param paramNames: Optional names of parameters to use.
        :param values (default None). If values is not None set standardParams to values passed in and return them
        :param scale (default False). If True scale values by parameter range so that 0 is min, 1 is max
        :param all (default False). Return the whole parameters' dict. Used when want to copy
        :return: pandas series (unless all is set)
        """
        if paramNames is None:   paramNames = self.paramNames()
        if values is not None:
            # check have Parameters and if not create it.
            if 'Parameters' not in self.Config:
                self.Config['Parameters'] = dict()  # create it as an empty ordered dict.

            self.Config['Parameters']['defaultParams'] = values
        else:
            values = self.Config['Parameters']['defaultParams']
        if all:
            return values

        svalues = pd.Series([values.get(k, np.nan) for k in paramNames], index=paramNames)
        if scale:
            range = self.paramRanges(paramNames=paramNames)  # get param range
            svalues = (svalues - range.loc['minParam', :]) / range.loc['rangeParam', :]
        return svalues.rename(self.name())

    def paramRanges(self, paramNames=None, values: dict = None):
        """
        :param paramNames -- a list of the parameters to extract ranges for.
        If not supplied the paramNames method will be used.
        :return: a pandas array with rows names minParam, maxParam, rangeParam
        """
        if paramNames is None: paramNames = self.paramNames()
        if values is not None:
            self.Config['Parameters']['minmax'] = values
        param = pd.DataFrame(self.Config['Parameters']['minmax'],
                             index=['minParam', 'maxParam'])
        # work out which names we have to avoid complaints from pandas.
        names = [p for p in paramNames if p in param.columns]
        param = param.loc[:, names]  # just keep the parameters we want and have.
        param = param.astype(float)
        param.loc['rangeParam', :] = param.loc['maxParam', :] - param.loc['minParam', :]  # compute range
        return param

    def beginParam(self,
                   begin: typing.Optional[pd.Series] = None,
                   paramNames: typing.Optional[typing.List[str]] = None,
                   scale: bool = False) -> pd.Series:

        """
        get the begin parameter values for the study. These are specified in the JSON file in initial block
        Any values not specified use the standard values
        :param begin -- if not None then set begin values to this.
           No scaling is done  and initScale will be set False.
        :param paramNames: Optional names of parameters to use.
        :param scale (default False). If True scale parameters by their range so 0 is minimum and 1 is maximum
        :return: pandas series of begin parameter values.
        """

        if begin is None:  # No values specified
            begin = self.Config['Parameters'].get('initParams')

        else:  # set them
            begin = begin.to_dict()  # convert from pandas series to dict for internal storage
            self.Config['Parameters']['initParams'] = begin
            self.Config['Parameters']["initScale"] = False

        scaleRange = self.Config['Parameters'].get("initScale")  # want to scale ranges?
        if paramNames is None:
            paramNames = self.paramNames()
        beginValues = {}  # empty dict
        standard = self.standardParam(paramNames=paramNames)

        range = self.paramRanges(paramNames=paramNames)  # get param range

        for p in paramNames:  # list below is probably rather slow and could be sped up!
            beginValues[p] = begin.get(p)
            if beginValues[p] is None:
                beginValues[p] = standard[p]  # Will trigger an error if standard[p] does not exist
            else:
                if scaleRange:  # values are specified as 0-1
                    beginValues[p] = beginValues[p] * range.loc['rangeParam', p] + range.loc['minParam', p]
            if scale:  # want to return params  in range 0-1
                beginValues[p] = (beginValues[p] - range.loc['minParam', p]) / range.loc['rangeParam', p]

        beginValues = pd.Series(beginValues, dtype=float)[paramNames]  # order in the same way for everything.

        # verify values are within range
        if scale:
            L = beginValues.gt(1.0) | beginValues.lt(0.0)
        else:
            L = range.loc['maxParam', :].lt(beginValues) | beginValues.lt(range.loc['minParam', :])

        if np.any(L):
            my_logger.warning("L  \n", L)
            my_logger.warning(f"begin: {beginValues}")
            my_logger.warning(f"range: {range}")
            my_logger.warning(f"Parameters out of range: {beginValues[L].index}")
            raise ValueError("Parameters out of range: ")

        return beginValues.astype(float).rename(self.name())

    def optimumParams(self, paramNames=None, normalise=False, **kwargs):
        """
        Set/get the optimum parameters. (VN2 version)
        :param normalise (default False). If True then normalise parameters.
        :param values: default None -- if set then parameter values in configuration gets updated with values
        :param paramNames -- name of parameters
        :return: values as pandas series.
        """

        if paramNames is None:  paramNames = self.paramNames()

        if len(kwargs) > 0:  # set the values
            self.Config['Parameters']['optimumParams'] = kwargs
            # add defaults for ones we have not got.
        default = self.standardParam(paramNames=paramNames)

        values = self.Config['Parameters'].get('optimumParams', None)
        if values is None:
            return pd.Series(np.full(len(paramNames), np.nan), index=paramNames)

        for k in paramNames:
            values[k] = self.Config['Parameters']['optimumParams'].get(k, default[k])
            # set it to std value if we don't have it...Note will trigger error if default not got value

        ##values = {k: values.get(k,None) for k in paramNames} # extract the requested params setting values to None if we don't have them.
        values = pd.Series(values)[paramNames]  # wrap it as a pandas series and order it appropriately.

        if normalise:
            range = self.paramRanges(paramNames=paramNames)  # get param range
            values = (values - range.loc['minParam', :]) / range.loc['rangeParam', :]

        return values.rename(self.name())

    def steps(self,
              paramNames: typing.Optional[typing.List[str]] = None,
              normalise: bool = False,
              steps: typing.Optional[dict] = None):
        """
        Compute perturbation  for all parameters supplied. If value specified use that. If not use 10% of the range.
        Quasi-scientific in that 10% of range is science choice but needs knowledge of the structure of the JSON file
             so in this module.

        :param paramNames -- optional the parameter names for step sizes. If not defined uses self.paramNames() to work
                them out
        :param normalise -- if True normalise by the range regardless of the value of scale in the configuration
        :param steps (optional). If not None then update the steps with values here. Should be a pandas series or dict.
              These values should be consistent with scale_step which determines the ... Which you can pass in.
        :return: the step sizes for the parameters as a pandas Series.
        """

        if paramNames is None: paramNames = self.paramNames()
        if steps is not None:  # setting steps. This is update. Existing values may still be there...
            for p, v in steps.items():
                self.Config['Parameters']['steps'][p] = v

        param = self.paramRanges(paramNames=paramNames)
        defaultStep = 0.1 * param.loc['rangeParam', :]  # 10% of range and default cases
        pert = {}
        scale = self.Config['Parameters']['steps'].get('scale_steps', False)
        # this is probably a bad design choice. TODO modify for vn3. Make steps the variable for all steps
        # and self.Config['Parameters']['scale_steps'] be scaling (or not)
        if scale:  # scaling so set default step to 0.1
            defaultStep.loc[:] = 0.1
        for p in paramNames:
            pert[p] = self.Config['Parameters']['steps'].get(p, defaultStep.loc[p])
            if pert[p] is None: pert[p] = defaultStep.loc[p]
        perturbation = pd.Series(pert)[paramNames]  # make sure in the same order.

        # scale everything if needed.
        if scale:  # scale set means steps specified as fraction of range so convert them
            # to abs values.
            perturbation = perturbation * param.loc['rangeParam', :]
        if normalise:
            # need to normalise data
            perturbation = perturbation / param.loc['rangeParam', :]

        return perturbation.astype(float).rename(self.name())

    # do some plotting
    def plot(self, figName='monitor', monitorFile=None):
        """
        plot cost, normalised parameter & obs values for runs.
        :param figName: name of figure to make -- default is monitor
        :param monitorFile: name of file to save figure to if not None. Default is None
        :return: figure, (costAxis, paramAxis, obsAxis)

        Note needs matplotlib
        """
        # get a bunch of annoying messages from matplotlib so turn them off...
        cost = self.cost()
        if len(cost) == 0:
            return  # nothing to plot
        try:
            # recent versions of matplotlib allow clear argment.
            fig, ax = plt.subplots(3, 1, num=figName, figsize=[8.3, 11.7], sharex='col', clear=True)
        except TypeError:
            fig, ax = plt.subplots(3, 1, num=figName, figsize=[8.3, 11.7], sharex='col')

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

            parm = self.parameters(normalise=True)
            parm = parm.reindex(index=cost.index)  # reorder
            X = np.arange(-0.5, parm.shape[1])
            Y = np.arange(-0.5, parm.shape[0])  # want first iteration at 0.0
            cm = paramAx.pcolormesh(Y, X, parm.T.values, cmap=cmap, vmin=0.0, vmax=1.)  # make a colormesh
            a = paramAx.set_yticks(np.arange(0, len(parm.columns)))
            a = paramAx.set_yticklabels(parm.columns)
            a = paramAx.set_title("Normalised Parameter")
            a = paramAx.axvline(minp, linestyle='dashed', linewidth=2, color='gray')

            # plot norm obs

            resid = self.simObs() - self.targets(scale=True)
            resid = resid.reindex(index=cost.index)  # reorder

            cov = self.Covariances(scale=True)['CovTotal']
            scale = np.sqrt(np.diag(cov))
            scale = pd.Series(scale, index=cov.index)
            normObs = resid / scale
            X = np.arange(-0.5, normObs.shape[1])
            Y = np.arange(-0.5, normObs.shape[0])
            cmO = obsAx.pcolormesh(Y, X, normObs.T.values, vmin=-4, vmax=4, cmap=cmap)
            a = obsAx.set_yticks(np.arange(0, len(normObs.columns)))
            a = obsAx.set_yticklabels(normObs.columns, fontsize='x-small')
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

        fig.suptitle(self.name() + " " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), fontsize='small', y=0.99)
        fig.tight_layout()
        fig.show()
        if monitorFile is not None:
            fig.savefig(str(monitorFile))  # save the figure
        return fig, (costAx, paramAx, obsAx)

    def extractDoc(self):
        """
        Extract documentation from configuration. Done through finding any key that ends in "_comment"
        :return: configuration with key -- the name and doc the doc string
        """

        def extract_doc_list(inputList, tgt_end='_comment'):
            """
            Extract comments from list or list like things
            :param inputList: input list
            :param tgt_end:  what is a comment (anythoing ending in this)
            :return: extract list.
            """

            doc = list()
            for v in inputList:
                if isinstance(v, dict):
                    ddoc = extract_doc_dict(v, tgt_end=tgt_end)

                    if ddoc is not None:
                        doc.append(ddoc)

                elif isinstance(v, list):
                    ddoc = extract_doc_list(v, tgt_end=tgt_end)
                    if ddoc is not None:
                        doc.append(ddoc)

            # breakpoint()
            # raise NotImplementedError("No support yet for lists")
            if len(doc) == 0:
                doc = None
            return doc

        def extract_doc_dict(inputDict, tgt_end='_comment'):

            doc = dict()

            for key, v in inputDict.items():
                if isinstance(v, dict):
                    ddoc = extract_doc_dict(v, tgt_end=tgt_end)
                    # see if have a comment key and if so add it in with name k_doc
                    if v.get('comment'):
                        doc[key + "_doc"] = v.get('comment')
                    if ddoc is not None:
                        doc[key] = ddoc
                elif isinstance(v, list):
                    ddoc = extract_doc_list(v, tgt_end=tgt_end)
                    if ddoc is not None:
                        doc[key] = ddoc
                elif key.endswith(tgt_end):  # a comment
                    new_key = key[0:-len(tgt_end)]
                    doc[new_key] = v

            if len(doc) == 0:
                doc = None

            return doc

        return extract_doc_dict(self.Config)

    def printDoc(self, stream=None, width=120):
        """
        Print out documentation within a study configuration. Uses extractDoc and pretty print.
        :param stream -- stream to print to -- see print.pprint for details
        :param width: -- width of screen for printing. Default is 120.
        :return: Nada
        """
        import pprint

        pprint.pprint(self.extractDoc(), stream=stream, width=width, compact=True, sort_dicts=False)


class OptClimConfigVn3(OptClimConfigVn2):
    """
    Vn3 of OptClimConfig.
    1) have runTime() and runCode() methods only work with runInfo block -- vn1/2 methods work with names in top level dict.
    2) Have generic way of dealing with dataframes and make all methods use and
       return pandas series or dataframes as appropriate.
    """

    def __init__(self, config: dictFile, check: bool = True):
        """
        Process all INCLUDE stuff
        Call super class __init__ method and then add comment_end attribute
        set to "_comment"

        :param config -- configuration used to initialise
        :param check -- if True check that parameters and observations are self-consistent
        """
        includes = dict()
        for key, value in config.Config.items():
            if isinstance(value, str) and value.startswith("INCLUDE "):
                my_logger.debug(f"Include from {value}")
                inc, pth = value.split(maxsplit=1)
                includes[key + "_INCLUDE_comment"] = pth  # raw path so can see what done

                pth = os.path.expanduser(os.path.expandvars(pth))
                pth = pathlib.Path(pth)
                my_logger.debug(f"Reading in {pth}")
                if not pth.exists():
                    raise ValueError(f"{pth} does not exist.")
                with open(pth, 'rt') as fp:
                    dct = json.load(fp)
                includes[key] = dct
        # now have a bunch of stuff in includes which we will use to update config.
        if (len(includes) > 0):
            my_logger.info(f"Updating the following keys: {' '.join(includes.keys())}")
            config.Config.update(includes)

        super().__init__(config)  # call super class init
        self.comment_end = '_comment'  # define what a comment looks like.
        # potentially convert dumped strings back to dataframes. Bit hacky
        # TODO when fix StudyConfig to use same generic json machinery as rest of code then
        # update here..
        # deal with covariances. If we have _covariance_matrices then we need to convert them to a dataframe
        # from the encoded version.
        cov = self.getv('_covariance_matrices', {})  # shallow copy
        for k, v in cov.items():
            if isinstance(v, dict) and 'dataframe' in v:  # we encoded a dataframe when we wrote it. So decode it
                self.Config['_covariance_matrices'][k] = self.dict2cov(v)
                my_logger.debug(f"covariance {k} converted to dataframe")
        cov = self.Covariances()  #  read in covariances in case we don't have them.
        if check:
            self.check()  # check we are OK

    def __eq__(self, other: OptClimConfigVn3) -> bool:
        """
        Work out if two OptClimConfigVn3 objects are identical. Compares the two Config attributes.
        :param other: another OptClimConfigVn3
        :return: True if equal or False if not.
        """
        if type(self) != type(other):
            print(f"self type: {type(self)} not the same as {type(other)}")
            return False
        ok = True
        for k, v in self.Config.items():
            if k not in other.Config:
                my_logger.warning(f"Failed to find {k} in other")
                ok = False
                break  # exit the loop

            if k == '_covariance_matrices':  # special for covariances.
                for k2, v2 in v.items():
                    if isinstance(v2, pd.DataFrame):
                        ok2 = v2.equals(other.Config[k][k2])
                        if not ok2:
                            my_logger.info(f"Dataframes {k}/{k2} differ")
                    else:
                        ok2 = (v2 == other.Config[k][k2])
                        if not ok2:
                            my_logger.info(f" {k}/{k2} differ")
            else:
                ok2 = (v == other.Config[k])

            ok = ok and ok2

        return ok

    @classmethod
    def expand(cls, filestr: typing.Optional[str]) -> typing.Optional[pathlib.Path]:
        """
        Expand any env vars, convert to path and then expand any user constructs. (Copy from model_base which does things we don't want to do!)
        :param filestr: path like string or None
        :return:expanded path or, if filestr is None, None.
        """
        if filestr is None:
            return None
        path = os.path.expandvars(filestr)
        path = pathlib.Path(path).expanduser()
        return path

    @classmethod
    def cov2dict(cls, cov: pd.DataFrame) -> dict:
        """
        Convert covariance (as a pandas dataframe) to a dict containing scale factor
          and converted scaled dataframe
        :param cov:covariance
        :return:jsonised version of df & scale factor.
           Values will be scaled by 1/abs(min) excluding zero values.
        """
        acov = np.abs(cov.values)
        scale = 1.0 / np.min(acov[acov > 0])  # min abs value where cov > 0. If everything 0 will fail.
        dct = dict()
        dct['dataframe'] = (cov * scale).to_dict(orient='tight')  # scale and convert to json
        dct['scale'] = scale  # store the scale
        return dct

    @classmethod
    def dict2cov(cls, dct: dict) -> pd.DataFrame:
        """
        Convert a dict representation back to a dataframe.
        :param dct: dict to be be convered. SHould contain scale and dataframe keys
        :return: Decoded dataframe
        """
        scale = dct.pop('scale')
        df = pd.DataFrame.from_dict(dct.pop('dataframe'), orient='tight')
        df /= scale  # rescale it.
        return df

    def save(self,
             filename: typing.Optional[typing.Union[str, pathlib.Path]] = None,
             verbose: bool = False) -> None:
        """
        saves dict to specified filename.
        :param filename to save file to. Optional and if not provided will use
           private variable filename in object. If provided filename that self uses
           subsequently will be this filename. Will save covariances into  output file.
        :param verbose (optional, default False).
            Does not do anything -- just for compatibility with older versions of code.

        :return: Nade
        """
        if filename is not None:
            self._filename = filename  # overwrite filename
        if filename is None and hasattr(self, "_filename"):
            filename = self._filename
        if filename is None: raise ValueError("Provide a filename")

        # convert covariance matrices to json so we can write them out.
        # Will need to convert back on read.
        dct = self.Config.copy()
        cov = dct['_covariance_matrices']  # saving on typing!
        json_cov = cov.copy()
        for k, v in cov.items():
            if isinstance(v, pd.DataFrame):
                my_logger.debug(f"Covar key: {k} converting  to jsonable dict")
                json_cov[k] = self.cov2dict(v)
        dct['_covariance_matrices'] = json_cov  # overwrite
        with open(filename, 'wt') as fp:  # write it out.
            json.dump(dct, fp, cls=NumpyEncoder, indent=4)
        my_logger.info(f"Wrote config to {filename}")

    def normalise(self, pandas_obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Normalize a parameters by range such that min is 0 and max is 1
        :param pandas_obj: dataframe or series of parameters to normalise
        :return: normalised values
        """
        pnames = pandas_obj.index if isinstance(pandas_obj, pd.Series) else pandas_obj.columns
        range = self.paramRanges(paramNames=pnames)  # get param range
        nvalues = (pandas_obj - range.loc['minParam', :]) / range.loc['rangeParam', :]
        return nvalues

    def best_obs(self, best_obs: typing.Optional[pd.Series] = None) -> pd.Series:
        """
        Set best_obs if set. Return best_obs as pandas series with name from self
        :param best_obs: a pandas series containing the best observations info
        :return: best_obs
        """

        if best_obs is not None:
            self.extra_alg_info(best_obs=best_obs.to_dict())

        best = pd.Series(self.extra_alg_info('best_obs')['best_obs']).rename(self.name())

        return best

    def extra_alg_info(self, *getargs: typing.Union[list[str], str], **setargs: dict) -> dict:
        """
        Store or get extra stuff in alg_info
        :param getargs -- arguments to get.
        :param **setargs -- arguments to set.
        example: config.set_dataFrameInfo(result=result,flaming=flaming,success=pd.Dataframe(data))
        """
        # raise NotImplementedError("Write self test")
        result = dict()
        alg_info = self.alg_info()
        for k, v in setargs.items():
            alg_info[k] = copy.deepcopy(v)  # store it
        for k in getargs:
            result[k] = copy.deepcopy(alg_info[k])

        return result

    def optimumParams(self,
                      paramNames: typing.Optional[list[str]] = None,
                      normalise: bool = False,
                      optimum: typing.Optional[pd.Series] = None):
        """
        Set/get the optimum parameters. (VN3 version)
        :param normalise (default False). If True then normalise parameters.
        :optimum -- optimal parameter values as panda Series
        :param paramNames -- name of parameters
        :return: values as pandas series.
        """

        if paramNames is None:  paramNames = self.paramNames()

        if optimum is not None:  # set the values
            self.Config['Parameters']['optimumParams'] = optimum.to_dict()
            # add defaults for ones we have not got.
        default = pd.Series(self.standardParam(paramNames=paramNames))

        values = self.Config['Parameters'].get('optimumParams', None)
        if values is None:
            return values  # return None
        values = pd.Series(values)
        values = values.combine_first(default).reindex(paramNames)

        if normalise:
            values = self.normalise(values)

        return values.rename(self.name())

    def referenceConfig(self, referenceConfig: typing.Optional[pathlib.Path] = None) -> pathlib.Path:
        """
        :param referenceConfig  -- set referenceConfig if not None .
        :return: full path to the reference configuration of model being used
        """
        if referenceConfig is not None:
            self.getv('study')['referenceModelDirectory'] = str(referenceConfig)
        modelConfigDir = self.getv('study').get('referenceModelDirectory')
        # and now expand home directories and env variables
        if modelConfigDir is not None:
            modelConfigDir = self.expand(modelConfigDir)
        return modelConfigDir

    def dfols_solution(self,
                       solution: typing.Optional["OptimResults"] = None) -> \
            typing.Optional["dfols.solver.OptimResults"]:
        """
        Store/return solution to DFOLS run.
        :param solution: solution (from DFOLS) which if not None will be converted to
           something that can be converted to json
        :return: solution
        """
        from dfols.solver import OptimResults
        if solution is not None:
            # convert solution to something jsonable.
            conversion = generic_json.dumps(vars(solution))  # use generic_json to convert.
            self.setv('DFOLS_SOLUTION', conversion)

        soln = self.getv('DFOLS_SOLUTION', None)
        if soln is None:  # not got anything so return None.
            return soln
        dct = generic_json.loads(soln)  # now have a dict.
        soln = OptimResults(*range(0, 9))  # create empty OptimResults object
        for k, v in dct.items():  # fill in the instances
            if not hasattr(soln, k):
                my_logger.warning(f"Would like to set attr {k} in soln but does not exist")
            setattr(soln, k, v)  # regardless will set the attribute.

        return soln

    def to_dict(self) -> dict:
        """ Convert StudyConfig to a dict suitable for writing out"""
        #raise ValueError("Not expected any calls to this.")
        dct = vars(self)
        # TODO Might need to modify dct so that covariances are jsonable.
        return dct

    # stuff to do with running a model.

    def run_info(self) -> dict:
        """
        :return run_info dict
        """

        run_info = self.getv("run_info")
        if run_info is None:  # if it is None set it to an empty dict.
            self.setv("run_info", {})
            run_info = self.getv("run_info")

        return run_info

    def set_run_info(self,
                     setNone: bool = False,
                     **kwargs) -> dict:
        """
        Set named value in set
        :param setNone: If True then if value is None set it. Otherwise, only if value is not None set it.
        :param kwargs: Key, value pairs to set.
        :return:
        """
        run_info = self.run_info()
        for k, v in kwargs.items():
            if (v is not None) or setNone:
                run_info[k] = v

        return run_info

    def runCode(self, value: typing.Optional[str] = None) -> str:
        """
        value: If not None runCode will be set to this
        :return: the runCode (or None)
        """
        self.set_run_info(runCode=value)
        return self.run_info().get("runCode")

    def runTime(self, value: typing.Optional[int] = None) -> int:
        """

        :return: the run time (or None)
        """
        self.set_run_info(runTime=value)
        return self.run_info().get("runTime")

    def machine_name(self, value: typing.Optional[str] = None) -> str:

        """
        Return name of Machine in config file. Fails if not defined
        """
        raise NotImplementedError  # should not be called. Just extract run_info
        self.set_run_info(machineName=value)
        return self.run_info()['machineName']

    def model_name(self, value: typing.Optional[str] = None) -> str:
        """
        :return name of the model to be created. Should have been registered. And must exist in run_info.
        """
        self.set_run_info(modelName=value)
        return self.run_info()["modelName"]

    def module_name(self,
                    value: typing.Optional[str] = None,
                    model_name: typing.Optional[str] = None) -> str:
        """
        Returns name of module to be loaded.
         Uses run_info['module_name']. If not present (or None) then model_name will be used
        :param value: If not None then 'module_name' will be set to this value.
        :param model_name: If not None then rather than using self.model_name() then model_name will be used
        :return: Module to be loaded.
        """
        self.set_run_info(module_name=value)  # set module_name
        module_name = self.run_info().get('module_name')
        if module_name is None:
            my_logger.debug("No module_name found. Using model_name()")
            if model_name is None:
                model_name = self.model_name()
            module_name = model_name
        return module_name

    def maxRuns(self, value: typing.Optional[int] = None) -> int | None:
        """
        Get/set the maximum numbers of runs
        :param value: If not None (default is None) then set the value
        :return: maximum number of runs to be done.
        """
        # if value is not None: # got a value to set then set value
        self.set_run_info(maxRuns=value)

        mx = self.run_info().get('maxRuns', None)
        if (mx is not None) and (mx < 1):
            raise ValueError(f"maxRuns {mx} < 1")
        # no default -- up to calling application to decide.
        return mx

    def max_model_simulations(self, value: typing.Optional[int] = None) -> typing.Optional[int]:
        """
        The maximum total number of simulations that should be submitted. Used by runSubmit
          Taken from run_info/max_model_sims
        :param value: If not None set max_model_sims to this value. Should be integer > 0
        :return: The value of max_model_sims or None if not found.
        """

        self.set_run_info(max_model_simulations=value)
        mx = self.run_info().get('max_model_simulations', None)
        if (mx is not None) and (mx < 1):  # should be > 0 if provided.
            raise ValueError(f"maxRuns {mx} < 1")
        # no default -- up to calling application to decide what to do..
        return mx

    def strip_comment(self, dct: dict) -> dict:
        """
        Recursively remove all keys ending in _comment from a dct. 
        :param: dct -- dict to have all _comment keys removed. 
        Defined by self.comment_end
        :return dct with all keys ending with _comment removed.
        """
        result_dct = {}
        for key, value in dct.items():
            if isinstance(value, dict):  # a dict -- call strip_comment
                my_logger.debug(f"Copying {key} as dict")
                result_dct[key] = self.strip_comment(value)
            elif isinstance(key, str) and key.endswith(self.comment_end):
                my_logger.debug(f"Ignoring {key}")
            else:
                my_logger.debug(f"Copying {key}")
                result_dct[key] = value  # just take the value across.

        return result_dct

    def logging_config(self, cfg: typing.Optional[dict] = None) -> typing.Optional[dict]:
        """
        Extract logging configuration from studyConfig.
        You can pass this straight into logging.config.dictConfig() to set up logging.
        Users are responsible that this dict is correct etc.
        All comments will be removed. 

        This is designed for codes that make use of the library.

        :param: cfg -- if not None set the value of logging to this value.
          
        """
        if cfg is not None:  # got something so use it to set the value
            self.setv("logging", cfg)

        cfg = self.getv("logging")  # get it

        if cfg is None:  # got nothing so return None
            return None

        cfg = self.strip_comment(cfg)
        return cfg

    def beginParam(self,
                   begin: typing.Optional[pd.Series] = None,
                   paramNames: typing.Optional[typing.List[str]] = None,
                   scale: bool = False) -> pd.Series:

        """
        get the begin parameter values for the study. These are specified in the JSON file in initial block
        Any values not specified use the standard values
        :param begin -- if not None then set begin values to this.
           No scaling is done  and initScale will be set False.
        :param paramNames: Optional names of parameters to use.
        :param scale (default False). If True scale parameters by their range so 0 is minimum and 1 is maximum
        :return: pandas series of begin parameter values.
        """

        initial = self.getv('initial')

        if begin is not None:  # values to set.
            begin = begin.to_dict()  # convert from pandas series to dict for internal storage
            initial["initParams"] = begin
            initial["initScale"] = False

        begin = initial.get('initParams')
        scaleRange = initial.get("initScale")  # want to scale ranges?

        if paramNames is None:
            paramNames = self.paramNames()
        beginValues = {}  # empty dict
        standard = self.standardParam(paramNames=paramNames)

        range = self.paramRanges(paramNames=paramNames)  # get param range

        for p in paramNames:  # list below is probably rather slow and could be sped up!
            beginValues[p] = begin.get(p)
            if beginValues[p] is None:
                beginValues[p] = standard[p]  # Will trigger an error if standard[p] does not exist
            else:
                if scaleRange:  # values are specified as 0-1
                    beginValues[p] = beginValues[p] * range.loc['rangeParam', p] + range.loc['minParam', p]
            if scale:  # want to return params  in range 0-1
                beginValues[p] = (beginValues[p] - range.loc['minParam', p]) / range.loc['rangeParam', p]

        beginValues = pd.Series(beginValues, dtype=float)[paramNames]  # order in the same way for everything.

        # verify values are within range
        if scale:
            L = beginValues.gt(1.0) | beginValues.lt(0.0)
        else:
            L = range.loc['maxParam', :].lt(beginValues) | beginValues.lt(range.loc['minParam', :])

        if np.any(L):
            print("L  \n", L)
            print("begin: \n", beginValues)
            print("range: \n", range)
            print("Parameters out of range", beginValues[L].index)
            raise ValueError("Parameters out of range: ")

        return beginValues.astype(float).rename(self.name())

    def fixedParams_keys(self) -> typing.List[typing.Hashable]:
        """
        Get the keys for fixedParams.
        :return: Labels (if any) for fixed parameter configurations or [] if not a multiple configuration
        """
        fix = self.getv('initial').get('fixedParams', {})
        fix = self.strip_comment(fix)  # remove all comments.
        if self.fixed_param_function():
            keys = [k for k in fix.keys() if isinstance(fix[k], dict)]  # if value of k is  a direct then k is a label!
        else:
            keys = []

        return keys

    def fixed_param_function(self) -> typing.Optional[type_fixed_param_function]:
        """
        Extract the function from string. Note will import the module that contains the function.
        Be very careful...
        :return: function or None if no multiple_function found.
        """
        import importlib
        fn_test = self.getv('initial').get('fixedParams', {}).get('multiple_function', None)
        if fn_test is None:
            return fn_test
        module, fn_name = fn_test.rsplit('.', 1)
        mod = importlib.import_module(module)  # import the module
        fn: type_fixed_param_function = getattr(mod, fn_name)  # extract the function
        return fn

    def set_none_std(self, params: dict) -> dict:
        """
        Set any values in params that are None to the standard values.
        :param params: Parameter dict
        :return: Parameters with any None values set to standard values.
        """
        # deal with Nones and see if have default value. Code nicked from initParams
        standard = self.standardParam(all=True)  # get all the standard values
        result = dict()
        for k, v in params.items():
            if v is None and standard.get(k) is not None:
                result[k] = standard[k]  # will trigger an error if standard[k] does not exist
            else:
                result[k] = v
        return result

    def fixedParams(self) -> dict:
        """
        :return: a dict of all the fixed parameters. All names ending _comment or called comment will be excluded.
        Values set to None with standard values will be set to standard values.
        If multiple_function is set then will return a dict of dicts with the keys being the labels.

        """

        fix_config = self.getv('initial').get('fixedParams', {})
        if self.fixed_param_function():  # Have multiple_configs
            keys = self.fixedParams_keys()
            fix = dict()
            # TODO consider having reference_dir included in fixedParams.
            for key in keys:  # copy over the wanted keys stripping comments out
                fix[key] = self.set_none_std(self.strip_comment(fix_config[key]))
                my_logger.debug(f'Extracted fixed params for {key} ')
        else:
            fix = self.set_none_std(self.strip_comment(fix_config))
            my_logger.debug('Extracted fixed params')
        return fix

    def paramNames(self, paramNames: typing.Optional[list[str]] = None) -> list[str]:
        """
        :param paramNames -- a list of paramNames to overwrite existing values.
        :return: a list of parameter names from the configuration files.
           First looks for paramNames (at top level). If not found or empty list then tries initParams in initial.
        """
        # try and get paramNames from top level.
        if paramNames is not None:
            assert isinstance(paramNames, list)
            self.setv('paramNames', paramNames)  # set it
        params = self.getv('paramNames', None)
        if (params is None) or (len(params) == 0):  # try and get from initial params
            initial = self.getv('initial')
            initial = self.strip_comment(initial)
            params = list(initial['initParams'].keys())  # return a copy of the list.
        return params

    def Covariances(self,
                    obsNames: typing.Optional[typing.List[str]] = None,
                    trace: bool = False,  # TODO remove trace -- replaced with logging
                    dirRewrite: typing.Optional[typing.Dict] = None,
                    #TODO consider removing this as saved config should have covariances in.
                    scale: bool = False,
                    constraint: typing.Optional[bool] = None,
                    read: bool = False,
                    CovTotal: typing.Optional[pd.DataFrame] = None,
                    CovIntVar: typing.Optional[pd.DataFrame] = None,
                    CovObsErr: typing.Optional[pd.DataFrame] = None):
        """
        If CovObsErr and CovIntVar are both specified then CovTotal will be computed from
        CovObsErr+2*CovIntVar overwriting the value of CovTotal that may have been specified.
        Unspecified values will be set equal to None.
        If CovIntVar is not present it will be set to diag(1e-12)
        If CovTotal is not present it will be set to the identity matrix

        :param obsNames: Optional List of observations wanted and in order expected.
        :param trace: optional with default False. If True then additional output will be generated.
        :param dirRewrite: optional with default None. If set then rewrite directory names used in readCovariances.
        :param scale: if set true  then covariances are scaled by scaling factors derived from self.scales()
        :param constraint: is set to True  (default is None) then add constraint weighting into Covariances. If set to None then
           if configuration asks for constraint (study.sigma set True) then will be set True. If set False then no constraint will be set.
            Total and ObsErr covariances for constraint will be set to 1/(2*mu) while IntVar covariance will be set to 1/(100*2*mu)
            This is applied when data is returned. If you don't want constraint set then see StudyConfig.constraint method.

        :param CovTotal -- if not None set CovTotal to  value overwriting any existing values.
           Should be a pandas datarrray
        :param CovIntVar -- if not None set CovIntVar to value overwriting any existing values.
        :param CovObsErr -- if not None set CovObsErr to value overwriting any existing values.
         In setting values you can make CovTotal inconsistent with CovIntVar and CovObsErr.
         This method does not check this. You should also pass in unscaled values as scaling is applied on data
        No diagonalisation  is done to these value. Constraint, if requested, added on.
         Scaling is then applied to these values (or original values)
        :param read -- if True use readCovariances to read in the data in essence resetting covariances
        :return: a dictionary containing CovTotal,CovIntVar, CovObsErr-  the covariance matrices and ancillary data.
         None if not present. Also may modify the configuration.

        TODO: Modify internal var covariance matrix as depends on ensemble size.
        """

        matrix_key = "_covariance_matrices"  # where we store the covariance matrices.
        keys = ['CovTotal', 'CovIntVar', 'CovObsErr']  # names of covariance matrices
        useConstraint = constraint
        if constraint is None:
            useConstraint = self.constraint()  # work out if we have a constraint or not.

        if obsNames is None: obsNames = self.obsNames(
            add_constraint=False)  # don't want constraint here. Included later
        cov = {}  # empty dict to return things in
        covInfo = self.getv('study', {}).get('covariance', {})
        # extract the covariance matrix and optionally diagonalise it.
        readData = (self.getv(matrix_key, None) is None) or read
        bad_reads = []
        if readData:
            logging.info("Reading covariance matrices")
            for k in keys:
                fname = covInfo.get(k, None)
                if fname is not None:  # specified in the configuration file so read it
                    try:
                        cov[k] = self.readCovariances(fname, obsNames=obsNames, trace=trace, dirRewrite=dirRewrite)
                        cov[k + "File"] = fname  # store the filename
                        if cov[k] is not None:  # got some thing to further process
                            if covInfo.get(k + "Diagonalise", False):  # want to diagonalise the covariance
                                # minor pain is that np.diag returns a numpy array so we have to remake the DataFrame
                                cov[k] = pd.DataFrame(np.diag(np.diag(cov[k])), index=obsNames, columns=obsNames,
                                                      dtype=float)
                                my_logger.info("Diagonalising " + k)
                    except ValueError as exception:  # error in readCovariance
                        bad_reads += [str(exception)]
            if len(bad_reads) > 0:  # Failed somehow so raise ValueError.
                raise ValueError("\n".join(bad_reads))
            # make total covariance from CovIntVar and CovObsErr if both are defined.
            if (cov.get('CovIntVar') is not None and
                    cov.get('CovObsErr') is not None):  # if key not defined will "get" None
                k = 'CovTotal'
                cov[k] = cov['CovObsErr'] + 2.0 * cov['CovIntVar']
                cov[k + '_info'] = 'CovTotal generated from CovObsErr and CovIntVar'
                my_logger.info("Computing CovTotal from CovObsErr and CovIntVar")
                if covInfo.get(k + "Diagonalise", False):  # diagonalise total covariance if requested.
                    my_logger.debug("Diagonalising " + k)
                    cov[k] = pd.DataFrame(np.diag(np.diag(cov['CovTotal'])), index=obsNames, columns=obsNames)
            for k, value in zip(['CovIntVar', 'CovObsErr', 'CovTotal'], [1e-12, 1, 1]):
                if cov.get(k) is None:  # Set it to something
                    my_logger.warning(f"{k} not set so setting to diag {value}")
                    cov[k] = pd.DataFrame(value * np.identity(len(obsNames)),
                                          index=obsNames, columns=obsNames,
                                          dtype=float)

            self.setv(matrix_key, cov)  # store the covariances as we have read them in.
        # end of reading in data.
        # overwrite if values passed in.

        # set up values from values passed in  overwriting values if necessary
        cov = self.getv(matrix_key)
        if CovTotal is not None:
            my_logger.debug("Setting covTotal")
            cov['CovTotal'] = CovTotal
            cov['CovTotal' + 'File'] = 'Overwritten '
        if CovIntVar is not None:
            my_logger.debug("Setting covIntVar")
            cov['CovIntVar'] = CovIntVar
            cov['CovIntVar' + 'File'] = 'Overwritten '
        if CovObsErr is not None:
            my_logger.debug("Setting covObsErr")
            cov['CovObsErr'] = CovObsErr
            cov['CovObsErr' + 'File'] = 'Overwritten '

        cov = copy.deepcopy(self.getv(matrix_key))  # copy from stored covariances.
        # Need a deep copy as cov is a dict pointing to datarrays. As the dataarrays get modified then
        # that would modify the underlying cached values.

        # apply constraint.
        if useConstraint:
            # want to have constraint wrapped in to covariance matrices. Rather arbitrary for all but
            # Total!
            consValue = 2.0 * self.optimise()['mu']
            consName = self.constraintName()
            for k, v in zip(keys, (consValue, consValue / 100., consValue)):
                # Include the constraint value. Rather arbitrary choice for internal variability
                if k in cov:
                    cov[k].loc[consName, :] = 0.0
                    cov[k].loc[:, consName] = 0.0
                    cov[k].loc[consName, consName] = v
        # scale data
        if scale:
            obsNames = self.obsNames(
                add_constraint=useConstraint)  # make sure we have included the constraint (if wanted) in obs
            scales = self.scales(obsNames=obsNames)

            cov_scale = pd.DataFrame(np.outer(scales, scales), index=scales.index, columns=scales.index)
            for k in keys:
                if k in cov and cov[k] is not None:
                    cov[k] = cov[k] * cov_scale
                    my_logger.debug(f"Scaling {k}")

        return cov

    def check_obs(self, obsNames=None):
        """
        Check that observation related stuff is OK.
          Check targets, scalings and covariances. Trapping errors and then reporting at end
        :return: True if OK/False if not.
        """
        bad = []  # where we store all the error messages
        try:
            targets = self.targets(obsNames=obsNames)
        except ValueError as exception:  # missing some errors
            bad += ['Targets have problems ' + str(exception)]

        try:
            covar = self.Covariances(obsNames=obsNames)
        except ValueError as exception:  # missing some errors
            bad += ['Covariances have problems ' + str(exception)]

        try:
            scales = self.scales(obsNames=obsNames)
        except ValueError as exception:  # missing some errors
            bad += ['scales have problems ' + str(exception)]

        if len(bad):  # something went wrong. So report all the trapped errors with a failure.
            for m in bad:
                my_logger.warning(m)

        return len(bad) == 0

    def check_params(self) -> bool:
        """
        Check parameter related configuration is consistent.
        checks begin, default and ranges
        Raises warnings if not consistent
        :return: True if OK, False if Not
        """
        bad = []
        expected_params = self.paramNames()
        my_logger.debug(f'Expected params are: {expected_params}')
        try:
            default = self.standardParam(paramNames=expected_params)
            if default.isnull().any():
                bad += ['Missing values for standard: ' + ", ".join(default[default.isnull()].index)]
        except (KeyError, ValueError):
            bad += ['Problem running standardParam']
        try:
            range = self.paramRanges(paramNames=expected_params)
            if range.isnull().any().any():
                bad += ['Missing values for range: ' + ", ".join(range.loc[:, range.isnull().any()].columns)]
        except (KeyError, ValueError):
            bad += ['Problem running paramRanges']
        try:
            begin = self.beginParam(paramNames=expected_params)
            if begin.isnull().any():
                bad += ['Missing values for begin: ' + ", ".join(begin[begin.isnull()].index)]
        except (KeyError, ValueError):  # some error
            bad += ['Problem running beginParam']

        if len(bad) > 0:
            for m in bad:
                my_logger.warning(m)

        return len(bad) == 0

    def check(self):
        """
        Check configuration is consistent. Raise ValueError if not
        :return: True if OK, False if not
        """
        OK = self.check_params() and self.check_obs()
        if not OK:
            raise ValueError("Configuration has problems")
        return OK

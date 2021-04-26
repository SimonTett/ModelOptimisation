"""
Provides classes and methods suitable for manipulating study configurations.  Includes two useful classes:
    fileDict which designed to provide some kind of permanent store across  invocations of the framework. 

    studyConfig which inherits from fileDict but "knows" about configuration for study and provides methods to
       manipulate it. Idea being to insulate rest of code from details of configuration file and allow complex processing 
       if necessary.



"""
import collections
import copy
import datetime
import json
import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray  # TODO -- consider removing dependence on xarray

__version__ = '0.3.0'
# subversion properties
subversionProperties = \
    {
        'Date': '$Date$',
        'Revision': '$Revision$',
        'Author': '$Author$',
        'HeadURL': '$HeadURL$'
    }


# functions available to everything.

def readConfig(filename, ordered=False):
    """
    Read a configuration and return object of the appropriate version.
    :param filename: name of file (or filepath) to read.
    :param ordered:  read in as a ordered dict rather than a dict. 
    :return: Configuration of appropriate type
    """
    path = pathlib.Path(os.path.expandvars(filename)).expanduser()

    if os.path.isfile(path) is False:
        raise IOError("File %s not found" % filename)
    config = dictFile(filename=path, ordered=ordered)  # read configuration using rather dumb object.
    vn = config.getv('version', default=None)
    if isinstance(vn, str):
        vn = float(vn)  # convert to float as version stored as string

    # use appropriate generator fn
    if vn is None or vn < 2:  # original configuration definition
        config = OptClimConfig(config)
    elif vn < 3:  # version 2 config
        config = OptClimConfigVn2(config)
    else:
        raise Exception("Version must be < 3")
    ## little hack for bestEval..
    try:
        if not hasattr(config, 'bestEval'):
            config.bestEval = config.cost().idxmin()
    except ValueError:
        config.bestEval = None
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
            return collections.OrderedDict(__ndarray__=data_list, dtype=str(obj.dtype), shape=obj.shape)
        elif 'dtype' in dir(obj):
            return collections.OrderedDict(__npdatum__=str(obj), dtype=str(obj.dtype))
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


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

    return collections.OrderedDict(dct)


class dictFile(dict):
    """
    extends dict to prove save and load methods.
    """

    def __init__(self, filename=None, ordered=False):
        """
        Initialise dictFile object from file
        :param (optional) filename -- name of file to load
        :param (optional) ordered -- if set True then load data as orderedDict. This is incompatable with
           decoding numpy data. TODO: Understand how to decode numpy data AND make an orderedDict
        :param *arg: arguments
        :param kw: keyword arguments
        :return: just call dict init and returns that.
        """
        if filename is not None:
            path = pathlib.Path(os.path.expandvars(filename)).expanduser()
            try:
                with path.open(mode='r') as fp:
                    if ordered:
                        dct = json.load(fp, object_pairs_hook=collections.OrderedDict)
                    else:
                        dct = json.load(fp, object_hook=decode)  ##,object_pairs_hook=collections.OrderedDict)
            except IOError:  # I/O problem
                dct = collections.OrderedDict()

        else:
            dct = collections.OrderedDict()  # make it empty ordered dict.

        self.Config = dct
        self._filename = path

    # def __getattr__(self, name):
    #     """
    #     Default way of getting values back.
    #     :param name: attribute name to be retrieved
    #     :param name: attribute name to be retrieved
    #     :return: value
    #     """
    #     # TODO -- consider making __getattr__ work if name not found -- when it returns None.
    #     # TODO -- seems to fail sometimes with massive recursion.
    #     # Disadvantage of returning None is that if key mistyped will get None so makes life more Error prone
    #     # One approach might be to have a list of names and then if key in there return None otherwise fail?
    #     if name in self:
    #         return self[name]
    #     elif name in self.Config:
    #         return self.Config[name]
    #     else:
    #         raise AttributeError(
    #             "Failed to find %s in self.Config" % (name))  # spec for __getattr__ says raise AttributeError

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
        with open(filename, 'w') as fp:
            json.dump(self.Config, fp, cls=NumpyEncoder, indent=4)
        if verbose:
            print("Wrote to %s" % filename)
        # return None

    def getv(self, key, default=None):
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

    # TODO Add __eq__ and __ne__ methods.


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
        if name is not None: self.setv("Name", name)
        name = self.getv("Name")
        if name is None:
            # use filename to work out name
            name = self.fileName().stem
            # name = 'Unknown'
        return name

    def paramNames(self):
        """
        :return: a list of parameter names from the configuration files
        """
        return self.Config['study']['ParamList'][:]  # return a copy of the list.

    def obsNames(self, add_constraint=True):
        """
        :param (optional default = False) add_constraint -- if True add the constraint name to the list of obs
        :return: a list  of observation names from the configuration files
        """
        obs = self.getv('study', {}).get('ObsList', [])[:]  # return a copy of the array.
        if add_constraint and self.constraint():  # adding constraint and its defined.
            obs.append(self.constraintName())
        return obs

    def paramRanges(self, paramNames=None):
        """
        :param paramNames -- a list of the parameters to extract ranges for.
        If not supplied the paramNames method will be used.
        :return: a pandas array with rows names minParam, maxParam, rangeParam
        """
        if paramNames is None: paramNames = self.paramNames()
        param = pd.DataFrame(self.Config['minmax'],
                             index=['minParam', 'maxParam'])
        param = param.loc[:,
                paramNames]  # just keep the parameters we want getting rid of comments etc in the JSON file
        param = param.astype(float)
        param.loc['rangeParam', :] = param.loc['maxParam', :] - param.loc['minParam', :]  # compute range
        return param

    def standardParam(self, paramNames=None, values=None, all=False, scale=False):
        """
        Extract standard parameter values for study
        :param paramNames: Optional names of parameters to use.
        :param values (default None). If values is not None set standardParam to values passed in and return them
        :param all (default False). Return the whole default parameters dict.
        :param scale (default False). If True scale the values by the range
        :return: pandas series (unless all is set)
        """
        if paramNames is None:   paramNames = self.paramNames()
        if values is not None:
            # check have Parameters and if not create it.
            if 'standardModel' not in self.Config:
                self.Config["standardModel"] = collections.OrderedDict()  # create it as an empty ordered dict.

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
        if obsNames is None:      obsNames = self.obsNames()
        svalues = pd.Series([self.Config['standardModel']['SimulatedValues'].get(k, np.nan) for k in obsNames],
                            index=obsNames, )
        if scale: svalues = svalues * self.scales(obsNames=obsNames)  # maybe scale it.

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
        if firstOptimise is None: firstOptimise = 'GN'
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
        if fileStore is None: fileStore = 'study_vars.json'
        return fileStore

    def cacheFile(self):
        """
        Get the pathname relative to the study directory of the cache file.
        This file holds information (directories in current design) on model simulation directories.
        :return:  filename relative to studyDir
        """

        fileStore = self.Config['begin'].get('studyCacheFile', 'cache_file.json')
        if fileStore is None: fileStore = 'cache_file.json'
        return fileStore

    def targets(self, obsNames=None, scale=False):
        """
        Get the target values for specific obs names
        :param obsNames: optional list of observations to use
        :param scale: optional if True (default is False) scale target values by scaling
        :return: target values as a pandas series
        """
        if obsNames is None:  obsNames = self.obsNames()
        tgt = self.getv('targets')
        tvalues = pd.Series([tgt.get(k, np.nan) for k in obsNames], index=obsNames)
        if scale: tvalues = tvalues * self.scales(obsNames=obsNames)
        return tvalues.rename(self.name())

    def constraint(self, value=None):
        """
        Work out if doing constrained optimisation or not.
        :return: True if constraint active.
       
        """
        # TODO: Consider just using mu -- if it is set then use it.
        opt = self.getv('optimise', {})  # get optimisation block
        if value is not None:  # want to set it
            opt['sigma'] = value
            self.setv('optimise', opt)  # store it back

        constraint = opt.get('sigma', False)
        return constraint

    def constraintName(self):
        """
        Extract the name of the constraint (if any) variable from the configuration file

        :return: the name of the constraint variable
        """
        return self.Config['study'].get("constraintName")

    def constraintTarget(self, constraintName=None, scale=False):
        """
        extract the value for the constraint variable target returning it as a pandas series.
        :param constraintName (optional) if not specified then constraintName() will be used
        :param scale (optional; default False) if True then constraint value will be appropriately scaled.
        :return: constraint target as a pandas series
        """

        if constraintName is None: constraintName = self.constraintName()  # get constraint name
        return self.targets(obsNames=[constraintName],
                            scale=scale)  # wrap name as list and use targets method to get value

    def scales(self, obsNames=None):
        """
        Get the scales for specified obsnamaes
        :param obsNames: optional list of observations to use
        :return: scales as a pandas series
        """
        scalings = self.Config.get('scalings', {})
        if obsNames is None: obsNames = self.obsNames()
        scales = pd.Series([scalings.get(k, 1.0) for k in obsNames], index=obsNames).rename(self.name())
        # get scalings -- if not defined set to 1.
        return scales

    def maxFails(self):
        """

        :return: the maximum number of fails allowed. If nothign set then return 0.
        """
        optimise = self.Config.get('optimise', {})

        maxFails = optimise.get('maxFails', 0)
        if maxFails is None: maxFails = 0
        return maxFails

    def Covariances(self, obsNames=None, trace=False, dirRewrite=None, scale=False, constraint=None):
        """
        If CovObsErr and CovIntVar are both specified then CovTotal will be computed from
        CovObsErr+2*CovIntVar overwriting the value of CovTotal that may have been specified.
        Unspecified values will be set equal to None.

        :param obsNames: Optional List of observations wanted and in order expected.
        :param trace: optional with default False. If True then additional output will be generated.
        :param dirRewrite: optional ith default None. If set then rewrite directory names used in readCovariances.
        :param scale: if set true (default is false) then covariances are scaled by scaling factors derived from self.scales()
        :param constraint: is set to True  (default is None) then add constraint weighting into Covariances. If not set then
           if configuration asks for constraint (study.sigma set True) then will be set True.
            Total and ObsErr covariances for constraint will be set to 1/(2*mu) while IntVar covariance will be set to 1/(100*2*mu)
        :return: a dictionary containing CovTotal,CovIntVar, CovObsErr --  the covariance matrices. None if not present.
        Could be modified to cache covariance matrices to save re-reading but not bothering..
        """
        keys = ['CovTotal', 'CovIntVar', 'CovObsErr']
        useConstraint = constraint
        if constraint is None:
            useConstraint = self.constraint()  # work out if we have a constraint or not.

        if obsNames is None: obsNames = self.obsNames(add_constraint=useConstraint)
        cov = {}  # empty dict to return things in
        covInfo = self.getv('study', {}).get('covariance')
        # extract the covariance matrix and optionally diagonalise it.

        for k in keys:
            fname = covInfo.get(k, None)
            if fname is not None:  # specified in the configuration file
                cov[k] = self.readCovariances(fname, obsNames=obsNames, trace=trace, dirRewrite=dirRewrite)
                cov[k + "File"] = fname  # store the filename
                if cov[k] is not None:  # got some thing to further process
                    if covInfo.get(k + "Diagonalise", False):  # want to diagonalise the covariance
                        # minor pain is that np.diag returns a numpy array so we have to remake the DataFrame
                        cov[k] = pd.DataFrame(np.diag(np.diag(cov[k])), index=obsNames, columns=obsNames, dtype=float)
                        if trace: print("Diagonalising " + k)

        # make total covariance from CovIntVar and CovObsErr if both are defined.
        if cov.get('CovIntVar') is not None and cov.get('CovObsErr') is not None:  # if key not defined will "get" None
            k = 'CovTotal'
            cov[k] = cov['CovObsErr'] + 2.0 * cov['CovIntVar']
            cov[k + '_info'] = 'CovTotal generated from CovObsErr and CovIntVar'
            if trace: print("Computing CovTotal from CovObsErr and CovIntVar")
            if covInfo.get(k + "Diagonalise", False):
                if trace: print("Diagonalising " + k)
                cov[k] = pd.DataFrame(np.diag(np.diag(cov['CovTotal'])), index=obsNames, columns=obsNames)
                # diagonalise total covariance if requested.
        # check have total covarince and raiseError if not
        if cov.get('CovIntVar') is None:  # make it very small!
            cov['CovIntVar'] = pd.DataFrame(1.0e-12 * np.identity(len(obsNames)),
                                            index=obsNames, columns=obsNames,
                                            dtype=float)
        if cov.get('CovTotal') is None:
            cov['CovTotal'] = pd.DataFrame(np.identity(len(obsNames)),
                                           index=obsNames, columns=obsNames,
                                           dtype=float)

            print("Warning: No covariance defined. Assuming Identity")
            # raise ValueError("No covtotal found for totalFile=", covInfo.get(k, None))

        # apply constraint
        if useConstraint:  # want to have constraint wrapped in to  covariance matrices. Rather arbitrary for all but Total!
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
            scales = self.scales(obsNames=obsNames)
            cov_scale = pd.DataFrame(np.outer(scales, scales), index=scales.index, columns=scales.index)
            for k in keys:
                if k in cov and cov[k] is not None:
                    cov[k] = cov[k] * cov_scale
                    if trace: print("Scaling " + k)

        return cov

    # TODO add cache functionality
    def transMatrix(self, scale=False, verbose=False, dataFrame=True):
        """
        Return matrix that projects data onto eigenvectors of total covariance matrix
        :param scale: (Default False) Scale covariance.
        :param verbose: (default False) Be verbose.
        :param dataFrame: wrap result up as a dataframe
        :return:
        """

        # compute the matrix that diagonalises total covariance.
        cov = self.Covariances(trace=verbose, scale=scale)  # get covariances.
        errCov = cov['CovTotal']
        # compute eigenvector and eigenvalues of covariances so we can transform residual into diagonal space.
        evalue, evect = np.linalg.eigh(errCov)
        transMatrix = (np.diag(evalue ** (-0.5)).dot(evect.T))  # what we need to do to transform to
        if dataFrame:
            transMatrix = pd.DataFrame(transMatrix, index=np.arange(0, len(errCov.columns)), columns=errCov.columns)
        return transMatrix  # TODO wrap this up as a pandas array.

    def steps(self, paramNames=None):
        """
        Compute perturbation  for all parameters supplied. If value specified use that. If not use 10% of the range.
        Quasi-scientific in that 10% of range is science choice but needs knowledge of the structure of the JSON file
             so in this module.
        :param paramNames -- optional the parameter names for step sizes. If not defined uses self.paramNames() to work
                them out
        :return: the step sizes for the parameters as a pandas Series.
        """

        if paramNames is None: paramNames = self.paramNames()

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
        :param olist: (optional) List of Observations wanted from covariance file
        :param trace: (optional) if set True then some handy trace info will be printed out.
        :param dirRewrite (optional) if set to something then the first key in dirRewrite that matches in covFile
              will be replaced with the element.
        :return: cov -- a covariance matrix sub-sampled to the observations

        Returns a covariance matrix from file optionally sub-sampling to named observations.
        Note if obsName is not specified ordering will be as in the file.
        """

        use_covFile = os.path.expanduser(os.path.expandvars(covFile))
        if dirRewrite is not None:
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
            # verify covariance is sensible.. Should nto have any missing data
            if cov.isnull().sum().sum() > 0:  # got some missing
                print(f"Covariance from {use_covFile} contains missing data. Do fix")
                print(cov)
                raise Exception(f'cov {use_covFile} has missing data')


        except ValueError:  # now likely have index
            cov = pd.read_csv(use_covFile, index_col=0)
        if obsNames is not None:  # deal with olist
            cov = cov.reindex(index=obsNames, columns=obsNames)  # extract the values comparing to olist
            expect_shape = (len(obsNames), len(obsNames))
            if cov.shape != expect_shape:  # trigger error if missing
                print("Sub-sampled covariance shape = ", cov.shape, "And expected = ", expect_shape)
                raise ValueError

        return cov

    def postProcessScript(self):
        """

        :return: the full path for the postprocessing script
        """
        ppScript = self.Config['postProcess'].get("script",
                                                  "$OPTCLIMTOP/obs_in_nc/comp_obs.py")  # get PostProcessScript
        ppScript = os.path.expanduser(os.path.expandvars(ppScript))  # expand shell variables and home
        return ppScript

    def postProcessOutput(self, value=None):
        """
            :param (optional) value -- if not None (which is default) value will be used to set
        :return: relative  path for output from post processing script. Path is taken relative to model directory

        """
        ppProcess = self.getv('postProcess', {})
        if value is not None:
            ppProcess['outputPath'] = value
            self.setv('postProcess', ppProcess)

        ppOutput = self.getv('postProcess').get("outputPath", "observations.nc")

        return ppOutput

    def referenceConfig(self, studyDir=None):
        """
        :param studyDir -- where study is. Default will be to use current working directory
        :return: full path to the reference configuration of model being used
        """
        # if studyDir is None: studyDir = os.getcwd()
        # modelConfigDir = getDefault(self.Config['study'], 'referenceModelDirectory', os.path.join(studyDir, "start"))
        # # and now expand home directories and env variables
        # modelConfigDir = os.path.expanduser(os.path.expandvars(modelConfigDir))
        # return modelConfigDir
        # simplify -- if none just return it.
        modelConfigDir = self.getv('study').get('referenceModelDirectory')
        # and now expand home directories and env variables
        if modelConfigDir is not None:
            modelConfigDir = os.path.expanduser(os.path.expandvars(modelConfigDir))
        return modelConfigDir

    def optimise(self):
        """
        Extract and package all optimistion information into one directory
        :return: a dict
        """

        return self.Config['optimise']

    def fixedParams(self):
        """
        :return: a dict of all the fixed parameters. All names ending _comment or called comment will be excluded 
        """

        fix = copy.copy(self.getv('Parameters').get('fixedParams', None))
        if fix is None: return fix  # nothing found so return None
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

    def submitFunction(self, fnTable):

        """
        uses value of submitName to lookup function in fnTable and then returns function (or None)
        :return: the submission function
        """

        name = self.getv('machineName', None)
        return fnTable.get(name)

    def optimiseFunction(self, fnTable):

        """
        Use value of optimiseFunction to look up and return function in fnTable

        :param: fnTable -- dict  of functions
        :return: optimisation function
        """

        name = getDefault(self.Config, 'optimiseFunction', 'default')
        return fnTable.get(name)

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
            GNinfo = collections.OrderedDict()
            GNinfo['comment'] = 'Gauss-Newton Algorithm information'

        if variable is None:
            variable = GNinfo.get(name, None)
            if variable is None: return None  # no variable so return None
            variable = np.array(variable)  # extract variable from GNinfo and convert to numpy array
        else:  # got variable so put it in the GNinfo
            GNinfo[name] = variable.tolist()
            self.setv('GNinfo', GNinfo)  # store it.

        return variable

    def GNjacobian(self, jacobian=None, normalise=False, constraint=False):
        """
        Set/Return the Jacobian array as an xarray DataArray
        :param jacobian: (default None). If not None should be a numpy 3D array which will be stored in the config
        :return: xarray version of the Jacobian array
        """

        jacobian = self.GNgetset('jacobian', jacobian)
        if jacobian is None: return None
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
        if hessian is None: return None
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
        if cost is None: return None
        iterCount = np.arange(0, cost.shape[0])
        name = self.name()
        cost = pd.Series(cost, index=iterCount, name=name)
        cost.index.rename('Iteration', inplace=True)

        return cost

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
        """
        Compute the covariance for parameter error.
        Theory is that (to first order) \vect{\delta O}= \matrix{J}\vect{\delta p}
          where \delta O are perturbations in observations, \delta p are perturbations to parameters and J is the Jacobian.
          Then multiple by J^+ (the pseudo-inverse) to give \vect{\delta p} = \matrix{J}^+  \vect{\delta O}.
        Then the covariance is \matrix{J}^+ C {\matrix{J}^+}^T

         or alternatively with covariance...
         P=(J^TC^{-1}J)^{-1}J^TC^{-1}
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

        # TODO merge this with vn2 code which could be done using  if self.version >= 2: etc
        if hasattr(kwargs, 'scale'):
            raise Exception("scale no longer supported use normalise")  # scale replaced with normalised 14/7/19
        if len(kwargs) > 0:  # set the values
            self.Config['study']['optimumParams'] = kwargs
        if paramNames is None:  paramNames = self.paramNames()
        stdParams = self.standardParam(paramNames=paramNames)
        values = self.Config['study'].get('optimumParams', None)
        values = {k: values.get(k, stdParams[k]) for k in paramNames}
        if values is not None:
            values = pd.Series(values)  # wrap it as a pandas series.
        if normalise:
            range = self.paramRanges(paramNames=paramNames)  # get param range
            values = (values - range.loc['minParam', :]) / range.loc['rangeParam', :]

        return values

    def GNsimulatedObs(self, set=None, obsNames=None):
        """
        Set/get the simulated observations from the best value in each iteration of the Gauss Newton algorithm.
        :param obsNames: Names of observations wanted
        :param set: a pandas array or xarray (or anything that hass a to_dict method) that will set values of the observations.
          Should only be the best values on each iteration. Perhaps this routne should have all values... 
        :return: Observations as a pandas array.

        """

        if set is not None:  # set the value
            # check we've got GNinfo
            GNinfo = self.getv('GNinfo',
                               collections.OrderedDict())  # should modify to use GNgetset but that assumes numpy array.
            GNinfo['SimulatedObs'] = set.to_dict()
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
        info = self.getv(alg_info_name,
                         collections.OrderedDict())  # should modify to use GNgetset but that assumes nump
        if len(argv) > 0:  # got some values to set
            for key, item in argv.items():  # iterate over any arguments we have
                info[key] = item
            self.setv(alg_info_name, info)  # set it

        return info

    def diagnosticInfo(self, diagnostic=None):
        """

        :param diagnosticInfo: A pandas dataframe. If set add ths to the confguration
        :return: the values as a  dataframe
        """

        if diagnostic is not None:
            self.alg_info(diagnostic=diagnostic.to_json(orient='split'))

        return pd.read_json(self.alg_info().get('diagnostic'), orient='split')

    def jacobian(self, jacobian=None):
        """
        Set jacobian if set. Return jacobian as pandas dataframe
        :param jacobian: a pandas dataframe containing the jacobian information
        :return: jacobian
        """

        if jacobian is not None:
            self.alg_info(jacobian=jacobian.to_json(orient='split'))

        return pd.read_json(self.alg_info().get('jacobian'), orient='split')

    def paramErrCovar(self, normalise=True, useCov=False):
        r"""
        Compute the covariance for parameter error. TODO -- verify what DFOLS does and work out how
          to store in the final json file. Then regardless of algorithm we can get it out again...
        Theory is that (to first order) \vect{\delta O}= \matrix{J}\vect{\delta p}
          where \delta O are perturbations in observations, \delta p are perturbations to parameters and J is the Jacobian.
          Then multiple by J^+ (the pseudo-inverse) to give \vect{\delta p} = \matrix{J}^+  \vect{\delta O}.
        Then the covariance is \matrix{J}^+ C {\matrix{J}^+}^T

         or alternatively with covariance...
         P=(J^TC^{-1}J)^{-1}J^TC^{-1}
        :param normalise (default = True) -- compute the normalised covariance error (fraction of range)
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
            self.alg_info(best_obs=best_obs.to_json(orient='split'))

        return pd.read_json(self.alg_info().get('best_obs'), orient='split',
                            typ='series')

    def DFOLSinfo(self, diagnosticInfo=None):
        """

        :param diagnosticInfo: (optional -- defult None). If set add ths to the confguration
        :return: the values as a  dataframe. Temp code for backward compatability
        """

        import warnings
        warnings.warn("Use diagnosticInfo not DFOLSinfo")
        return self.diagnosticInfo(diagnostic=diagnosticInfo)

    def DFOLS_config(self):
        """
        Extract the DFOLS specific information
        :return:
        """

        return self.optimise().get('dfols', {})

    def DFOLS_userParams(self, userParams=None):
        """

        Extract the user parameters for DFOLS
        :param userParams (default None) -- default user params
        :return: user parameters
        """

        if userParams is None:
            result = {}
        else:
            result = userParams.copy()

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
                               collections.OrderedDict())  # should modify to use GNgetset but that assumes numpy array.
        if diagnosticInfo is not None:
            BOBYQAinfo['diagnostic'] = diagnosticInfo.to_json(orient='split')

        self.setv("BOBYQA", BOBYQAinfo)  # store the diagnostic info.

        return pd.read_json(BOBYQAinfo.get('diagnostic'), orient='split')

    def simObs(self, simObs=None, best=False):
        """

        :param simObs: optional -- default None and if passed should a pandas dataframe
        :param best: optional-- defaultFalse. If true return the obs for the best iteration
        :return:  dataframe of simulated observations in order in which they were ran (which may not be
           the smae as the algorithm sees them)
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

        cost = pd.Series(self.getv('costd'))
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
        if len(directories) is 0: return pd.Series()
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

        return self.getv('maxRuns')
        # no default -- up to calling application to decide.

    def runJacobian(self, jacobian=None, normalise=False, scale=True):
        """
        Store or return jacobian.
        :param jacobian: jacobian as dataset with mean (Jacobian) and variances (Jacobian_var)
        :return: jacobian as xaarry dataset.
        TODO : Consider removing dependance on xarray -- which means storing as dataframe.
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

    def baseRunID(self, value=None):
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
        Return (and optionaly set) the maxDigits for model run names

        If not defined return None -- up to the user of this to deal with it.

        :param Value -- value to be set -- default is not to set it

        """

        if value is not None:
            self.setv('maxDigits', value)

        return self.getv('maxDigits', default=None)  # nothing defined -- make it None.

    def copy(self, filename=None):
        """
        :param filename (optional default None): Name of filename to save to.
        :return: a copy of the configuration but set _fileName  to avoid accidental overwriting of file.
        """

        result = copy.deepcopy(self)  # copy the config
        result._filename = filename  # set filename
        return result


# TODO add a method to retrieve the max no of characters in a job with default value 5.

class OptClimConfigVn2(OptClimConfig):
    """
    Version 2 of OptClimConfig -- modify OptClimConfig methods and see OptClimConfig.__init__() for  
    """

    # NB __init__ method is just the superclasss OptClimConfig) __init__ method.

    def paramNames(self):
        """
        :return: a list of parameter names from the configuration files
        """
        keys = self.Config['Parameters']['initParams'].keys()  # return a copy of the list.
        # remove comment keys
        keys = [k for k in keys if 'comment' not in k]
        return keys

    def standardParam(self, values=None, paramNames=None, all=False, scale=False):
        """
        Extract standard parameter values for study
        :param paramNames: Optional names of parameters to use.
        :param values (default None). If values is not None set standardParams to values passed in and return them
        :param scale (default False). If True scale values by parameter range so that 0 is min, 1 is max
        :param all (default False). Return the whole parameters dict. Used when want to copy
        :return: pandas series (unless all is set)
        """
        if paramNames is None:   paramNames = self.paramNames()
        if values is not None:
            # check have Parameters and if not create it.
            if 'Parameters' not in self.Config:
                self.Config['Parameters'] = collections.OrderedDict()  # create it as an empty ordered dict.

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

    def paramRanges(self, paramNames=None):
        """
        :param paramNames -- a list of the parameters to extract ranges for.
        If not supplied the paramNames method will be used.
        :return: a pandas array with rows names minParam, maxParam, rangeParam
        """
        if paramNames is None: paramNames = self.paramNames()
        param = pd.DataFrame(self.Config['Parameters']['minmax'],
                             index=['minParam', 'maxParam'])
        # work out which names we have to avoid complaints from pandas.
        names = [p for p in paramNames if p in param.columns]
        param = param.loc[:, names]  # just keep the parameters we want and have.
        param = param.astype(float)
        param.loc['rangeParam', :] = param.loc['maxParam', :] - param.loc['minParam', :]  # compute range
        return param

    def beginParam(self, paramNames=None, scale=False):
        # TODO -- make this a set/get (if values passed in then set them).
        """
        get the begin parameter values for the study. These are specified in the JSON file in begin block
        Any values not specified use the standard values
        :param paramNames: Optional names of parameters to use.
        :param scale (default False). If True scale parameters by their range so 0 is minimum and 1 is maximum
        :return: pandas series of begin parameter values.
        """
        if paramNames is None:  paramNames = self.paramNames()
        begin = {}  # empty dict
        standard = self.standardParam(paramNames=paramNames)
        scaleRange = self.Config['Parameters'].get("initScale")  # want to scale ranges?
        range = self.paramRanges(paramNames=paramNames)  # get param range

        for p in paramNames:  # list below is probably rather slow and could be sped up!
            # begin[p] = self.Config['Parameters']['initParams'].get(p, standard.get(p))
            begin[p] = self.Config['Parameters']['initParams'].get(p)
            if begin[p] is None:
                begin[p] = standard[p]  # Will trigger an error if standard[p] does not exist
            else:
                if scaleRange:  # values are specified as 0-1
                    begin[p] = begin[p] * range.loc['rangeParam', p] + range.loc['minParam', p]
            if scale:  # want to return params  in range 0-1
                begin[p] = (begin[p] - range.loc['minParam', p]) / range.loc['rangeParam', p]

        begin = pd.Series(begin, dtype=float)[paramNames]  # order in the same way for everything.

        # verify values are within range
        if scale:
            L = begin.gt(1.0) | begin.lt(0.0)
        else:
            L = range.loc['maxParam', :].lt(begin) | begin.lt(range.loc['minParam', :])

        if np.any(L):
            print("L  \n", L)
            print("begin: \n", begin)
            print("range: \n", range)
            print("Parameters out of range", begin[L].index)
            raise ValueError("Parameters out of range: ")

        return begin.astype(float).rename(self.name())

    def optimumParams(self, paramNames=None, normalise=False, **kwargs):
        """
        Set/get the optimum parameters.
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
            return values  # no optimum values so return none

        for k in paramNames:
            values[k] = self.Config['Parameters']['optimumParams'].get(k, default[k])
            # set it to std value if we don't have it...Note will trigger error if default not got value

        ##values = {k: values.get(k,None) for k in paramNames} # extract the requested params setting values to None if we don't have them.
        values = pd.Series(values)[paramNames]  # wrap it as a pandas series and order it appropriately.

        if normalise:
            range = self.paramRanges(paramNames=paramNames)  # get param range
            values = (values - range.loc['minParam', :]) / range.loc['rangeParam', :]

        return values.rename(self.name())

    def steps(self, paramNames=None, normalise=None):
        """
        Compute perturbation  for all parameters supplied. If value specified use that. If not use 10% of the range.
        Quasi-scientific in that 10% of range is science choice but needs knowledge of the structure of the JSON file
             so in this module.
        :param paramNames -- optional the parameter names for step sizes. If not defined uses self.paramNames() to work
                them out
        :param normalise -- if True normalise by the range regardless of the value of scale in the configuration
        :return: the step sizes for the parameters as a pandas Series.
        """

        if paramNames is None: paramNames = self.paramNames()

        param = self.paramRanges(paramNames=paramNames)
        defaultStep = 0.1 * param.loc['rangeParam', :]  # 10% of range and default cases
        pert = {}
        scale = self.Config['Parameters']['steps'].get('scale_steps', False)
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

    def cacheFile(self):
        """
        Get the pathname relative to the study directory of the cache file.
        This file holds information (directories in current design) on model simulation directories.
        :return:  filename relative to studyDir
        TOD: Don't think this used so remove it.
        """

        fileStore = self.Config.get('studyCacheFile', 'cache_file.json')
        if fileStore is None: fileStore = 'cache_file.json'
        return fileStore

    # do some plotting
    def plot(self, figName='monitor', monitorFile=None):
        """
        plot cost, normalised parameter & obs values for runs.
        :param figName: name of figure to make -- default is monitor
        :param monitorFile: name of file to save figure to if not None. Default is None
        :return: figure, (costAxis, paramAxis, obsAxis)

        Note needs matplotlib
        """

        cost = self.cost()
        if len(cost) == 0:
            return  # nothing to plot
        try:
            # recent versions of matplotlib allow clear argment.
            fig, ax = plt.subplots(3, 1, num=figName, figsize=[8.3, 11.7], sharex='col', clear=True)
        except TypeError:
            fig, ax = plt.subplots(3, 1, num=figName, figsize=[8.3, 11.7], sharex='col')

        (costAx, paramAx, obsAx) = ax  # name the axis .
        cmap = plt.cm.get_cmap('RdYlGn')
        cmap.set_under('skyblue')
        cmap.set_over('black')
        try:  # now to plot
            nx = len(cost)
            costAx.plot(np.arange(0, nx), cost.values)
            costAx.set_xlim(-0.5, nx)
            minv = cost.min()
            minp = cost.values.argmin()  # use location in array (as that is what we plot)
            costAx.set_title("Cost", fontsize='small')
            costAx.plot(minp, minv, marker='o', ms=12, alpha=0.5)
            costAx.axhline(minv, linestyle='dotted')
            costAx.set_yscale('log')
            yticks = [1, 2, 5, 10, 20, 50]
            costAx.set_yticks(yticks)
            costAx.set_yticklabels([str(y) for y in yticks])
            # plot params

            parm = self.parameters(normalise=True)
            parm = parm.reindex(index=cost.index)  # reorder
            X = np.arange(-0.5, parm.shape[1])
            Y = np.arange(-0.5, parm.shape[0])  # want first iteration at 0.0
            cm = paramAx.pcolormesh(Y, X, parm.T.values, cmap=cmap, vmin=0.0, vmax=1.)  # make a colormesh
            paramAx.set_yticks(np.arange(0, len(parm.columns)))
            paramAx.set_yticklabels(parm.columns)
            paramAx.set_title("Normalised Parameter")
            paramAx.axvline(minp, linestyle='dashed', linewidth=2, color='gray')

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
            obsAx.set_yticks(np.arange(0, len(normObs.columns)))
            obsAx.set_yticklabels(normObs.columns, fontsize='x-small')
            obsAx.set_xlabel("Iteration")
            xticks = np.arange(0, nx // 5 + 1) * 5
            obsAx.set_xticks(xticks)
            obsAx.set_xticklabels(xticks)
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


class OptClimConfigVn3(OptClimConfigVn2):
    """
    Vn3 of OptClimConfig. Currently does nothing but is a place to grab things for next upgrade. 
    1) have runTime() and runCode() methods only work with runInfo block -- vn1/2 methods work with names in top level dict. 
    """

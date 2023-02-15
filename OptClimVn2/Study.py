"""
Class to provide ways of looking at a study directory.
Goal is to merge this into ModelSubmit (eventaually) 
"""

import copy
import functools
import logging
import os
import pathlib  # needs python 3.6+
import shutil

import numpy as np
import pandas as pd

import optClimLib
import sys

class Study(object):
    """
    Class to support a study. Idea is that eventually modelSubmit 
    sub-classes this. This class proides support for reading info 
    from a study -- both in progress or completed. But does not 
    handle submitting models or realising that a new one needs to be generated.
    TODO: generate test cases based on Submit.py.
    """

    def __init__(self,config,modelFn,rootDir=None,verbose=False):
        """
        Create study instance.
        :param config -- the configuration
        :param modelFn -- funcion to create model instantance.
        :param rootDir -- where model data lives
        :param verbose -- if True be verbose
        """

        self.verbose=verbose
        self.config = config
        self.modelFn=modelFn
        self._models = dict()  # initialise ordered collection where models store
        if rootDir is None:  # no rootDir defined. Use current working dir
            self.rootDir = pathlib.Path.cwd()# default path
        else:
            self.rootDir = pathlib.Path(rootDir)  # make sure it is a Path.


        self.readDir(self.rootDir)

        return

    def obsNames(self):
        """

        :return: the obsnames from the config
        """
        return self.config.obsNames()
        
    def paramNames(self):
        """

        :return: param names from config
        """
        return self.config.paramNames()

    def targets(self, scale=True):
        """
        get target values
        :param scale (default True) apply scaling
        :return: scaled target values from config
        """

        return self.config.targets(obsNames=self.obsNames(), scale=scale)

    def scales(self):
        """
        Get scales from config
        :return: scaling values as pandas series. See StudyConfig scales.
        """
        return self.config.scales(obsNames=self.obsNames())

    def transMatrix(self, scale=True, dataFrame=True):
        """
        Return  transform matrix by calling StudyConfig.transMatrix
        :param (optional) scale (default True) -- scale transformation matrix;
        :param (optional) dataframe (default True) -- if True  wrap as a dataframe.
        :return: transform matrix.
        """
        return self.config.transMatrix(scale=scale, dataFrame=dataFrame)

    def readModelDir(self, dir):
        """
        Read model configuration from directory, generate key and store it.
        :param dir: path to directory where model configuration is to be found
        :return: model if succeeded, None if failed.
        """

        try:
            if self.verbose:
                print("Reading from %s" % (dir))
            # configuration overrides what was in the model when created... 
            config = self.config
            obsWant = config.obsNames()
            model = self.modelFn(dir, obsNames=obsWant,
                                 ppOutputFile=config.postProcessOutput())
            # read in the model. 
            # what to do if model crashes but want to continue anyhow???
            # generate params dict suitable for generating key.
            key = self.genModelKey(model)
            if self.verbose:
                print("Read Dir %s Key is:\n %s " % (dir, key))
                if self._models.get(key) is not None:  # warn if verbose on and got duplicate dir.
                    print("Got duplicate dir =  %s " % (dir) + "\n key = " + repr(key))
                    
            self._models[key] = model  # store the model in modelRuns indexed by its key.            
            return model

        except (IOError, EOFError):  # likely failed to read something so error!
            # IOError when no config file found; EOFError when config file corrupted.
            if self.verbose:
                print("Failed to read from %s" % (dir))
            return None

    def readDir(self, dir):
        """
        reads all model configurations from dir and stores them.
        :param dir: directory (filePath) where models to found

        :return:
        """

        for d in sorted(dir.iterdir()):  # loop over sub-directories etc
            if d.is_dir():  # only try and read from directories.
                model = self.readModelDir(d)  # read the configuration. readModelDir traps errors.

    def genKey(self, paramDict, fpFmt='%.4g', ):

        """
        Generate key from keys and values in paramDict
        This should be unique (to some rounding on float parameters)
        :param paramDict -- a dictionary (or something that behaves likes a dict) of variable parameters.
        :param fpFmt -- format to convert float to string. (Default is %.4g)
        :return: a tuple as an index. tuple is key_name, value in sorted order of key_name.
        #TODO -- make the paramNames/values be everything. Passed in as an iterable.
        # then construct keys from name/values as now. Note this means that would be possible to
        # make this a method of the model which would have lots of advantages!
        """

        key = []
        paramKeys = sorted(paramDict.keys())
        # removed possible rewrite on refDir

        # removed code to deal with ignoreKeys

        # deal with variable parameters -- produced by optimisation so have names and values.
        for k in paramKeys:  # iterate over keys in sorted order.
            key.append(k)
            v = paramDict[k]
            # remove code that allowed keyfn here too. 
            if isinstance(v, float):
                key.append(fpFmt % v)  # float point number so use formatter.
            else:  # just append the value.
                key.append(repr(v))  # use the object repr method.
        key = tuple(key)  # convert to tuple
        return key


    def model(self, params, verbose=None):
        """
        Return model or None that matches key.
        :param self:
        :param params: parameters as a dict or iterable
        :param verbose (default None). If set overrule value in self.verbose
        :return: model that has requested parameters. None if nothing found.
        """
        doVerbose = self.verbose
        if verbose is not None:
            doVerbose = verbose
        paramAll = {}
        paramAll.update(self.fixedParams())
        paramAll.update(params)

        key = self.genKey(paramAll)
        if doVerbose:
            print("Key is\n", repr(key), '\n', '-' * 60)
        model = self._models.get(key, None)
        return model


    def allModels(self):
        """
        A generator function for key and value for all models
        :return: a items generator. (use as k,v in ModelSubmit.allModels():)
        """

        return self._models.items()


    def params(self, includeFixed=False):
        """
        Extract the parameters used in the simulations. Will include ensembleMember -- as a "fake" parameter
        :includeFixed (optional; default True) -- include all fixed params if True
        :return: pandas dataframe of parameters

        """
        p = []
        indx = []
        names = []
        if includeFixed:
            names.extend(self._fixParams)  # add in the fixParams
        params = self.paramNames()[:]
        # params.sort()
        names.extend(params)
        # add ensembleMember for times when we have an ensemble.
        names.append('ensembleMember')

        for key, model in self.allModels():
            param = model.getParams(series=True, params=names)
            if includeFixed:  # add refDir
                param.refDir = model.refDirPath()
            p.append(param)
            indx.append(model.name())
        paramsDF = pd.DataFrame(p, index=indx)
        return paramsDF

    def obs(self, scale=True,remove_missing=False):
        """
        Extract the Obs used in the *individual* simulations
        No ensemble averaging is done and you need the param data to work out which ensemble is which.
        :param scale If True data will be scaled.
        :param remove_missing. If True remove any obs with missing data.
        :return: pandas dataframe of observations


        """
        o = []
        for key, model in self.allModels():
            obs = model.getObs(series=True)
            obs = obs.reindex(self.obsNames())
            o.append(obs)
        if len(o) == 0:  # empty list
            return None

        obsDF = pd.DataFrame(o)
        if scale:
            obsDF *= self.config.scales()
        if remove_missing:
            obsDF=obsDF.dropna() # drop row with any missing data.
        return obsDF
        
    def cost(self,scale=True):
        """
        compute cost from data. 
        :param: scale -- scale data.
        :return pandas series of costs.
        TODO: Add test cases. 
        """
        
        config = self.config
        obs = self.obs(scale=scale,remove_missing=True)  # get params & obs
        if obs is None:  # no data
            return None
        tMat = self.transMatrix(scale=scale)  # which puts us into space where totalError is Identity matrix.
        
        nObs = len(obs.columns)
        resid = (obs - self.targets(scale=scale)) @ tMat.T
        cost = np.sqrt((resid ** 2).sum(1).astype(float) / nObs)
        return cost



    def genModelKey(self, model):
        """
        Generate key from model
        :param model: model for which key gets generated
        :return: key
        """

        params = {}
        params.update(self.fixedParams())
        params.update(model.getParams(verbose=self.verbose))
        if 'refDir' not in self.fixedParams():
            params.update(refDir=model.refDirPath())

        key = self.genKey(params)  # Generate key.
        return key

        
    def fixedParams(self):
        """

        :return: fixed parameters
        """

        return copy.copy(self.config.fixedParams())  ## return a *copy* of this

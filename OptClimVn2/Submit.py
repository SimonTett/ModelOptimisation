"""
Support for handling model submission. Idea is various functions needed are passed into the __init__ method.
See config.py for some actual functions that do the work.
TODO rename this as does not just handle submission..
and code is ratehr messy and would benefit from some re-engineering. Which means thinking quite hard about what this module actually doess.
It is a place that handles all the models soprobably should also handle the various algorithms and thier data needs.
"""
import collections  # TODO remove as dict in python 3.7+ is ordered.
import copy
import functools
import logging
import os
import pathlib  # needs python 3.6+
import shutil

import numpy as np
import pandas as pd

import optClimLib

# import optClimLib

__version__ = '0.4.0'
# subversion properties
subversionProperties = \
    {
        'Date': '$Date$',
        'Revision': '$Revision$',
        'Author': '$Author$',
        'HeadURL': '$HeadURL$'
    }


class runModelError(Exception):
    """
    Error when need to run a model!
    """
    pass


class ModelSubmit(object):
    """
    provides methods to support working out which models need to be submitted
    """

    # so replacing modelDirs functionality. (rootDir & config would then be compulsory)
    # TODO add a new argument -- restart which if set would remove all stuff in rootDir
    # wonder if modelFn & submitFn can be extracted from config???

    def __init__(self, config, modelFn, submitFn, optimiseFn, fakeFn=None,
                 rootDir=None, ignoreKeys=None, verbose=False, readOnly=False, renameRefDir=None,
                 keyFn=None, noObs='fail'):
        """
        Create ModelSubmit instance
        :param config: configuration information
        :param modelFn: function that creates model instance
        :param submitFn: function that submits the models -- will likely be system (& model) dependent
        :param optimiseFn: function that does the work -- it gets mixed up with the instance
          TODO: Document the expected function signature.

        :param fakeFn (optional = None): if provided then nothing will actually be submitted.
            Instead the fakeFunction will be ran which will generate fake values for the observations file.
        :param rootDir (optional = None): root dir where new directories are to be created.
          If None will be current dir/config.name()
        :param ignoreKeys: Model parameters to ignore for uniqueness -- default values are:
            runCode, runTime and RUNID
        :param verbose: be verbose (and  all methods will use this value)
        :param readOnly (default False): If True then no new models will be created, nor submitted.
        :param renameRefDir (default None): If not None then assumed to be a
            directory like object which contains keys to be rewritten. Useful for testing.
        :param keyFn (default None): If not none a directory whose keys are the keys used for generating keys and values
          are functions that get applied to the string. Useful for testing
        :param noObs (default 'fail') WHat to do when observations are missing from a model directory.
            Stored in self. see readModelDir for description.

        :return: instance of ModelSubmit

        submitFn(model_list, config, rootDir, verbose=False, resubmit=None, *args, **kwargs)
         where model_list is a list of models to be submitted, config is a StudyConfig, rootDir is the
           root where and files can be created, verbose - -be verbose, resubmit -- what to resubmit,
            other args and kwargs are to be interpreted as needed!

        fakeFn(params, config) # given input parameters as pandas Series returns obs as pandas series.


        """

        self.modelFn = modelFn
        self.fakeFn = fakeFn
        self.config = config
        self.submitFn = submitFn
        self.noObs = noObs  # what to do when we have no observations

        # if readOnly setfakeFn and submitFn to None -- so if they are called nothing happens
        if readOnly:
            # alas need modelFn set so can read in info from configurations.
            self.fakeFn = None
            self.submitFn = None

        self.renameRefDir = renameRefDir
        self.keyFn = keyFn

        self._optimiseFn = optimiseFn
        # self._optimiseFn_args = optimiseFn_args
        self.readOnly = readOnly
        if ignoreKeys is None:
            self.ignoreKeys = ['runCode', 'runTime', 'RUNID']  # values to ignore when generating keys.#
            # TODO consider generalising -- might be too specific and maybe best to come from model gen fn in some way.
        else:
            self.ignoreKeys = ignoreKeys

        self.refDir = config.referenceConfig()
        if rootDir is None:
            self.rootDir = pathlib.Path.cwd() / config.name()  # default path
        else:
            self.rootDir = pathlib.Path(rootDir)  # make sure it is a Path.

        self._fixParams = self.config.fixedParams()
        # add on refDir if defined
        if self.refDir is not None:
            self._fixParams.update(refDir=str(self.refDir))

        self._modelsToSubmit = collections.OrderedDict()  # where we store models to be submitted.
        self._modelsToRerun = collections.OrderedDict()  # where we store models to be reran.

        self.verbose = verbose
        # store models.
        # self.dirCount = len(modelDirs)  # how many directories we got -- not the same as the number of models.
        self._models = collections.OrderedDict()  # initialise ordered collection where models stored.

        if not self.rootDir.is_dir():  # dir does not exist so create it.
            self.rootDir.mkdir(parents=True)
        self.readDir(self.rootDir)  # read all directories.

        return  # and all done

    # methods that extract information from config. caching to avoid recomputation.
    # suspect there is a more pythonic way to generate them..

    @functools.lru_cache()
    def obsNames(self):
        """

        :return: the obsnames from the config
        """
        return self.config.obsNames()

    @functools.lru_cache()
    def fixedParams(self):
        """

        :return: fixed parameters
        """

        return copy.copy(self._fixParams)  ## return a *copy* of this

    @functools.lru_cache()
    def paramNames(self):
        """

        :return: param names from config
        """
        return self.config.paramNames()

    @functools.lru_cache()
    def targets(self, scale=True):
        """
        get target values
        :param scale (default True) apply scaling
        :return: scaled target values from config
        """

        return self.config.targets(obsNames=self.obsNames(), scale=scale)

    @functools.lru_cache()
    def scales(self):
        """
        Get scales from config
        :return: scaling values
        """
        return self.config.scales(obsNames=self.obsNames())

    @functools.lru_cache()
    def transMatrix(self, scale=True):
        """
        Return  transform matrix by calling config method.
        :param (optional) scale (default True) -- scale transformation matrix
        :return: transform matrix.
        """
        return self.config.transMatrix(scale=scale)

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

    def rerunModels(self, model=None):
        """
        :param model -- add model to list of models to rerun
        :return: list of  models to rerun (no post processing needed -- 'coz something got wrong and pp job waiting)
        """

        if not hasattr(self, '_modelsRerun'):
            self._modelsRerun = []  # empty list

        if model is not None:
            self._modelsRerun.append(model)

        return self._modelsRerun

    def readModelDir(self, dir):
        """
        Read model configuration from directory, generate key and store it.
          Behaviour when no observations found depends on self.noObs
             'fail': fail!
             'continue: attempt to continue the simulation
             'perturb': perturb the simulation and restart it.
             'clean': remove the directory.
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
            if self._models.get(key) is not None:
                print("Got duplicate dir =  %s " % (dir) + "\n key = " + repr(key))
            # check model has observations and if not do something!
            obs = model.getObs()
            if (obs is None) or any([v is None for v in obs.values()]):  # Obs is None or any  obs are None.
                # deal with noObs cases.
                if self.noObs == 'fail':
                    raise Exception(f"Some observations for {dir} are none")
                elif self.noObs == 'continue':  # mark for continuation.
                    model.continueSimulation(minimal=True)
                    self.rerunModels(model=model)
                    # want to submit it too but that depends on system we are on so gets handled later.
                elif self.noObs == 'perturb':  # perturb model
                    # tricky -- don't want to change the parameters.
                    # think this is a modelSimulation method. Though not very comfortable with config and state being
                    # inconsistent..
                    # NEED a remove continue option...
                    param = model.perturbParams()
                    model.perturb(params=param)
                    self.rerunModels(model=model)
                elif self.noObs == 'clean':
                    # remove the directory
                    shutil.rmtree(model.dirPath, onerror=optClimLib.errorRemoveReadonly)
                    print("Cleaned %s" % (model.dirPath))
                    return None  # no model to return.
                else:  # unknown noObs case.
                    raise Exception(f"Unknown noObs {self.noObs}")
            else:
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
        :return: a tuple as an index.
        #TODO -- make the paramNames/values be everything. Passed in as an iterable.
        # then construct keys from name/values as now. Note this means that would be possible to
        # make this a method of the model which would have lots of advantages!
        """

        key = []
        paramKeys = sorted(paramDict.keys())
        # deal with possible rewrite on refDir
        if self.renameRefDir is not None:
            refDir = paramDict['refDir']
            refDir = self.renameRefDir.get(refDir, refDir)
            # if refDir in keys for renameDir will get new value otherwise no change
            print("Rewritten: %s -> %s " % (paramDict['refDir'], refDir))
            paramDict['refDir'] = refDir

        # remove ignoreKeys
        for ignore in self.ignoreKeys:
            try:
                paramKeys.remove(ignore)
            except ValueError:
                pass

        # deal with variable parameters -- produced by optimisation so have names and values.
        for k in paramKeys:  # iterate over keys in sorted order.
            key.append(k)
            v = paramDict[k]
            try:
                v = self.keyFn[k](v)  # apply function listed in keyFn
            except (TypeError, KeyError):
                pass  # nothing to do

            if isinstance(v, float):
                key.append(fpFmt % (v))  # float point number so use formatter.
            else:  # just append the value.
                key.append(repr(v))  # use the object repr method.
        key = tuple(key)  # convert to tuple
        return key

    def modelSubmit(self, keyValue=None):
        """
        Add to list of runs to be submitted if value provided.
        Otherwise return  list of models to make.
        Return list
        :param self: object 
        :param keyValue: two element tuple ; key and values to be stored
        :return: 
        """

        if keyValue is not None:
            (key, value) = keyValue
            self._modelsToSubmit[key] = value  # and store it.

        return self._modelsToSubmit.values()  # just return the values.

    def model(self, params, update=False, verbose=None):
        """
        Return model or None that matches key.
        :param self:
        :param params: parameters as a dict or iterable
        :param update (default False). If True and params not found then as
              well as returning None will update _modelsSubmit dict
        :param verbose (default None). If set overrule value in self.verbose


        :return: model that has parameters. None if nothing found.
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
        if (model is None) and update:
            if doVerbose:
                print("Failed to find", paramAll)
            self.modelSubmit((key, paramAll))

        return model

    def rerunModel(self, model):
        """
        Add existing (and maybe modified) model to directory of models to be reran.
        :param model: model
        :return: nothing
        """
        key = self.genModelKey(model)
        self._modelsToRerun[key] = model

    def allModels(self):
        """
        A generator function for key and value for all models
        :return: a items generator. (use as k,v in ModelSubmit.allModels():)
        """

        return self._models.items()

    def nextName(self):
        """
        generate the next name -- this is a generator function.
         :return: iterator that gives names
        """
        # initialisation

        base = self.config.baseRunID()  # get the baserun
        # TODO -- get maxLen from config file/model?
        maxDigits = self.config.maxDigits()  # get the maximum length of string for model.
        if maxDigits is None:
            maxDigits = 3  # default is 3.
        # for HadCM3 names > 5 characters cause trouble. 
        # other models are better written but for moment will 
        # restrict to 10 characters.
        lenBase = len(base)
        useBase = base
        if lenBase > 10:  # truncate name length to 10 if too long
            useBase = useBase[:10]

        # TODO allow letters a-z as well as numbers 0-9. That gives 36^maxDigits values ~ 46,000 values for 2 letter base
        # and use lower/upper case to, That gives 62^maxDigits cases or 238,328 cases for 2 letter base or 3844 cases for 3 letter base
        # or ~1000 values for 3 letter base.
        maxValue = 10 ** maxDigits - 1  # can't have value larger than this.
        # Special for maxDigits= 0 where we don't put in a number...
        if maxDigits == 0:
            maxValue = 1
        num = 0  # start count at zero. Code below will keep going until dir not found.
        fmtStr = '%%%d.%dd' % (maxDigits, maxDigits)
        while True:  # keep going
            num += 1
            if num > maxValue: raise ValueError("num to large")  # TODO mark as end of iterator ??
            if maxDigits == 0:
                digitStr = ''
            else:
                digitStr = fmtStr % num
            name = useBase + digitStr
            # check if dir exists.
            # TODO Consider moving this outside the nextName logic.
            createDir = self.rootDir / name
            if createDir.exists():
                continue  # go round the loop again.
            yield (createDir, name)  # "return" the directory to be created and the name.

    def submit(self, resubmit=None, dryRun=False, verbose=None, postProcess=True,
               *args, **kwargs):
        """
        Submit models that need to be submitted using self.submitFn -- see init documentation
            (repeating documentation)

            submitFn should expect a list of models to submit and take at least the following named arguments:
                verbose -- provide detailed output. TODO: just move to logging ??
                resubmit -- If not None the command to resubmit once all cases are done.
                rootDir -- the rootDirectory where model configurations are created.
                   Useful for any files submission needs to create.
            The following information will be extracted and passed to submitFunction from the configuration used
               to generate the ModelSubmit object.
                runTime
                runCode
                name
        :param resubmit -- if provided command to resubmit the calling script.
        :param dryRun (optional; default False). If True don't submit any runs though do create them.
        :param verbose (optional: default None). If set override value in self.
        :param args: unnamed arguments passed into submitFn
        :param kwargs: keyword arguments passed into submitFn
        :return: status of submission (depends on submit function) and number of models passed to submit. True -- all OK; False -- something went wrong!
        """

        if self.readOnly:  # read only so return not much.
            return True, 0

        doVerbose = self.verbose
        if verbose is not None:
            doVerbose = verbose
        maxRuns = self.config.maxRuns()
        runTime = self.config.runTime()
        runCode = self.config.runCode()
        ## all setup so go and submit models.

        submitModels = self.rerunModels()
        if len(submitModels) > 0:  # got minimal models to submit. Submit them **all** regardeless of max models.
            # TODO think about what happens if fail because have too many jobs submitted..
            if dryRun:  # doing dry run so no actual submission
                return True, len(submitModels)
            # otherwise submit using submitFunction
            if self.fakeFn is None:  # actually submit the model
                status = self.submitFn(submitModels, self.config, self.rootDir,
                                       verbose=self.verbose,
                                       resubmit=None, postProcess=False, runTime=runTime, runCode=runCode,
                                       *args, **kwargs)  # submit the models
                print("(re)Submited %i runs " % (len(submitModels)))

            else:  # been asked to fake the models so run the fake function.
                for model in submitModels:
                    fakeObs = self.fakeFn(model, self.config)  # fake the obs
                status = True
            return status, len(submitModels)  # minimal models so don't do any more!

        # Do possible continuation runs.
        submitModels = []
        for model in self._modelsToRerun.values():
            if (maxRuns is not None) and (len(submitModels) >= maxRuns):
                print("Created %i with maxRuns= %i" % (len(submitModels), maxRuns))
                break  # stop generating models as made too many!
            # now go and create new model.
            submitModels.append(model)
        # and then possible new runs,
        for param, (createDir, name) in zip(self.modelSubmit(), self.nextName()):
            # deal with maxRuns logic
            if (maxRuns is not None) and (len(submitModels) >= maxRuns):
                print("Created %i with maxRuns= %i" % (len(submitModels), maxRuns))
                break  # stop generating models as made too many!
            # now go and create new model.
            obsNames = self.config.obsNames()
            param['RUNID'] = name
            refDir = param.pop('refDir')  # remove refDir from param dict.
            model = self.modelFn(createDir, obsNames=obsNames, create=True,
                                 name=name, runTime=runTime, runCode=runCode,
                                 ppExePath=self.config.postProcessScript(), refDirPath=refDir,
                                 ppOutputFile=self.config.postProcessOutput(),
                                 parameters=param, verbose=doVerbose)
            submitModels.append(model)

            if doVerbose:
                print("createDir is %s path is %s" % (createDir, os.getcwd()))
                print("Params are:\n", repr(param), "\n", '-' * 80)

        # end of iterating over models  to generate.

        if dryRun:  # doing dry run so no actual submission
            return True, len(submitModels)
        # otherwise submit using submitFunction
        if self.fakeFn is None:  # actually submit the model
            status = self.submitFn(submitModels, self.config, self.rootDir,
                                   verbose=self.verbose,
                                   resubmit=resubmit, runTime=runTime, runCode=runCode,
                                   *args, **kwargs)  # submit the models
        else:  # been asked to fake the models so run the fake function.
            for model in submitModels:
                fakeObs = self.fakeFn(model, self.config)  # fake the obs
            status = True
        return status, len(submitModels)

    def params(self, includeFixed=False):
        """
        Extract the parameters used in the simulations. Will include ensembleMember -- as a "fake" parameter
        :includeFixed (optional; default True) -- include all fixed params if True
        :return: pandas dataframe of parameters
        """
        p = []
        indx = []
        names = self.paramNames()[:]
        names.sort()  # sort names.
        # add ensembleMember for times when we have an ensemble.
        names.append('ensembleMember')
        if includeFixed:
            names.extend(self._fixParams)  # add in the fixParams
        for key, model in self.allModels():
            param = model.getParams(series=True, params=names)
            if includeFixed:  # add refDir
                param.refDir = model.refDirPath()
            p.append(param)
            indx.append(model.name())
        paramsDF = pd.DataFrame(p, index=indx)
        return paramsDF

    def obs(self, scale=True):
        """
        Extract the Obs used in the simulations
        :param scale (optional; default = True)
        :return: pandas dataframe of observations
        """
        o = []
        for key, model in self.allModels():
            obs = model.getObs(series=True)
            print(obs.name)
            if scale:  # scale obs.
                scales = self.config.scales()
                name = obs.name
                obs = scales * obs
                obs = obs.rename(name)
                # need to reset name
            o.append(obs)

        obsDF = pd.DataFrame(o)
        return obsDF

    def runConfig(self, filename=None):
        """
        **copy** config and add parameters, obs and cost to it. Config is returned and maybe saved if filename provided.
        :param self:
        :param filename (default None). If set then new config will be saved to this file.

.
        :return:
        """
        newConfig = self.config.copy(filename=filename)

        paramObs = self.paramObs()  # get params & obs
        if paramObs is None:
            if filename is not None:
                newConfig.save()
            return newConfig  # no data so return the config.

        params = paramObs.reindex(columns=self.paramNames())
        obs = paramObs.reindex(columns=self.obsNames())  # extract the obs.
        scales = self.scales()  # get the scales
        obs = obs * scales  # normalise scales.
        tMat = self.transMatrix()  # which puts us into space where totalError is Identity matrix.
        nObs = len(self.obsNames())
        resid = tMat.dot((obs - self.targets(scale=True)).T)

        cost = np.sqrt(np.sum(resid ** 2, axis=0).astype(float) / nObs)
        # print("Cost is ",cost)
        cost = pd.Series(cost, index=obs.index)

        # update newConfig
        cost = newConfig.cost(cost)
        newConfig.parameters(params)
        newConfig.simObs(obs)
        # work out best directory by finding minimum cost.
        bestEval = None
        if len(cost) > 0:
            bestEval = cost.idxmin()

        newConfig.setv('bestEval', bestEval)  # best evaluation

        if filename is not None:
            newConfig.save()

        return newConfig

    def optimiseFunction_args(self, clear=False, **extraArgs):
        """
        Setup extra named arguments for the optimise function
        :param clear (Default False): If True set internal arguments to {}
        :param extraArgs: named arguments which get slurped into a dict and then used
           to update arguments already provided.
        :return:Nada
        """

        if not hasattr(self, '_extraArgs') or clear:
            self._extraArgs = {}

        self._extraArgs.update(**extraArgs)

        return self._extraArgs

    # TODO -- change name of this from optimise as jacobian uses it and that is not optimisation!
    def optimiseFunction(self, fn=None):
        """
        :param fn (optional default None): If not None then store this function for later
           and then continue
        :return: a function suitable for use in an optimisation  algorithm.
        by generating a partial function which has MODELRUN=self in the argument list
        and any extra args (as named arguments) that are present.
        """
        if fn is not None:
            self._optimiseFn = fn  # store the fn.

        fn = functools.partial(self._optimiseFn, MODELRUN=self)

        return fn

    def paramObs(self, params=None, obs=None, clear=False):
        """

        :param params: (optional -- default None). If params are set then observations (if provided) will be set for that value
           if not set then all observations will be returned. params should be a pandas series.
        :param obs: (optional -- default None). If set then params must be set and these values will be set.
          If not set then obs will be returned. obs should also be a pandas series. When returning a dataframe the series name will be the index.
        :param clear: (optional -- default False). If True remove all previously stored obs.
        :return: the observations & parameters for the specified parameters as a pandas series or all observations/parameters as a pandas dataframe.
        """

        if clear and hasattr(self, '_paramObs'):
            del (self._paramObs)

        # initialise _paramObs.
        if not hasattr(self, '_paramObs'):
            self._paramObs = collections.OrderedDict()  # intitialise _paramObs with empty orderedDict.

        if obs is not None:
            # got some obs so store them!
            key = self.genKey(params.to_dict())
            # will trigger error if params is None (or technically does not have a to_dict() method)
            s = obs.append(params).rename(obs.name)
            self._paramObs[key] = s

        # now to return values.
        if params is not None:
            key = self.genKey(params.to_dict())
            return self._paramObs[key]

        # got to here so no params set -- so return everything
        if len(self._paramObs) == 0:  # nothing there
            return None

        return pd.DataFrame(list(self._paramObs.values()))  # name coming from obs series.

    def optFunction(self, params,ensembleMember=None):
        """
        Runs user supplied function, does various post processing and returns result.
        :param params -- parameters (as a np array) to run with.
        :return: result of user function.
        """

        fn = self.optimiseFunction()
        # fn = self._optimiseFn
        result = fn(params,MODELRUN=self,ensembleMember=ensembleMember)
        obsNames = self.obsNames()
        # result = result.loc[self.obsNames()]  # order it consistently
        # Optionally catch Nan and raise runModelError if any. Really for algorithms which don't trap!
        if self._extraArgs.get('failNan', True) and np.any(np.isnan(result)):
            logging.debug('Found NaN and raising runModelError')
            raise runModelError("ERROR")  # ("params are %"%(repr(params)))

        # optionally transform data to eigenvectors of covariance matrix
        if self._extraArgs.get('transform', True):
            logging.debug('Transforming data')
            tMat = self.transMatrix()
            result = result * self.scales().values  # scale it
            result = result - self.targets().values  # make it a residual
            # do dot product with non-missing values
            for indx in range(0, result.shape[0]):
                if not np.any(pd.isnull(result[indx, :])):
                    result[indx, :] = tMat.dot(result[indx, :])  # transform into orthogonal basis of covar matrix

        if self._extraArgs.get('sumSquare', False):
            # compute sum of squares...
            result = np.sum(result ** 2, 1)  # sum along dim 1

        return np.squeeze(result)

    def runOptimized(self):
        """
        :arg self
        Run optimised case using reference model which may be p[otential different from configuration used to optimize.
        Cares about number of ensemble members (default is 1 if not set) and the optimum parameters which should be set.

        One issue is what to name it. As things stand it is up to the user. They should also set maxDigits sensibly.
        For the UM the suk of length of baseRun & maxDigits should  be 5 or less.



        """

        #
        # basic idea is that users takes final json file and edits it to suit their needs

        configData = self.config

        nEns = configData.ensembleSize()  # how many ensemble members to run
        simObs = []
        fn = self.optimiseFunction()  # get the function!
        # add ensembleMember to the list of parameters IFF nEns > 1
        # this is tricky as StudyConfig gets that from the initial parameters (as it thinks about optimisation)
        # a hack might be to add ensembleMember =0 to the initial parameters...
        # note here we have start.
        if nEns > 1:
            pass
            # self.config.Config['Parameters']['initParams'].update(ensembleMember=0)
            # SADLY this  hack fails because ensembleMember has no known range...
            # I don't want to make a bunch of extensive changes to code base -- which would inlcude identifying some parameters as
            # unscalable and so not scaling them...
            # TODO remove this evil hack of modifying initParams.
            # At the level of a model simulation then ensembleMember is a parameter -- just not one that you optimize.
            # At teh algorithm level ensembles hsouldl be hidden -- they get run and averaged for the optimisation...
        start = configData.optimumParams()
        for ens in range(0, nEns):  # generate ensemble members getting data returned as a dataframe.
            series = fn(start.values, MODELRUN=self, ensembleMember=ens, df=True).iloc[0, :]  # make it a series
            simObs.append(series)
        simObs = pd.DataFrame(simObs)  # pack all the series together to get the simulated observations as a dataframe.
        # see if have any missing obs. If we do raise runModelError which should cause them to be run if the calling code
        # follows the prototype approach.
        someMissing = np.any(simObs.isnull())
        if someMissing:
            raise runModelError("Run Optimized Cases")

        # now to set up information having done all cases.
        finalConfig = self.runConfig()  # get final runInfo
        # need to hack a bit by adding ensembleMember to the parameters
        params = finalConfig.parameters()
        params.loc[:, 'ensembleMember'] = range(0, nEns)
        finalConfig.parameters(params)
        # and add optimum parameters (which as a hack will be min cost)
        idxbest = finalConfig.cost().idxmin()
        bestParams = params.loc[idxbest, :]
        finalConfig.optimumParams(bestParams.to_dict())
        # and remove begin parameters fromm finalConfig as makes no sense in this context
        finalConfig.Config['Parameters'].pop('initParams')  # removbe init parameters from config.
        # TODO -- do not use direct acess to Config. Instead use a method which means making initParams a method with a delete option.
        # FIXME -- problem I am having is that ensembleMember is not a formal parameter... but it should be added
        # if nEns > 1.
        # need to deal with ensemblemember here -- relly as a hack for "best" case.
        # add the best case (min cost)

        # fixes to make -- work out what information goes into finalConfig and so is returned...
        # start parameters; best parameters -- they should all be the same so will pick first ensemnble

        return finalConfig

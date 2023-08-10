"""
Provide class and methods for ModelSimulation"
"""
import collections
import copy
import json
import logging
import os
import pickle
import shutil
import stat
import tempfile
import datetime
import f90nml  # available from http://f90nml.readthedocs.io/en/latest/
import netCDF4
import numpy as np
import pandas as pd  # pandas
import warnings  # so we can turn of warnings..
import pathlib  # TODO slowly move to use this rather than os.path
import logging # TODO replace prints with logging.info/logging.debug

import xarray

import optClimLib  # provide std routines



# import stdRoutines # provide standard routines.


class modelEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Convert obj into something that can be converted to JSON
        :param obj -- object to be converted
        :return "primitive" objects that can be converted + type of object
        """
        stype = str(type(obj))  # string rep of data-type
        # try various approaches to generate json file
        fnsToTry = ['to_dict', 'tolist']
        for fn in fnsToTry:
            f = getattr(obj, fn, None)
            if f is not None:
                return dict(stype=stype, data=f())  # run the method we've found
        # failed to run fns that made stuff can turn into JSON test
        if 'dtype' in dir(obj):
            return dict(data=str(obj), dtype=stype)
        else:
            return json.JSONEncoder(self, obj)
            # Let the base class default method try to convert and raise the TypeError


_namedTupClass = collections.namedtuple('TupClassxx',
                                        ('var', 'namelist', 'file'))  # named tuple for namelist information


# TODO -- make this a class function which is available not private.
# or make it a dataclass with methods to handle namelists.
class ModelSimulation(object):
    """
    Class to define Model Simulations. This class is top class that
       provides basic functionality. In particular the configuration as updated
       is written to disk as a pickle file. Each set does this.
       Define new model simulations that do more by sub-classing this class.
       To get the observed values do model.getObs()
       To get parameters do model.getParams()
       To see what you have do model.get() and examine the orderedDict you have got.
       model.readObs() reads in data from data on disk. This is provided so that when a model
       has been run (and postprocessed) data can be read in and manipulated by functions outside this module.
       To support usecase of model failure readObs method will fall back to reading a json StudyConfig file and extracting
       data from it. See readObs for more details.

       TODO -- write more extensive documentation  esp on namelist read/writes.
       TODO -- move namelist stuff out of ModelSimulation into separate module??
       TODO -- do not use pickle but save as a json file. Though that will make reading in tricky.
    """

    _simConfigPath = 'simulationConfig.cfg'  # default configuration filename for the Model Simulation.

    def __init__(self, dirPath, obsNames=None, runTime=None, runCode=None,
                 create=False, refDirPath=None, name=None, ppExePath=None, ppOutputFile=None, parameters=None,
                 # options for creating new study
                 update=False,  # options for updating existing study
                 verbose=False):
        """
        Create an instance of ModelSimulation class. Default behaviour is to read from dirPath and prohibit updates.
        :param dirPath -- path to directory where model simulation exists or is to be created.
                        Shell variables and ~ will be expanded
        :param create (optional with default False). If True create new directory and populate it.
            If directory exists it will be deleted. 
            Afterwards the ModelSimulation will be readOnly.
            These options should be specified when creating a new study otherwise they are optional and ignored
            :param refDirPath -- reference directory. Copy all files from here into dirPath
            :param name -- name of the model simulation. If not provided will be taken from dirPath
            :param ppExePath --  path to post processing executable
            :param ppOutputFile -- Filename for  output of post processing executable. Default is observations.nc
            :param parameters -- dict of parameter names and values.
            :param obsNames -- list of observations to be readin. (see readObs())
            :param runTime -- the time limit (depends on your job system) for the model to run for.
                Default is None which means values from reference configuration used.
            :param runCode -- the code (depends on your job system) for the model to run for.
                Default which means values from your reference configuration are used.
        :param update -- allow updates to the simulation information.
        :param verbose -- provide  verbose output. (See individual methods). Default is False.
        :returns initialised object.
        """
        # stage 1 common initialisation
        self._convNameList = dict() # information on the variable to Namelist mapping
        self._metaFn = dict() # functions used for meta parameters.
        self.config = dict()
        self.dirPath = pathlib.Path(os.path.expandvars(os.path.expanduser(dirPath))) # expand dirPath and store it
        # convert to an absolute path -- needed when jobs get submitted as don't know where we might be running from
        self.dirPath = self.dirPath.absolute()
        self._readOnly = True  # default is read only.
        self._configFilePath = self.dirPath/self._simConfigPath  # full path to configuration file

        ppModifyMark = '# =insert post-process script here= # '
        self.postProcessMark = ppModifyMark
        # names of submission files. Modify for your own model.  See submit for details.
        self.SubmitFiles={'start':'submit.sh','continue':'submit_cont.sh'}


        postProcessFile = ppOutputFile

        # verify not got create and update both set
        if create and update:
            raise Exception("Don't specify create and update")
        # create new model Simulation
        # TODO move this outside  object creation. Which implies some changes to rest of system.

        if create:
            self.createModelSimulation(parameters=parameters, ppExePath=ppExePath, obsNames=obsNames, name=name,
                                       ppOutputFile=ppOutputFile, refDirPath=refDirPath, verbose=verbose)
        else:
            self.readModelSimulation(update=update, obsNames=obsNames, ppOutputFile=ppOutputFile, verbose=verbose)
        # done reading in information

    def __eq__(self, other):
        """
        Equality -- two models are equal if their types, parameters, refDirs and obs are the same
        :param other: an other model to compare with
        :return: True if equal, False if not
        """

        equal = True
        equal = equal and (type(self) == type(other))
        try:
            equal = equal and (self.refDirPath() == other.refDirPath())  # same refDir
            equal = equal and (self.getParams() == other.getParams())  # same params
            equal = equal and (self.readObs() == other.readObs())  # same obs.
        except AttributeError:  # when other doesn't have methods
            equal = False

        return equal

    def __ne__(self, other):
        """
        Not equal. Negation of equal
        :param other: other thing to compare with
        :return: True if not equal, False if equal
        """

        return not (self == other)

    def readConfig(self, verbose=False):
        """
        Read configuration file
        :param verbose (Default = False) if set then print out more information
        :return: configuration
        """
        # TODO replace this by reading parameters, pp stuff and obsnames.
        # let these be overwritten from the init values if set.
        # pickle makes it difficult to update while running which might be an advantage
        with open(self._configFilePath, 'rb') as fp:
            config = pickle.load(fp)

        return config

    def readModelSimulation(self, obsNames=None, update=False, ppOutputFile=None, verbose=False):
        """
        Read in information on modelSimulation
        :param obsNames -- obs names being used -- if set will override original config
        :param ppOutputFile -- name of post processing output file. If set overrides original config
        :param update (default = False) -- this model sim can be updated if true
        :param  verbose(default = False) be verbose
        :return:
        """

        # need to read config file
        config = self.readConfig(verbose=verbose)
        # for update I think failure to read is fine. We'd just return null...
        # if ObsNames set use it

        #if obsNames is not None:
        #    #raise FutureWarning("Passing obsNames to readModelSimulation is deprecated")
        #    logging.info("setting obsNames")
        #    obs = {k: None for k in obsNames}
        #    config['observations'] = obs
            # TODO -- not sure why I do this. the config was pickled so right now is just an ordered collection.
            #  And probably should not be using obsNames anyhow -- when we read the obs we read what ever is there.
        if ppOutputFile is not None:
            postProcess = config.get('postProcess', {})
            postProcess['outputPath'] = ppOutputFile
            config['postProcess'] = postProcess
        self.set(config, write=False, verbose=verbose)
        self.readObs()  # and read the observations
        self._readOnly = not update

    def set(self, keys_values, write=True, verbose=False):  # suspect can do this with object methods...
        """
        sets value in configuration information and writes configuration out.
        :param keys_values:  dict/ordered dict of keys and values
        :param write (default = True). If true writes configuration information out to disk
        :param verbose (default = False). If true produces verbose output.

        :return:
        """
        # raise exception if readOnly.
        if self._readOnly and write: raise Exception("Config is read only")
        # set value in config
        for k, v in keys_values.items():
            self.config[k] = v
        # write whole of config out to dir.
        if write:
            if os.path.isfile(self._configFilePath):
                os.chmod(self._configFilePath, stat.S_IWUSR)  # make it write only so we can delete it
                os.remove(self._configFilePath)  # remove file

            with open(self._configFilePath, 'wb') as fp:  # create it
                # TODO replace this dumping to a json file of the current values -- means if object inherits it will need to interface to this.
                # Easiest way to do is dump the config variable which shoudl contain all information. Documentation to
                # object shoud be clear that all entries should be json serialisible. Or overwrite read & write methods to do what you want to do!
                # so perhaps output the configuration to a json file.. with some additional meta data.
                # But the read will need to try and read the pickle file if the json file is not present...
                pickle.dump(self.config, fp)
            mode = os.stat(self._configFilePath).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            if verbose:
                print("Written current config to %s with mode %o " % (self._configFilePath, mode))
            os.chmod(self._configFilePath, mode)  # set its mode

    def set2(self, write=True, verbose=False, **kwargs):  # suspect can do this with object methods...
        """
        sets value in configuration information and writes configuration out.
        :param write (default = True). If true writes configuration information out to disk
        :param verbose (default = False). If true produces verbose output.
        :keywords -- set those values.
        :return:
        """
        # raise exception if readOnly.
        if self._readOnly and write: raise Exception("Config is read only")
        # set value in config
        for k, v in kwargs.items():
            self.config[k] = v
        # write whole of config out to dir.
        if write:
            if os.path.isfile(self._configFilePath):
                os.chmod(self._configFilePath, stat.S_IWUSR)  # make it write only so we can delete it
                os.remove(self._configFilePath)  # remove file

            with open(self._configFilePath, 'wb') as fp:  # create it
                pickle.dump(self.config, fp)
            mode = os.stat(self._configFilePath).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            if verbose:
                print("Written current config to %s with mode %o " % (self._configFilePath, mode))
            os.chmod(self._configFilePath, mode)  # set its mode

    def get(self, keys=None, default=None):
        """
        get values from configuration.
        :param keys: keys to extract from configuration. If not defined returns the entire configuration.
          If a single string returns that value.
        :return: dict indexed by keys
        """

        if keys is None:
            return self.config  # just return the entire config

        if type(keys) is str:  # , unicode):
            # keys is a single string or unicode
            return self.config.get(keys, default)

        # otherwise assume list and work through the elements of it.
        result = []
        for k in keys: result.append(self.config.get(k, default))
        # convert 1 element list to scalar
        if len(result) == 1: result = result[0]
        return result

    def createModelSimulation(self, parameters, ppExePath=None, obsNames=None, name=None,
                              ppOutputFile=None, refDirPath=None, verbose=False):
        """
        Create (in filesystem) a model simulation. After creation the simulation will be read only.
        :param parameters -- dict of parameter names and values OR pandas series.
        :param ppExePath --  path to post processing executable -- Default None
        :param obsNames -- list of observations being used. -- Default None
        :param  name ((optional)) -- name of the model simulation. If not provided will be taken from dirPath
        :param  ppOutputFile (optional)  -- name of output file where output from postprcessing is (default comes from config)
        :param refDirPath (optional) -- reference directory. Copy all files from here into dirPath
        :param  verbose (optional) -- if true be verbose. Default is False
        """
        # general setup
        self._readOnly = False  # can write if wanted.

        if refDirPath is not None:
            refDirPath = os.path.expandvars(os.path.expanduser(refDirPath))
        #  fill out configuration information.
        config = dict()
        if name is None:
            config['name'] = os.path.basename(self.dirPath)
        else:
            config['name'] = name


        config['ppExePath'] = ppExePath
        config['ppOutputFile'] = ppOutputFile
        if refDirPath is not None: config['refDirPath'] = refDirPath


        config['parameters'] = parameters

        config['newSubmit'] = True  # default is that run starts normally.
        config['history'] = dict()  # where we store history information. Stores info on resubmit, continue information.

        if verbose:   print("Config is ", config)

        if os.path.exists(self.dirPath):  # delete the directory (if it currently exists)
            shutil.rmtree(self.dirPath, onerror=optClimLib.errorRemoveReadonly)

        if refDirPath is not None:  # copy all files and directories from refDirPath to dirPath and make then user writable
            mode = os.umask(0)  # get current vale of umask
            os.umask(mode)  # and reset it
            mode = mode | stat.S_IWUSR  # mode we want is umask + write
            if os.path.exists(self.dirPath):  # got directory so need to remove it
                shutil.rmtree(self.dirPath, onerror=optClimLib.errorRemoveReadonly)  # delete where we are making data.
            shutil.copytree(refDirPath, self.dirPath)  # copy everything
            # now change the permissions so user can write
            if verbose: print("Copied files from %s to %s " % (refDirPath, self.dirPath))
            for root, dirs, files in os.walk(self.dirPath, topdown=True):  # iterate
                for name in dirs:  # iterate over  the directories
                    dname = os.path.join(root, name)
                    mode = os.stat(dname).st_mode | stat.S_IWUSR
                    os.chmod(dname, mode)
                for name in files:  # iterate over files in directory
                    fname = os.path.join(root, name)
                    mode = os.stat(fname).st_mode | stat.S_IWUSR  # mode has user write permission.
                    #                print "Setting mode on %s  to %d"%(fname,mode)
                    os.chmod(fname, mode)
        else:
            # Make target directory.
            os.makedirs(self.dirPath)  # create directory
        self.set(config)  # set (and write) configuration
        # TODO add setParams call here..
        # and no longer able to write to it.
        self._readOnly = True

    # end of createModelSimulation

    def refDirPath(self):
        """
        Provide path for refdir
        :return: name of the reference directory or None
        """

        return self.get('refDirPath', None)

    def name(self):
        """
        Return the name of the Model Configuration
        """

        return self.config['name']

    def ppOutputFile(self, file=None):
        """
        Return the name of the file to be produced by the post Processing 
        :param file -- if defined set value for ppOutputFile to file.
        :return: 
        """
        if file is not None:
            self.set('ppOutputFile', file)  #
        return self.get(['ppOutputFile'])

    def ppExePath(self):
        """
        Return the name of the post processing executable file. 
        :return: 
        """

        return self.get(['ppExePath'])

    def readObs(self,series=False,obsNames=None,flush=False):
        """
        Read the post processed data.
         This default implementation reads netcdf, json or csv data and
         stores it in the configuration
        :param obsNames: If not None then return the defined names. Any missing are set to None/np.nan
        :param series , If true return a pandas series containing the values.
        :param flush. If True flush the cached data and read data
        :return: a dict of observations (or pandas series) wanted. Values not found when requested will be set to None.
        """
        obs = self.get('observations',{}).copy()
        if (len(obs) == 0 ) or flush: #read in the data either because we don't have it or want to flush cache
            obsFile = self.dirPath/self.ppOutputFile()
            if not obsFile.exists(): # file does not exist. Return empty obs..
                logging.info(f"File {obsFile} does not exist. Returning None")
                print(f"File {obsFile} does not exist. Returning None")
                obs = None
                if obsNames is not None: # deal with case when obsNames provided.
                    # replication of logic below. TODO only have this logic once.
                    obs={k:None for k in obsNames}
                    if series:
                        obs=pd.Series(obs).rename(self.name())
                return obs


            fileType = obsFile.suffix  # type of file wanted
            #read in data. Details depend on type of file.
            if fileType == '.nc': # netcdf
                ds= xarray.load_dataset(obsFile)
                obs = {var:float(ds[var] ) for var in ds.data_vars if ds[var].size == 1}

            elif fileType == '.json': # json file
                with open(obsFile, 'r') as fp:
                    obs = json.load(fp)
                logging.info("json file got "+" ".join(obs.keys()))

            elif fileType == '.csv':  # data is a csv file.
                obsdf = pd.read_csv(obsFile, header=None, index_col=False)
                obs = obsdf.to_dict()
                logging.info("csv file got " + " ".join(obs.keys()))

            else:  # don't know what to do. So raise an error
                raise NotImplementedError(f"Do not recognize {fileType}")

            logging.info(f"Read {fileType} data from {obsFile}")
            self.set({'observations': obs.copy()}, write=False) # cache the data now

        if obsNames is not None:  # Only want those observations with missing ones set to None
            obs = {var:obs.get(var,None) for var in obsNames}


        if series:
            obs = pd.Series(obs).rename(self.name())  # convert to series.

        return obs  # return the obs.

    def writeObs(self, obs, verbose=False):
        """
        Write the observations to file. Type of file determines how write done
        :param obs -- observations (dict with keys obsNames) (or pandas series)
        :param verbose: (default False) be verbose if True
        :return: nada!
        """

        file = self.ppOutputFile()  # filename
        file = os.path.join(self.dirPath, file)  # full path to output

        fileType = os.path.splitext(file)[1]  # type of file
        logging.info(f"Writing data to {file}")
        if fileType == '.nc':  # netcdf file
            rootgrp = netCDF4.Dataset(file, "w", format="NETCDF4")
            try:
                for key, obsV in obs.items():  # iterate over index in series.
                    if verbose:
                        print(f"key:{key} value {obsV}")
                    v = rootgrp.createVariable(key, 'f8')  # create the NetCDF variable
                    if verbose:
                        print("Var is ", v, obsV, v.size)
                    v[:] = obsV  # write to it -- will fail if not a scalar I imagine!

            finally:  # any failure close the netcdf file
                rootgrp.close()
        elif fileType == '.json':  # json file
            with open(file, 'w') as fp:  # just dump the obs to the file.
                if type(obs) is pd.Series:  # it is a series -- convert to dict to wrte
                    json.dump(obs.to_dict(), fp, indent=4)
                else:  # already a dict (I hope)
                    json.dump(obs, fp, indent=4)
        elif fileType == '.csv':  # data is a csv file.
            # assume it is a pandas series.
            obs.to_csv(file, header=True)
        else:  # don't know what to do. So raise an error
            raise NotImplementedError("Do not recognize %s" % fileType)

        # have read the obs. We should set the obs
        self.set({'observations': obs.copy()}, write=False)



    def setParams(self, params, addParam=False, write=True, verbose=False):
        """
        Set the parameter values and write them to the configuration file
        :param params -- dictionary (or ordered dict) of the parameter values
        :param addParam (default False) -- if True add to existing parameters
        :param write (default True) -- if True update configuration file.
        :param verbose (default False) -- if True provide more verbose output
        :return:
        """
        if params is None:
            raise Exception
        if addParam:
            if verbose: print("Updating existing parameters")
            oldParams = self.getParams()
            for k, v in params.items():
                oldParams[k] = v  # update
            self.set({'parameters': oldParams}, verbose=verbose, write=write)
        else:
            self.set({'parameters': params}, verbose=verbose, write=write)  # set the parameters

    def getParams(self, verbose=False, params=None, series=False):
        """
        Extract the parameter values
        :param verbose (default = False) if True print out information
        :param params (default is None) If not None then only return those parameters in the same order.
        :param series (default is False). If True return params as a pandas series.
        :return: the parameter values
        """

        p = self.get('parameters').copy()

        if params is not None:  # enforce particular order and select reqd values.
            t = dict()
            for pp in params: t[pp] = p.get(pp)  # should return None if pp not found
            p = t
        # convert to series if requested
        if series:
            p = pd.Series(p)

        return p

    def registerMetaFn(self, varName, function, verbose=False):
        """
        Register a function to process meta parameters
        The function should take three  argument -- a value for forward call and a dict with names and values for inverse call
        , a named argument inverse with default False, and a named argument namelist. value should be optional and default value is something sensible.
        It should return a dict of values keyed by namelist tuple.
        TODO: Add an example of how this works -- see test_modelSimulation.py
        :param varName: name of meta parameter
        :param function: Function to register
        :param verbose: be verbose. Default is False
        :return: nothing
        """
        if verbose:
            try:
                print("Registering function %s for %s" % (function.func_name, varName))
            except AttributeError:
                pass
        # verify fn works as expected.
        res = function()  # run it with default value.
        keys = res.keys()

        # verify namelist keyword works
        nlKeys = function(namelist=True)
        if not ((set(nlKeys) == set(keys)) and (len(nlKeys) == len(keys))):
            print("nlKeys are", set(nlKeys))
            print("keys are ", set(keys))
            raise Exception("namelist keys and expected keys differ")
        # verify nlKeys and keys are the same.

        a = function(res, inverse=True)  # make sure inverse works.
        res2 = function(a)  # having got the default arg back we run again,
        for k, v in res2.items():
            if type(v) is np.ndarray:
                L = np.all(v != res[k])
            else:
                L = (v != res[k])
            if L:
                raise Exception("Fn not invertible")

        self._metaFn[varName] = function

    def genVarToNameList(self, param, nameListVar, nameListName, nameListFile, verbose=False):
        """
        Generate a conversion list for use in converting model parameter names to namelist values.
         Recommended approach if have a easy case with one framework variable matching directly to
         one namelist variable.
        :param param: the name of the parameter (as a string)
        :param nameListVar  :  variable name in the namelist
        :param nameListName: name of namelist
        :param nameListFile: Paths *relative* to the configuration dir for file containing namelist
        :param verbose (optional -- default = False) be verbose if True
        :return: a named tuple containing variable, namelist and file.
        """
        nt = _namedTupClass(var=nameListVar, namelist=nameListName, file=nameListFile)
        self._convNameList[param] = [nt]

        if verbose: print("var %s ->" % (param), self._convNameList[param])

    def applyMetaFns(self, verbose=False, fail=False, **params):
        """
        Apply transformation functions to meta parameters
          :param verbose: optional default is False. If True be verbose.
          :param fail: optional default is False. If true fail if parameter not found
          keywords are parameters and value.
        :return: returns dict of keys (which should be named tuple defining namelist var, namelist and file)  and their values,
            and list of meta parameters that were used.



        """
        result = dict()
        metaParamsUsed = []
        for k, v in params.items():
            if k in self._metaFn or fail:
                # got fn (or not fail) so run it
                fnResult = self._metaFn[k](v)
                metaParamsUsed.append(k)
                for fk, fv in fnResult.items():
                    result[fk] = fv

        # sort metaParamsUsed so order is deterministic...
        return result, sorted(metaParamsUsed)

    # TODO (if ever need it) add a NameList method that returns the namelist info possibly nicely printed.

    def writeNameList(self, verbose=False, fail=False, **params):
        # TODO make parameters a simple dict rather than kwargs
        """
        Modify existing namelist files using information generated via genConversion
        Existing files will be copied to .bak
        :param verbose (optional -- default is False). If True provide more information on what is going on.
        :param fail (optional default is False). If True fail if a parameter not found.
        :keyword arguments are parameters and values.
        :return:  ordered dict of parameters and values used.
        """
        if self._readOnly:
            raise IOError("Model is read only")

        params_used = dict()  #
        files = dict()  # list of files to be modified.
        for param, value in params.items():  # extract data from conversion indexed by file --
            # could this code be moved into genVarToNameList as really a different view of the same data.
            # NO as we would need to do this only once we've finished generating namelist translate tables.
            # potential optimisation might be to cache this and trigger error in writeNameList if called after genNameList
            # search functions first
            if param in self._metaFn:  # got a meta function.
                if verbose: print(f"Running function {self._metaFn[param].__name__}")
                metaFnValues = self._metaFn[param](value)  # call the meta param function which returns a dict
                params_used[param] = metaFnValues  # and update return var
                for conv, v in metaFnValues.items():  # iterate over result of fn.
                    if conv.file not in files:
                        files[conv.file] = []  # if not come across the file set it to empty list
                    files[conv.file].append((v, conv))  # append the  value  & conversion info.
            elif param in self._convNameList:  # got it in convNameList ?
                for conv in self._convNameList[param]:
                    if conv.file not in files:
                        files[conv.file] = []  # if not come across the file set it to empty list
                    files[conv.file].append((value, conv))  # append the value  & conversion
                    params_used[param] = value  # and update return var
            elif fail:
                raise KeyError("Failed to find %s in metaFn or convNameList " % param)
            else:
                pass

        # now have conversion tuples ordered by file so let's process the files
        for file in files.keys():  # iterate over files
            # need to create backup? Only do if no back up exists. This allows generateNameList to be run multiple times
            # doing updates. First time it runs we assume we have a directory ready to be modified.
            filePath = os.path.join(self.dirPath, file)  # full path to namelist file
            # check file exists if not raise exception
            if not os.path.isfile(filePath):
                # raise IOError("file %s does not exist"%(filePath))
                continue  # skip this file.
            backup_file = filePath + "_nl.bak"  # and full path to backup fie.
            if not os.path.isfile(backup_file):
                shutil.copyfile(filePath, backup_file)
            # now create the namelist file.
            with open(filePath) as nmlFile:
                nl = f90nml.read(nmlFile)
            nl.end_comma = True
            nl.uppercase = True
            nl.logical_repr = ('.FALSE.', '.TRUE.')  # how to reprsetn false and true
            # Need a temp file
            with tempfile.NamedTemporaryFile(dir=self.dirPath, delete=False, mode='w') as tmpNL:
                # Now construct the patch for the  namelist file for all conversion tuples.

                for (value, conv) in files[file]:
                    if conv.namelist not in nl:
                        nl[conv.namelist] = dict()  # don't have ordered dict so make it
                    if type(value) is np.ndarray:  # convert numpy array to list for writing.
                        value = value.tolist()
                    elif isinstance(value, str):  # may not be needed at python 3
                        value = str(value)  # f90nml can't cope with unicode so convert it to string.
                    nl[conv.namelist][conv.var] = copy.copy(
                        value)  # copy the variable to be stored rather than the name.
                    if verbose:
                        print("Setting %s,%s to %s in %s" % (conv.namelist, conv.var, value, filePath))
                try:
                    nl.write(tmpNL.name, force=True)
                except StopIteration:
                    print("Problem in f90nml for %s writing to %s" % (filePath, tmpNL.name), nl)
                    raise  # raise exception.

            if verbose: print("Patched %s to %s" % (filePath, tmpNL.name))
            os.replace(tmpNL.name, filePath)  # and copy the modified file back in place.

        return params_used

    def readNameList(self, params, fail=False, verbose=False, full=False):
        """
        Read parameter value from registered namelist
        :param fail: If True fail if param not found
        :param verbose (Optional -- default False). Provide verbose information.
        :param params -- a list of parameters.
        :param full -- return namelist rather than interpreted parameters. Allows testing.
        :example self.readNameList(['RHCRIT', 'VF1'])
        :return:An OrderedDict with the values indexed by the param names
        """
        # TODO make work when params is a single string.
        result = dict()
        for param in params:
            # have it as meta function?
            if param in self._metaFn:  # have a meta funcion -- takes priority.
                result[param] = self.readMetaNameList(param, verbose=verbose, full=full)
            elif param in self._convNameList:  # in the conversion index
                nlValue = self.readNameListVar(self._convNameList[param], verbose=verbose)
                if len(nlValue) != 1: raise ValueError("Should only have one key")
                for k in nlValue.keys():  # in this case shoudl only have one key and we don't want it as a list
                    result[param] = nlValue[k]
                # just want the value and as single param SHOULD return
            elif fail:  # not found it and want to fail
                raise KeyError("Param %s not found" % param)
            else:
                pass
        return result  # return the result.

    def readNameListVar(self, nameListVars, verbose=False):
        """
        Read single parameter specified via named tuple defining namelist variable

        :param verbose: default False. If True be verbose
        :param NameListVars: list of namelist variables to be retrieved
        :return: an ordered dict indexed by namelist info (if found) with values retrieved.
        """

        result = dict()
        namelists = {}
        # iterate over all parameters reading in namelists.
        for var in nameListVars:
            if var.file not in namelists:  # not already read this namelist.
                namelists[var.file] = f90nml.read(nml_path=os.path.join(self.dirPath, var.file))
        # now having read all needed namelists extract the values we want
        for var in nameListVars:
            nlvalue = namelists[var.file][var.namelist][var.var]
            result[var] = nlvalue
        return result

    def readMetaNameList(self, param, verbose=False, full=False):
        """
        Retrieve value of meta parameter  by reading namelists and running inverse function.
        :param param:  name of meta-parameter
        :param verbose: be verbose
        :param full: do not run inverse function -- return all values.
        :return:  value of meta-parameter
        """
        # work out what fn is and run it with default value to work out what namelist values we need to retrieve.
        fn = self._metaFn[param]  # should generate an error if not found
        nlInfo = fn(namelist=True)  # get the namelist info by asking the function for it!
        # retrieve appropriate values from namelist
        if verbose: print("Retrieving ", )
        var = self.readNameListVar(nlInfo, verbose=verbose)  # read from namelist.
        if verbose: print("Retrieved", var)
        if full:
            result = var  # skip inverse processing
        else:
            result = fn(var, inverse=True)  # run function in reverse

        return result

    def setReadOnly(self, readOnly=None):
        """
        Modify read only flag
        :param readOnly: value to set (default is None -- which doesn't modify status.
             True means any attempts to write to dir will trigger error
        :return: current status
        """

        if readOnly is not None:
            self._readOnly = readOnly

        return self._readOnly

    def allParamNames(self):
        """
        
        :return: list of all parameter names -- suitable  for passing through.
        """

        names = list(self._convNameList.keys())
        names.extend(self._metaFn.keys())
        return names


    def runStatus(self,value=None):
        """
        provide current value of runStatus -- useful for dealing with model failures.
        This is persistent as status is stored in the configuration.,
        :param value: set runStatus to this value if not None.
           Allowed values are come from self.SubmitFiles
           This is used by the submit method to decide what script to submit.
        :return:  current value of runStatus with default value of 'start'
        """

        # deal with value
        if value is not None:
            if value not in self.SubmitFiles.keys(): # list of allowed runStatus values
                raise ValueError(f"unrecognized value {value}")
            self.setReadOnly(False)
            self.set2(runStatus=value)
            self.setReadOnly()

        return self.get('runStatus',default='start')





    def continueSimulation(self, verbose=False):
        """
        Default continue simulation. Will modify the configuration so that next time it is submitted
          it will be a continuation case rather than a new simulation. History information will be updated.
        This is useful when the simulation has crashed and can safely be continued.
        :return: list of all timestamps when runs set for continuation
        """

        self.setReadOnly(readOnly=False)  # let us write to the configuration.
        history = self.get('history', default={})
        cont_list = history.get('cont', [])
        cont_list.append(datetime.datetime.now())
        history['cont'] = cont_list
        self.set2(history=history)  # store if minimal or not
        self.runStatus('continue') # will continue the simulation.
        self.setReadOnly(readOnly=True)  # no more writing to the configuration.
        return cont_list

    def restartSimulation(self, verbose=False):
        """
        Restart simulation. This is useful if the model  crashed because it ran out of
          time and needs restarting but cannot be continued safely.
          This will start the run again from scratch.
        :param verbose -- If True (default is False) print out some information.
        :return: list of all restarts -- gives times when model has been restarted
        """

        status = self.setReadOnly()  # let us write to the configuration.
        self.setReadOnly(readOnly=False)
        history = self.get('history', default={})
        restart = history.get('Restart', [])
        restart.append(datetime.datetime.now())
        history.update(restart)
        self.runStatus('start')  # we want to have the run start again.
        self.setReadOnly(readOnly=status)  # reset readOnly status...
        return restart  # return the timestamps from the history file.

    def perturbParams(self, verbose=False, pScale=1.000001):
        """
        Work out new values for parameters for perturbation. Algorithm just keeps adding parameters perturbing them
        :param verbose: be verbose if True
        :param pScale: How much to scale parameters
        :return: modified parameter dict.
        """

        perturbList = self.perturb()  # work out what has already been perturbed
        nPerturb = len(perturbList)
        params = self.getParams(verbose=verbose)  # get the parameters
        # fail if nPerturb > len(params)

        modParams = {}  # parameters want to modify
        # iterate over nPerturb perturbing those parameters that are floats..
        countFloat = 0
        for param, value in params.items():

            if countFloat > nPerturb:  # done perturbing
                break  # exit the loop
            if isinstance(value, float):
                modParams[param] = value * pScale  # scale value.
                countFloat += 1  # found another float.

        return modParams

    def perturb(self, params=None, verbose=False):
        """
        Perturb configuration. This method updates config info & history info by list of parameters perturbed.
          It does not modify runStatus. This is useful if the model crashes and needs rerunning with a perturbation.
        :param params -- parameters to perturb. If not set then the list of previous perturbations will be returned
          and nothing will be changed in the configuration.
        :param verbose -- be verbose. Currently does nothing..

        :return: list of perturbations.
        """
        perturbList = self.get('perturbList', default=[])
        if params is None:
            return perturbList  # just return the   list

        # got some params so add to list.
        perturbList.append(params)
        self.setReadOnly(readOnly=False)  # let us write to the configuration.
        history = self.get('history', default={})  # get the history info.
        perturb = history.get('perturb', [])
        perturb.append(datetime.datetime.now())
        perturb.extend(perturbList)
        history.update(perturb=perturb)
        self.set2(history=history)
        self.set2(perturbList=perturbList)
        self.setReadOnly(readOnly=True)  # no more writing to the configuration.
        return perturbList

    def submit(self, runStatus=None):
        """
        Provides full path to submit script AND makes it executable if it exists.
        :param runStatus -- If 'start' return path to new submit script 
                            If 'continue' return path to continuation submit script (defined in self.SubmitFile)
                            If None then use value from runStatus method to chose.
         new submit and continue submit are both defined in self.SubmitFile
        :return: path to appropriate submit script that should be ran to submit the model
        """

        if runStatus is None:
            newSubmit = self.runStatus()
        else:
            newSubmit = runStatus
        script = pathlib.Path(self.dirPath)/self.SubmitFiles[newSubmit]
        # if it exists make it executable.
        if script.exists():
            mode=script.stat().st_mode | stat.S_IXUSR
            script.chmod(mode)

        return script

    def createPostProcessFile(self, postProcessCmd):

        """
        Used by the submission system to allow the post-processing job to be submitted when the simulation
        has completed. As needs to be implemented for each model  this abstract version just raises an
        NotImplementedError with message to create version for your model.

        :param postProcessCmd -- a string which is the postProcessCmd. For example qrls XXXX.n
        :return -- the path to the file that was created/modified. (could be a list or something else)
        """

        raise NotImplementedError("implement createPostProcessFile for your model")

        file= 'notImplimented'
        return file

    def nl_info(self,params):
        """
        Work out namelists etc to be changed
        Returns dict indexed by filename with nl info as values.
        TODO add test for this case
        :return:
        """
        files = dict()  # list of files to be modified.
        for param,value in params.items():  # Iterate over parameters and thier values
            # search functions first
            if param in self._metaFn:  # got a meta function.
                logging.info(f"Running function {self._metaFn[param].__name__}")
                metaFnValues = self._metaFn[param](value)  # call the meta param function which returns a dict
                for conv, v in metaFnValues.items():  # iterate over result of fn.
                    if conv.file not in files:
                        files[conv.file] = []  # if not come across the file set it to empty list
                    files[conv.file].append((v, conv))  # append the  value  & conversion info.
            elif param in self._convNameList:  # got it in convNameList ?
                for conv in self._convNameList[param]:
                    if conv.file not in files:
                        files[conv.file] = []  # if not come across the file set it to empty list
                    files[conv.file].append((value, conv))  # append the value  & conversion
            else:
                raise KeyError("Failed to find %s in metaFn or convNameList " % param)

        return files
##I don't think code below is used or necessary
class EddieModel(ModelSimulation):
    """
    A simple model suitable for running on Eddie -- this allows testing of the whole approach.
    """

    def __init__(self, dirPath, obsNames=None,
                 create=False, refDirPath=None, name=None, ppExePath=None, ppOutputFile=None, studyConfig=None,
                 # options for creating new study
                 update=False,  # options for updating existing study
                 verbose=False, parameters=None):
        """
        Create an instance of HadCM3 class. Default behaviour is to read from dirPath and prohibit updates.
        :param dirPath -- path to directory where model simulation exists or is to be created
        :param create (optional with default False). If True create new directory and populate it.
            Afterwards the ModelSimulation will be readOnly.
            These options should be specified when creating a new study otherwise they are optional and ignored
            :param refDirPath -- reference directory. Copy all files from here into dirPath
            :param name -- name of the model simulation. If not provided will be taken from dirPath
            :param ppExePath --  path to post processing executable
            :param ppOutputFile -- output file for  post processing executable
            :param obsNames -- list of observations to be readin. (see readObs())
            :param studyConfig -- written into directory.
            :param parameters -- a dict on pandas series specifying hte parameter values
        :param update -- allow updates to the simulation information.
        :param verbose -- provide  verbose output. (See individual methods). Default is False.
        :returns initialised object.
        """

        # no parameters should be provided unless create or update provided
        if ((parameters is not None) and (len(parameters) > 0)) and not (create or update):
            raise ValueError("Provided parameters but not specified create or update")

        # call superclass init
        super(EddieModel, self).__init__(dirPath,
                                         obsNames=obsNames, create=create, refDirPath=refDirPath, name=name,
                                         ppExePath=ppExePath,
                                         ppOutputFile=ppOutputFile, parameters=parameters,
                                         # options for creating new study
                                         update=update,  # options for updating existing study
                                         verbose=verbose)
        if studyConfig is not None:
            studyConfig.save(
                filename=os.path.join(self.dirPath, "config.json"))  # write study configuration for fakemodel.



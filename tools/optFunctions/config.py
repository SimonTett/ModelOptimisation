"""Configuration information/functions for runOptimise and similar.
Provides standard optimisation function, fake function and submission
functions for  different systems

"""

import collections
import fileinput
import subprocess
import unittest
from contextlib import contextmanager

import numpy as np
import pandas as pd

import HadCM3
import optClimLib


# function for optimisation.
def stdOptFunction(params, ensembleMember=None, MODELRUN=None, df=False, *args, **kwargs):
    """
    Standard Function used for optimisation. Returns values from cache if already got it.
      If not got it return array of Nan  and set value to None.

    :param: params -- a numpy array with the parameter values. TODO: convince myself that ordering is fixed.
    :param: ensembleMember -- ensemble member for this case.


    TODO use "random" perturbation approach to find how many parallel cases we can run.
    """

    if MODELRUN is None:
        raise Exception("Specify MODELRUN")
    paramNames = MODELRUN.paramNames()
    # params can be a 2D array...
    if params.ndim == 1:
        use_params = params.reshape(1, -1)

    elif params.ndim == 2:
        use_params = params

    else:
        raise Exception("params should be 1 or 2d ")
    nsim = use_params.shape[0]
    nparams = use_params.shape[1]
    if nparams != len(paramNames):
        raise Exception(
            "No of parameters %i not consistent with no of varying parameters %i\n" % (nparams, len(paramNames)) +
            "Params: " + repr(use_params) + "\n paramNames: " + repr(paramNames))
    nObs = len(MODELRUN.obsNames())  # How many observations are we expecting?
    result = np.full((nsim, nObs), np.nan)  # array of np.nan for result
    for indx in range(0, nsim):  # iterate over the simulations.
        pDict = dict(zip(paramNames, use_params[indx, :]))  # create dict with names and values.
        if ensembleMember is not None:
            pDict.update(ensembleMember=ensembleMember)
        mc = MODELRUN.model(pDict, update=True)
        if mc is not None:  # got a model
            obs = mc.getObs(series=True)  # get obs from the modelConfig
            MODELRUN.paramObs(pd.Series(pDict), obs)  # store the params and obs.
            result[indx, :] = obs.reindex(MODELRUN.obsNames()).values # sort it and extract values.

    if df:
        result = pd.DataFrame(result,columns=MODELRUN.obsNames())



    return result


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def arcSubmit(model_list, config, rootDir, verbose=False, resubmit=None, *args, **kwargs):
    """
    Submit models to arc,  and the next iteration in the algorithm.
    :param model_list: a list of model configuration to submit
    :param config: study configuration file
    :param rootDir: root directory where files etc are to be created and found
    :param args -- the arguments that the main script was called with.
    :param resubmit -- default None. If not None next iteration cmd to be submitted.
        Normally should be the script the user ran so next stage of model running happens.
    :param runCode -- default None, If not none specifies the project code.
    :return: status of submission

    Does the following:
        2) Submits (if provided) resubmit so once the array of post processing jobs has completed the next bit of the algorithm gets ran.
        3) Submits the model simulations

    This algorithm is not particularly robust to failure -- if anything fails the various jobs will be sitting around
    Releasing them will be quite tricky! You can always kill and run again!
    """

    jobID = []
    for model in model_list:
        # put some dummy data in the ouput file
        modelSubmitName = model.submit()
        if verbose: print("Submitting ", modelSubmitName)
        with cd(model.dirPath):
            jID = subprocess.check_output("sbatch -J %s --export=ALL %s" % (model.name(), modelSubmitName),
                                          shell=True)  # submit the script (change devel after, and shouldn't have to ssh in)
        jobID.append(jID[20:-1])

    jobIDstr = ':$'.join(jobID)  # make single string appropriately formatted of job ids..
    # now re-run this entire script so that the next iteration in the algorithm.
    # can be run
    if resubmit is not None:
        # Submit the next job in the iteration. runOptimise is very quick so no need to submit to ARC again - just run on the front end.

        jobName = 'RE' + config.name()
        # TODO move to better python syntax for var printing. Think can use named vars in...
        cmd = ["sbatch -p devel --export=ALL --time=10 --dependency=afterany:%s -J %s " % (jobIDstr, jobName)]
        cmd.extend(resubmit)  # add the arguments in including the programme to run..
        cmd = ' '.join(cmd)  # convert to one string.
        if verbose: print("Next iteration cmd is ", cmd)
        jid = subprocess.check_output(cmd, shell=True)  # submit the script. Good to remove shell=True
        if verbose: print("Job ID for next iteration is %s" % jid[20:-1])

    return True


def easyFake(model, studyCfg, verbose=False):
    """
    Fake model -- computes obs without running! Write for your purpose.
    Fake model values --
    Assume that simulated values are a linear sum of effects from each parameter and that each parameter affects two
      types of parameters. With each region behaving similarly.
    :param model: model that we are running.
    :param studyCfg: study configuration -- which contains much useful information
    :param verbose (optional) If True (default is False) more information will be printed out.
    :return: simulated observations as a pandas series
    """
    paramNames = studyCfg.paramNames()  # index is names
    paramV = model.getParams(series=True)

    seed = optClimLib.genSeed(paramV)  # all parameters used to start the RNG.
    paramV = paramV.loc[paramNames]
    np.random.seed(seed)  # set the RNG up
    obsNames = studyCfg.obsNames()
    standardObs = studyCfg.standardObs(obsNames=obsNames, scale=False)  # standard obs
    nobs = standardObs.shape[0]
    index = [0, 1, 2, 3]
    linTerm = 200 * np.array([[0.1] * 10, [0.2] * 10, [0.3] * 10]).flatten()[0:nobs]
    sqrTerm = linTerm * 2.
    cubeTerm = linTerm * 4.
    pwr = pd.DataFrame([standardObs.values,
                        linTerm,  # linear term
                        sqrTerm,  # square term
                        cubeTerm],  # cubic term
                       index=index, columns=standardObs.index)
    cov = studyCfg.Covariances(scale=False)
    noise = cov['CovIntVar']  # noise.
    err = cov['CovTotal']
    obsSd = pd.Series(np.sqrt(np.diag(err)), index=err.columns)
    # noise = 0.0*noise
    # don't worry about noise on constraint as constraint gets re-computed!

    standardParam = studyCfg.standardParam(paramNames=paramNames)  # standard values

    rangep = studyCfg.paramRanges(paramNames=paramNames)
    delta_p = (paramV - standardParam) / rangep.loc['rangeParam', :]  # scale parameters
    delta_p = delta_p.dropna()  # remove any NaN's
    result = pd.Series(np.random.multivariate_normal(pwr.iloc[0].loc[obsNames].values,
                                                     noise.loc[obsNames, obsNames].values),
                       index=obsNames)
    # initialise with standard values + noise realisation.
    # hardwire for moment # need to make this a matrix -- converting from fn of parameters to fn of obs,
    # iterate over parameters
    # TODO -- make this work more generally...
    obsRoot = np.array([s.split("_", 1)[0] for s in obsNames if 'mslp' not in s])
    # extract obs root names (so region) but removing mslp
    nobs = len(obsRoot)  # how many different types of obs
    for i, param in enumerate(delta_p.index):
        obs = obsRoot[np.array([i, i + 3]) % nobs]  # those are the obs this parameter affects
        # and nhx, shx and tropics to them
        obs = [o + root for root in ['_nhx', '_shx', '_tropics'] for o in obs]  # full obs names
        # print(param,delta_p[param],"\n",pwr.ix[1:3, obs])
        for p in range(1, 4):  # hardwired limit on powers -- up to Cubic.
            addStuff = pwr.iloc[p].reindex(obs) * (delta_p[param] ** p) * obsSd.loc[obs]
            result.loc[obs] += addStuff
    # compute constraint
    obs = ['olr_nhx', 'olr_tropics', 'olr_shx', 'rsr_nhx', 'rsr_tropics', 'rsr_shx']
    wt = pd.Series([0.25, 0.5, 0.25, 0.25, 0.5, 0.25], index=obs)
    # overwrite the constraint value
    constraint = 340.25 - (result.reindex(obs) * wt).sum()
    result[studyCfg.constraintName()] = constraint
    result.rename(studyCfg.name())
    model.writeObs(result)

    return result

    ## end of fakeModel.


def eddieSubmit(model_list, config, rootDir, verbose=False, postProcess=True, resubmit=None, Submit=True, archiveDir=None, *args,
                **kwargs):
    """
    Submit models to eddie, the post processing and the next iteration in the algorithm.
    :param model_list: a list of model configuration to submit
    :param config: study configuration file
    :param rootDir: root directory where files etc are to be created and found
    :param verbose: be verbose
    :param postProcess If true submit postprocessing.
    :param args -- the arguments that the main script was called with.
    :param resubmit -- default None. If not None and postProcess next iteration cmd to be submitted.
          Normally should be the script the user ran so next stage of model running happens.
        post processing will be submitted the main script will not be submitted. For dummy submission this doesn't do much!
    :param Submit -- default True. If False then no jobs will be submitted.
    :param archiveDir -- default None. If not none then where data should be archived too. TODO: write code.
    :return: status of submission

    Does the following:
        1) Submits the post processing jobs as a task array in held state.
        2) Submits (if provided) resubmit so once the array of post processing jobs has completed the next bit of the algorithm gets ran.
        3) Submits the model simulations -- which once each one has run will release the appropriate post processing task

    This algorithm is not particularly robust to failure -- if anything fails the various jobs will be sitting around
    Releasing them will be quite tricky! You can always kill and run again!
    """
    outputDir = os.path.join(rootDir, 'jobOutput')  # directory where output goes.
    # try and create the outputDir
    try:
        os.makedirs(outputDir)
    except OSError:
        if not os.path.isdir(outputDir):
            raise
    runCode = config.runCode()
    configName = config.name()
    cwd = os.getcwd()  # where we are now.
    sshCmd = 'ssh login01.ecdf.ed.ac.uk " cd %s ; ' % (cwd)
    submitProcessCount = 0  # how many processes were submitted.
    # need to ssh to a login node to do things to Q's and cd to current dir
    if postProcess:
        modelDirFile = os.path.join(rootDir,'tempDirList.txt')
        # name of file containing list of directories for post processing stage
        with open(modelDirFile, 'w') as f:
            for m in model_list:
                text = m.dirPath + ',' + m.ppExePath() + ',' + m.ppOutputFile() + '\n'
                f.write(text)  # write out info for post processing job.
        # submit the following.. Need path to postProcess.sh
        jobName = 'PP' + config.name()
        ## work out postprocess script path
        postProcess = os.path.expandvars('$OPTCLIMTOP/eddie/postProcess.sh')
        scriptName = os.path.expandvars('$OPTCLIMTOP/eddie/qsub.sh')
        # need to run the resubmit through a script because qsub copies script beign run
        # so somewhere temporary. So lose file information needed for resubmit.
        qsub_cmd = 'qsub -l h_vmem=4G -l h_rt=00:30:00 -V '
        qsub_cmd += f'-cwd -e {outputDir} -o {outputDir}'

        # std stuff for submission
        # means        #  4 Gbyte Mem   30 min run, cur env, curr wd, output (error & std) in OutputDir
        # deal with runCode
        if runCode is not None: qsub_cmd += ' -P %s ' % (runCode)
        if len(model_list) == 0:
            print("No models to submit -- exiting")
            return False
        cmd = qsub_cmd + ' -t 1:%d -h -N %s ' % (len(model_list), jobName)
        cmd += postProcess
        cmd += " %s %s " % (modelDirFile, config.fileName())
        if verbose: print("postProcess task array cmd is ", cmd)
        # run the post process and get its job id

        if Submit:
            submitCmd = sshCmd + cmd + '"'
            if verbose:
                print("SUBMIT cmd is ", submitCmd)
            jid = subprocess.check_output(submitCmd, shell=True)
            #  '"' and shell=True seem necessary. Would be good to avoid both
            # and decode from byte to string. Just use defualt which may fail..
            jid = jid.decode()
            postProcessJID = jid.split()[2].split('.')[0]  # extract the actual job id as a string
            # TODO wrap this in a try/except block.
        else:
            postProcessJID = str(submitProcessCount)  # fake jid

        if verbose: print("postProcess array job id is %s" % postProcessJID)
        submitProcessCount += 1
        # write the jobid + N into the model -- for later when
        #  model gets some processing.
        for indx in range(len(model_list)):
            model_list[indx].jid = postProcessJID + '.%d' % (indx + 1)

    # Submit the models.


    # submit the archive script which will be held until the post-processing scripts have all ran.
    if archiveDir is not None:
        # TODO add archiveDIR to json configuration. And then can pass it through to
        # the submit function.
        raise NotImplementedError("archiving is not yet implemented and tested")
        jobName = 'Ar' + configName
        archive_cmd = 'qsub  -e %s -o %s' % (outputDir, outputDir)
        # put error and output into outputDir.
        archive_cmd += '-hold_jid %s -V -N %s %s %s %s ' % (postProcessJID, jobName, 'archive.sh', rootDir, archiveDir)
        # hold the job till postProcessing finished, name the job jobName and
        # run archive.sh with rootDir and archiveDir as arguments.
        if verbose:
            print("archiving command is %s" % archive_cmd)
        # now to actually submit it.
        if Submit:
            jid = subprocess.check_output(sshCmd + archive_cmd + '"', shell=True)
            # submit the script. Good to remove shell=True and '"'
            jid = jid.split()[2]  # extract the actual job id.
        else:
            jid = str(submitProcessCount)
        submitProcessCount += 1
        if verbose: print("Job ID for archving is %s" % jid)

    # now (re)submit this entire script so that the next iteration in the algorithm can be ran
    if (resubmit is not None) and postProcess:
        # submit the next job in the iteration. -hold_jid jid means the post processing job will only run after the
        # array of post processing jobs has ran.
        jobName = 'RE' + configName
        cmd = [qsub_cmd, f'-hold_jid {postProcessJID} -V -N {jobName} {scriptName}']
        cmd.extend(resubmit)  # add the arguments in including the programme to run..
        cmd = ' '.join(cmd)  # convert to one string.
        if verbose: print("Next iteration cmd is ", cmd)
        if Submit:
            submitCmd = sshCmd + cmd + '"'
            print("SubmitCmd is ",submitCmd)
            jid = subprocess.check_output(submitCmd,shell=True) 
            # submit the script. Good to remove shell=True and '"'
            jid = jid.split()[2]  # extract the actual job id.
        else:
            jid = str(submitProcessCount)
        submitProcessCount += 1
        if verbose: print("Job ID for next iteration is %s" % jid)
    # now submit the models
    for m in model_list:
        if postProcess:
            # need to put the post processing job release command in the model somehow. Depends on the model
            # but we have a mark and a file. So will modify the file. The model should define this..
            # and insert the mark into the file. Would I think be easier to keep the line no and goto that.
            with fileinput.input(m.postProcessFile, inplace=1, backup='.bak2') as f:
                for line in f:
                    # if m.postProcessFile does not exist then  get an error which is what we want!
                    # fix your model method!
                    if m.postProcessMark in line:  # got the mark so add some text.
                        print(sshCmd, 'qrls ', m.jid, '"')  # this releases the post processing job.
                    else:
                        print(line[0:-1])  # just print the line out.
            # dealt with modifying main file.
        modelSubmitName = m.submit()
        if verbose: print("Submitting ", modelSubmitName)
        if Submit:
            subprocess.check_output(sshCmd + modelSubmitName + '"', shell=True)  # submit the script
        submitProcessCount += 1

    if verbose: print("Submitted %i jobs " % (submitProcessCount))
    return True


# end of eddieSubmit


# define lookup tables that turn names into functions. This should include fairly standard things.

# stuff below if to allow reload of config -- if variables already exist they won't be deleted.
try:
    sz = len(modelFunctions)  # see if it exists!
except NameError:  # need to create them as they do not exist
    modelFunctions = collections.OrderedDict()
    submitFunctions = collections.OrderedDict()
    optFunctions = collections.OrderedDict()
    fakeFunctions = collections.OrderedDict()

# now have lookup tables  can set values up
modelFunctions.update(HadCM3=HadCM3.HadCM3)  # lookup table for model functions to run.Your model fn goes here.
modelFunctions.update(HadAM3=HadCM3.HadCM3)  # HadAM3 is HadCM3!
submitFunctions.update(eddie=eddieSubmit,
                       ARC=arcSubmit)  # lookup table for submission functions -- depends on architecture
optFunctions.update(default=stdOptFunction)  # Functions to be used for optimising
fakeFunctions.update(default=easyFake)  # function for faking run as part of testing.
# names for fake and optFunction

# test code
import tempfile
import os
import shutil
import StudyConfig
import collections
import copy
import ModelSimulation


class testSubmit(unittest.TestCase):

    def setUp(self):
        """
        Standard setup for all test cases
        :return:
        """
        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = self.tmpDir.name
        refDir = '$OPTCLIMTOP/test_in'

        testDir = os.path.expanduser(os.path.expandvars(testDir))
        refDir = os.path.expandvars(os.path.expanduser(refDir))
        if os.path.exists(testDir):  # remove directory if it exists
            shutil.rmtree(testDir, onerror=optClimLib.errorRemoveReadonly)
        self.dirPath = testDir
        self.refPath = refDir
        refDirPath = os.path.join(refDir, 'start')
        # now do stuff.

        jsonFile = os.path.join('Configurations', 'example.json')
        config = StudyConfig.readConfig(filename=jsonFile, ordered=True)  # parse the jsonFile.
        begin = config.beginParam()
        keys = begin.keys()
        keys = sorted(keys)
        parameters = collections.OrderedDict([(k, begin[k]) for k in keys])
        parameters.update(ensembleMember=0)
        models = []
        parameters.update(config.fixedParams())
        parameters.update(refDir=refDirPath)
        self.parameters = []
        for count, dir in enumerate(['zz001', 'zz002']):
            createDir = os.path.join(testDir, dir)
            parameters.update(ensembleMember=count)
            self.parameters.append(parameters.copy())
            models.append(ModelSimulation.ModelSimulation(createDir, name=dir, create=True,
                                                          refDirPath=refDirPath,
                                                          ppExePath=config.postProcessOutput(),
                                                          ppOutputFile=config.postProcessOutput(),
                                                          parameters=parameters.copy(), obsNames=config.obsNames(),
                                                          verbose=False
                                                          ))
            outFile = os.path.join(createDir, config.postProcessOutput())

            shutil.copy(os.path.join(refDir, '01_GN', 'h0101', 'observables.nc'), outFile)
            # copy over a netcdf file of observations.

        self.models = models
        self.config = copy.deepcopy(config)  # make a COPY of the config.

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        optClimLib.delDirContents(self.tmpDir.name)

    def test_easyFake(self):
        """
        Test that easyFake works
        :return:
        """
        import pandas as pd
        paramDF = pd.DataFrame({'VF1': [1.0, 1.5, 2.0], 'RHCRIT': [0.6, 0.7, 0.8]},
                               index=['run01', 'run02', 'run03'])
        o = easyFake(self.models[0], self.config)
        o2 = easyFake(self.models[1], self.config)

        self.assertFalse(np.all(o2 == o))


if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  # actually run the test cases

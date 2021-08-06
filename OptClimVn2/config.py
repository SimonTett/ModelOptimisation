"""Configuration information/functions for runOptimise and similar.
Provides fake function and submission functions for  different systems
"""

import collections
import fileinput
import subprocess
import unittest
from contextlib import contextmanager

import numpy as np
import pandas as pd

import HadCM3  # only model, currently, used.
import optClimLib


## set up fake fn.

def bare_fn(params, config=None, verbose=False, var_scales=None, rng=None):
    """
    Wee test fn for trying out things.
    :param params -- numpy array of parameter values
    :param config -- configurations -- default is None. If not available then fn will crash.
    :param verbose -- if True print out more information.
    :param var_scales -- scales to apply to variables.
        Should be a pandas series. If not provided no scaling will be done
    :param rng -- random number generator. Default None
      If provided than random multivariate noise  based on the internal variance covariance will be added.

    for everything but params given design of optimisation algorithms you will need to find a way of getting
      the extra params in. One way is to make a lambda fn. Another is to wrap it in a function.

    returns  "fake" data. If params is a pandas series then result will be a pandas series.
    """
    pranges = config.paramRanges()
    min_p = pranges.loc['minParam', :].values
    max_p = pranges.loc['maxParam', :].values
    scale_params = max_p - min_p
    pscale = (params - min_p) / scale_params
    pscale -= 0.5  # tgt is at params = 0.5
    result = 10 * (pscale + pscale ** 2)
    # this fn has one minima and the no maxima between the boundaries and the minim. So should be easy to optimise.
    if var_scales is not None:
        result /= var_scales.values  # make sure changes are roughly comparable size after scaling.

    tgt = config.targets().values

    delta_len = len(tgt) - result.shape[-1]
    if delta_len < 0:
        tgt = np.append(tgt, np.zeros(-delta_len))  # increase tgt
    elif delta_len > 0:
        result = np.append(result, np.zeros(delta_len), axis=-1)  # increase result

    result += tgt
    if rng is not None:
        intVar = config.Covariances()['CovIntVar']
        result += rng.multivariate_normal(tgt.values, intVar)  # add in some noise usually not needed for testing

    return result


def fake_fn(model, studyCfg, verbose=False, var_scales=None, rng=None):
    """
    Fake model -- computes obs without running! uses bare_fn to actually compute values.
    Fake model values -- uses fn to actually compute values but modifies slightly. (defined above)

    :param model: model that we are running.
    :param studyCfg: study configuration -- which contains much useful information
    :param verbose (optional) If True (default is False) more information will be printed out.
    :param var_scales -- scaling applied to each obs -- used by bare_fn.  See that for documentation.
    :param rng -- random number generator. See bare_fn to see what this does
    :return: simulated observations as a pandas series
    """

    paramNames = studyCfg.paramNames()
    obsNames = studyCfg.obsNames()

    simObs = bare_fn(model.getParams(series=True, params=paramNames).values, config=studyCfg, verbose=verbose,
                     var_scales=var_scales, rng=rng)  # get values
    simObs = pd.Series(simObs, index=obsNames)  # turn into a pandas series
    model.writeObs(simObs, verbose=verbose)  # and write them out


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
    raise NotImplementedError("Needs to be updated")
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


def eddieSubmit(model_list, config, rootDir, verbose=False, postProcess=True, resubmit=None, Submit=True,
                archiveDir=None, *args,
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
        modelDirFile = os.path.join(rootDir, 'tempDirList.txt')
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
            print("SubmitCmd is ", submitCmd)
            jid = subprocess.check_output(submitCmd, shell=True)
            # submit the script. Good to remove shell=True and '"'
            jid = jid.split()[2]  # extract the actual job id.
        else:
            jid = str(submitProcessCount)
        submitProcessCount += 1
        if verbose: print("Job ID for next iteration is %s" % jid)
    # now submit the models
    for m in model_list:
        if postProcess:
            # need to put the post processing job release command in the model. That is what postProcessFile does...
            cmd = sshCmd + f' qrls {m.jid} ' + '"'
            m.postProcessFile(cmd)

        modelSubmitName = m.submit()  # this gives the script to submit.
        if verbose:
            print("Submitting ", modelSubmitName)
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
    modelFunctions = dict()
    submitFunctions = dict()
    fakeFunctions = dict()

# now have lookup tables  can set values up
modelFunctions.update(HadCM3=HadCM3.HadCM3)  # lookup table for model functions to run.Your model fn goes here.
modelFunctions.update(HadAM3=HadCM3.HadCM3)  # HadAM3 is HadCM3!
submitFunctions.update(eddie=eddieSubmit,
                       ARC=arcSubmit)  # lookup table for submission functions -- depends on architecture
# optFunctions.update(default=stdOptFunction)  # Functions to be used for optimising
fakeFunctions.update(default=fake_fn)  # function for faking run as part of testing.
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


if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  # actually run the test cases

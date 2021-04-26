#!/usr/bin/env python
"""
A fake model for testing optimisation (or related) frameworks.
"""
import argparse # parse command line arguments
import collections # want ordered dict
import exceptions
import os
import shutil
import stat
import subprocess
import sys
import netCDF4
import numpy as np
import pandas as pd
from OptClimVn2 import ModelSimulation, StudyConfig, Optimise

def fakeModel(paramV, studyCfg, obsNames=None, trace=False):
    """
    Fake model values --
    Assume that simulated values are a linear sum of effects from each parameter and that each parameter affects two
      types of parameters. With each region behaving similarly.
    :param paramV: parameter vector to run at
    :param studyCfg: study configuration -- which contains much useful information
    :param obsNames: (optional) the observations to make. If not provided studyCfg will be queried for them
    :param trace (optional) If True (default is False) more information will be printed out.
    :return: simulated observations as a pandas object
    """

    if obsNames is None :
        use_obsNames=studyCfg.obsNames()
    else:
        use_obsNames=obsNames.copy()


    paramNames=paramV.index # index is names
    deterministic = True # hard wire -- set False if want truly random numbers
    # code below is largely cut-n-paste from optimise with name change and use of values
    if deterministic: # initialise RNG based on paramV
        # this is tricky because we need to convert them to integer arrays...
        values=paramV.values
        nonZero=np.where(np.abs(values) > 1e-9)
        int_v=np.int_(values)
        scale = 10.0**np.floor(np.log10(np.abs(values[nonZero]))-3) # allow a thousand different values
        int_v[nonZero]= np.int_(np.floor(np.abs(values[nonZero]/scale)+1)) #now got it as integer
        if (int_v.sum() < 0):
            raise ValueError("int_v stuffed")
        np.random.seed(int_v.sum()) # set the seed up
        if trace: print ": Seed set to ",int_v

    standardObs = studyCfg.standardObs(obsNames=use_obsNames,scale=False) # standard obs
    nobs=standardObs.shape[0]
    index=[0,1,2,3]
    linTerm=np.array([[0.1]*10,[0.2]*10,[0.3]*10]).flatten()[0:nobs]/studyCfg.scales(obsNames=obsNames)
    sqrTerm=linTerm*2.
    cubeTerm=linTerm*4.
    pwr=pd.DataFrame([standardObs.values,
                     linTerm, # linear term
                     sqrTerm, # square term
                     cubeTerm], # cubic term
                     index=index,columns=standardObs.index)
    cov=studyCfg.Covariances(scale=False)
    noise = cov['CovIntVar'] # noise.

    standardParam = studyCfg.standardParam(paramNames=paramNames) # standard values

    rangep = studyCfg.paramRanges(paramNames=paramNames)
    delta_p = (paramV-standardParam)/rangep.loc['rangeParam',:] # scale parameters
    result = pd.Series(np.random.multivariate_normal(pwr.ix[0,use_obsNames].values,
                                                     noise.loc[use_obsNames,use_obsNames].values),index=use_obsNames)
    # initialise with standard values + noise realisation.
    # hardwire for moment # need to make this a matrix -- converting from fn of parameters to fn of obs,
    # iterate over parameters
    obsRoot=np.array([s.split("_",1)[0] for s in use_obsNames if 'mslp' not in s]) # pack up strings into array to allow slice and dice
    nobs=len(obsRoot) # how many different types of obs
    for i,param in enumerate(paramNames):
        obs=obsRoot[np.array([i, i+1]) % nobs] # those are the obs this parameter affects
        ##pandas nhx, shx and tropics to them
        obs=[o+root for root in ['_nhx','_shx','_tropics'] for o in obs] # full obs names
        for p in range(1,4): # hardwired limit on powers -- up to Cubic.
            result.loc[obs] += pwr.ix[p,obs]*(delta_p[param]**p)

    # compute constraint
    obs = ['olr_nhx','olr_tropics','olr_shx','rsr_nhx','rsr_tropics','rsr_shx']
    wt= pd.Series([0.25,0.5,0.25,0.25,0.5,0.25],index=obs)

    constraint=pd.Series(340.25-(result.loc[obs]*wt).sum(),index=[studyCfg.constraintName()])
    result=pd.concat((result,constraint))
    return result
## main code

print "Running fakemodel.py"
jsonFile="config.json"
config= StudyConfig.OptClimConfig(jsonFile) # parse the jsonFile.
m = ModelSimulation.EddieModel('./',update=True)  # read the model configuration.
params=m.getParams()
obs=fakeModel(params,config) # compute observations.
# write data out as netCDF
rootgrp = netCDF4.Dataset('A/modelRun.nc', "w", format="NETCDF4")
for o in obs.index:  # iterate over index in series.
    v = rootgrp.createVariable(o, 'f8')  # create the NetCDF variable
    v[:] = obs[o]  # write to it
    print v, v[:]
    print "====================="
rootgrp.close()  # close the file

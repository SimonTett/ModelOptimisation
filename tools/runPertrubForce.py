#!/usr/bin/env python
"""
Python script to generate and submit simulations derived from optimised parameter cases.
Currently knows about:
ctl -- do a control simulation
ctlia -- do a control simulation with Jones et al, 2001 indirect aerosol scheme on.
atmos -- run an ensemble of two extended atmos only simulations
atmosia -- run an ensemble of two extended atmos only simulations with interactive indirect aerosol on.
1percent -- run a 1 percent simulation starting from the ctl in year 40.
1percentia --  run a 1 percent interactive indirect aerosol simulation starting from the ctl in year 40.
2xCO2 -- run an instantaneous 2xCO2 simulation
2xCO2ia -- run an instantaneous 2xCO2 simulation with initeractive aerosol
4xCO2 -- run an instantaneous 4xCO2 simulation.
4xCO2 -- run an instantaneous 4xCO2 simulation with interactive aerosol
It should be fairly easy to add more if needed. Hard work is to generate reference configuration to modify.

"""
import argparse
# TODO reeniginer to use Submit module so that post-processing can be ran.
# challange will be what post processing should be done.
# default should be to run mkpph and ummonitor on all dirs.
import os
import subprocess

import HadCM3 as HadCM3
import StudyConfig as StudyConfig

# TODO -- needs an update to
#  1) Remove six and so make "pure" python 3 Done.
#  2) import config and use it ??
#  3) Use submit module to control submission etc which will give caching and post-processing.
# But leave till actually got a need!

# what we know about.
refConfig={
    'ctl':('xnmea','HadCM3','ctl'), # control sim
    '1percent':("xnmeb",'HadCM3','1pe'), # 1%
    '2xCO2':("xnmec",'HadCM3',"2xC"), # 2xCO2
    '4xCO2':("xnmed",'HadCM3',"4xC"),  # 4xCO2
    'atmos':("xnmee",'HadAM3',"at"), # atmos only run
    'ctlia': ('xnmef','HadCM3','ctl'), # ctl with interactive indirect aerosols
    'atmosia': ("xnmeg","HadAM3","at"),  # atmos only run with interactive indirect aerosol
    '1percentia': ("xnmej", 'HadCM3', '1pe'),  # 1% with initeractive indirect aerosol
    '2xCO2ia': ("xnmeh", 'HadCM3', "2xC"),  # 2xCO2 with initeractive indirect aerosol
    '4xCO2ia': ("xnmei", 'HadCM3', "4xC"),  # 4xCO2 with initeractive indirect aerosol
   }
parser = argparse.ArgumentParser(description="Perturb and run one or all of ctl, 1%, 2xCO2 & 4xCO2 HadCM3 simulation")
parser.add_argument("jsonFile",help="json file that defines the run -- created by runOptimise  or similar")
parser.add_argument("-r","--runs",nargs='*',default='none',choices=refConfig.keys())
parser.add_argument("-d","--dir",help="path to root directory where model runs will be created. ",default=os.getcwd())
parser.add_argument("-P","--runcode",help="code to run model with",default=None)
parser.add_argument("-v","--verbose",action='store_true',help="Provide Verbose information")
parser.add_argument("--dryrun",action='store_true',help="if set do not submit any jobs but do create directories. Good for testing")

args=parser.parse_args() # parse cmd line arguments
runs=args.runs
if isinstance(runs, str):
    runs=[runs] # make it a 1 element list
jsonFile=os.path.expanduser(os.path.expandvars(args.jsonFile))
config= StudyConfig.readConfig(filename=jsonFile,ordered=True) # parse the jsonFile.
paramsInit=config.optimumParams().to_dict() # extract the optimum parameters
# add the fixed configurations but remove run_time, run_target
fixed=config.fixedParams()
for k in ['RUN_TARGET','START_TIME']:
    if k in fixed:
        del fixed[k]
paramsInit.update(fixed) # now have full list of params

baseRunID=config.baseRunID() # get the baseRunID.
refPath=os.path.join("$OPTCLIMTOP","Configurations") # where the reference configurations live
sshCmd = 'ssh login04.ecdf.ed.ac.uk " cd %s ; ' % (os.getcwd())  # need to ssh to a login node to do things to Q's and cd to current dir


models=[]
for sim in runs: # iterate over cases to be done.
    params = paramsInit.copy() # copy inital params -- will then modify them
    ref=refConfig[sim]
    refDir=os.path.join(refPath,ref[0]) # path to reference HadCM3 case
    refDir=os.path.expandvars(os.path.expanduser(refDir)) # expand all env vars and user stuff
    runID=baseRunID+ref[2] # generated runID for the model.
    params.update(RUNID=runID)
    dir=os.path.join(args.dir,ref[1],sim,runID) # work out dir for run
    if sim in  ['ctl','ctlia']:
        models.append(
                HadCM3.HadCM3(dir, name=runID, runCode=args.runcode, create=True, refDirPath=refDir,
                      verbose=args.verbose, parameters=params)
                      )  # generate the model.
    elif sim in ['1percentia','2xCO2ia','4xCO2ia']:
        # need to work out and set AINITIAL, OINITIAL which is how these configs set up.
        ctlName = baseRunID + "ctl"
        # assume ctl dumps already  exist in $MYDUMPS which is defined in the configuration.
        refAtmosDump = os.path.join('$MY_DUMPS', ctlName + 'a@da041c1')
        refOcnDump = os.path.join('$MY_DUMPS', ctlName + 'o@da041c1')
        params.update(AINITIAL=refAtmosDump,OINITIAL=refOcnDump)
        models.append(
            HadCM3.HadCM3(dir,name=runID, runCode=args.runcode,create=True, 
                          refDirPath=refDir, verbose=args.verbose, parameters=params)
                      ) # generate the model.
    elif sim in ['1percent','2xCO2','4xCO2']:
        # need to work out and set ADUMP, ODUMP which is how these configs set up.
        ctlName = baseRunID + "ctl"
        # assume ctl dumps already  exist in $MY_DUMPS which is defined in the configuration.
        refAtmosDump = os.path.join('$MY_DUMPS', ctlName + 'a@da041c1')
        refOcnDump = os.path.join('$MY_DUMPS', ctlName + 'o@da041c1')
        params.update(ASTART=refAtmosDump,OSTART=refOcnDump)
        models.append(
            HadCM3.HadCM3(dir, name = runID,runCode=args.runcode,create=True, refDirPath=refDir,
                          verbose=args.verbose,parameters=params)
                      ) # generate the model.
    elif sim in ['atmos','atmosia']: # atmos only -- run two simulations.
        for index in ['1','2']:
            rid=runID+index
            params.update(RUNID=rid)
            d=dir+index # dir for this ensemble member
            models.append(HadCM3.HadCM3(d, name=rid, runCode=args.runcode, create=True, refDirPath=refDir,
                                        verbose=args.verbose,
                                         parameters=params)
                          )  # generate the model.
    else: # failed to find case
        raise  Exception("Failed to find %s"%(sim))
# Done generating all models so lets submit them
# TODO replace this with submit code??
for model in models:
    submit=model.submit()
    cmd=sshCmd + submit + '"'
    if args.dryrun: # dryrun so don't submit
        print("Run %s not being submitted: cmd is %s"%(model.name(),cmd))
    else:
        subprocess.check_output(cmd, shell=True)  # actually submit the script




#!/usr/bin/env python
"""
Test that can generate AND submit a HadAM3 configuration.
NEED to define OPTCLIMTOP & python path.

"""

import HadCM3 as HadCM3
import Submit as Submit
import StudyConfig as StudyConfig
import os
runid='a9999' 
name='a9999'
testDir='test_out'
obsNames=['temp@500_nhx', 'temp@500_tropics', 'temp@500_shx']
refDir=os.path.join("Configurations","HadAM3_ed3_SL7_15m")
configPath=os.path.join("Configurations","example.json")
config=StudyConfig.readConfig(configPath,ordered=True)
exe=os.path.join(os.environ['OPTCLIMTOP'],'um45','obs_in_nc','comp_obs.py')
mlist=[]
start_time=[1999,12,1] # post processing does from 2000-03-01 so need to get a whole year done!
run_target=[1,3,1] # short run
runTime=4*6*60*1.10
# time for a season is ~6 mins, 10% leeway and want to run for 4 seasons
mlist.append(HadCM3.HadCM3(testDir, name='a0001', create=True, refDirPath=refDir,
                           ppExePath=exe, ppOutputFile='obs.nc',
                           obsNames= config.obsNames(),runTime=runTime,
                           verbose=True,
                           START_TIME=start_time,RUN_TARGET=run_target)) # std case

mlist.append(
    HadCM3.HadCM3(testDir+'2', name='a0002', create=True, 
                  refDirPath=refDir,
                  ppExePath=exe, ppOutputFile='obs.nc',runTime=runTime,
                  obsNames= config.obsNames(),ALPHAM=0.6,
                  START_TIME=start_time,RUN_TARGET=run_target)
) # perturb ice case

Submit.eddieSubmit(mlist,config,'.',verbose=True)

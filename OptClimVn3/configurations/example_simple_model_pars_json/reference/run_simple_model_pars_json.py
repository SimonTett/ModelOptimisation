#!/bin/env python
# test simple model code. Reads in config file and parameter set then generates fake fields of obs based on this.
# The obs are set to go with a simple post-process script that just copies them. Submit with qsub.
# The challenge is how to get information on where the StudyConfig file is.
import pathlib

from genericLib import fake_fn
import StudyConfig
import json
import time
import numpy as np
import sys



cpath = pathlib.Path('basic_config.json')
config = StudyConfig.readConfig(cpath)  # read in the config.
with open("parameters.json", 'r') as fp:
    params = json.load(fp)  # read in the parameters

# params is now grouping.
system_params = params['system']
# remove any keys that look like comments from params
model_params = {k:v for k,v in params['model_params'].items() if not k.endswith('_comment')}


print("Model Parameters are ",model_params)
sleep_time = system_params.get('sleep_time',15)
# time model  is running. Long enough can check in the Q; short enough to get rapid turnaround to test system
time.sleep(sleep_time)
# see if we fail.
pfail = system_params.get('fail_probability',0.0)
if np.random.uniform(0,1.0) < pfail:
    print("Failed (randomly)")
    sys.exit(1)


sim_obs = fake_fn(config, model_params).to_dict() # generate some fake obs!
with open("model_output.json", "w+t") as fp: # and write them out
    json.dump(sim_obs, fp,indent=2)
#sys.exit(0) # we are finished

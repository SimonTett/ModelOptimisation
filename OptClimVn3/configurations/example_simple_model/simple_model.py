#!/bin/env python
# test simple model code. Reads in config file and parameter set then generates fake fields of obs based on this.
# The obs are set to go with a simple post-process script that just copies them. Submit with qsub.
from genericLib import fake_fn
import StudyConfig
import json
import time
import subprocess
import importlib.resources
traverse = importlib.resources.files('Models')
pth_root = traverse.as_file('scripts/set_model_status.py')
if not pth_root.exists():
    raise ValueError(f"Script at {pth_root} does not exist")
subprocess.check_output([str(pth_root),'RUNNING'])
config = StudyConfig.readConfig('config.json')  # read in the config.
with open("params.json", 'r') as fp:
    params = json.load(fp)  # read in the parameters
time.sleep(15)  # model  is running. Long enough can check in the Q; short enough to get rapid turnaround to test system
sim_obs = fake_fn(config, params).to_dict() # generate some fake obs!
with open("model_output.json", "w") as fp: # and write them out
    json.dump(sim_obs, fp)

subprocess.check_output([str(pth_root),'SUCCEEDED']) # we have ran OK!

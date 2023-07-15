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
import argparse

parser = argparse.ArgumentParser(description="A simple model. For testing.")
parser.add_argument("config_path",type=str,help="path to config file")
args = parser.parse_args()
cpath = pathlib.Path(args.config_path)
config = StudyConfig.readConfig(cpath)  # read in the config.
with open("params.json", 'r') as fp:
    params = json.load(fp)  # read in the parameters

print("Parameters are ",params)
sleep_time = params.pop('sleep_time',15)
# time model  is running. Long enough can check in the Q; short enough to get rapid turnaround to test system
time.sleep(sleep_time)
# see if we fail.
pfail = params.pop('fail_probability',0.0)
if np.random.uniform(0,1.0) < pfail:
    print("Failed (randomly)")
    exit(1)


sim_obs = fake_fn(config, params).to_dict() # generate some fake obs!
with open("model_output.json", "w") as fp: # and write them out
    json.dump(sim_obs, fp)
exit(0) # we are finished

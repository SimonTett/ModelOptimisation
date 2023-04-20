#!/usr/bin/env python

"""
test post-processing. Does nothing except return some "fake" obs.
But useful as an example.

"""
import argparse

import generic_json
from model_base import model_base # so we have expand

parser = argparse.ArgumentParser(description="""
    test_pp_script
    test script for post-processing Example use is:
        pp_script_test.py input_file output_file 
      """
                                 )
parser.add_argument("INPUT", help="The Name of the input file. Should be readable by generic_json.load and contain a "
                                  "dict called fake_obs ")
parser.add_argument("OUTPUT",  help="The name of the output file. ")
parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
args = parser.parse_args()  # and parse the arguments
# expand filenames
input = model_base.expand(args.INPUT)
output = model_base.expand(args.OUTPUT)
with open(input, "r")  as fp: # load up the post-process data.
    post_process = generic_json.load(fp)

sim_obs = post_process["fake_obs"]
if not isinstance(sim_obs,dict):
    raise ValueError(f"fake_obs of type {type(sim_obs)}")

with open(output,'w') as fp:
    generic_json.dump(sim_obs,fp)
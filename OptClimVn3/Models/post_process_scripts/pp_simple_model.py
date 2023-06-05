#!/bin/env python
"""
Simple post processing for simple_model.
Copies the input to output with no changes.
"""


import argparse  # parse arguments
import json  # get JSON library
import os  # OS support
import pathlib
import logging

import numpy as np
import StudyConfig
import genericLib
import subprocess


parser = argparse.ArgumentParser(description="Copy output from simple_model.py to output file"
                                 )
parser.add_argument("CONFIG", help="The Name of the Config file")
parser.add_argument("-d", "--dir", help="The Name of the input directory")
parser.add_argument("OUTPUT", nargs='?', default=None,
                    help="The name of the output file. Will override what is in the config file")
parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
args = parser.parse_args()  # and parse the arguments
# setup processing

config = StudyConfig.readConfig(args.CONFIG)
options = config.getv('postProcess', {})
input_file = pathlib.Path("model_output.json")
if args.dir:
    input_file=pathlib.Path(args.dir)/input_file


mask_file = options.get('mask_file')
if mask_file is not None:
    mask_file = genericLib.expand(mask_file)
mask_name = options.get('mask_name')
start_time = options.get('start_time')
end_time = options.get('end_time')
if args.OUTPUT is None:
    output_file = options['outputPath']  # better be defined so throw error if not
else:
    output_file = args.OUTPUT

output_file = genericLib.expand(output_file)  # expand users and env vars.
if output_file.suffix != '.json':
    raise ValueError("Expecting json output. Please modify your post-processing script")

verbose = args.verbose

if verbose:  # print out some helpful information..
    print("CWD", pathlib.Path.cwd())
    print("mask_file", mask_file)
    print("land_mask", mask_name)
    print("start_time", start_time)
    print("end_file", end_time)
    print("output", output_file)
    print("dir",args.dir)
    if verbose > 1:
        print("options are ", options)

# and do the post processing by copying the input file to output file!
# Do this by reading in the json file and checking nothing is NaN. If it is NaN something went wrong...

with open(input_file,'r') as fp:
    input_obs = json.load(fp)

if any([value == np.nan] for value in input_obs.values()):
    raise ValueError(f"Got some nan values in input from {input_file}")
logging.info(f"Read {input_obs} from {input_file}")

with open(output_file,'w') as fp:
    json.dump(input_obs,fp) # and dump it
logging.info(f"Dumped {input_obs} to {output_file}")



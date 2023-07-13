#!/bin/env python
"""
Simple post-processing for simple_model.
Copies the input to output with no changes.
"""

import argparse  # parse arguments

import pathlib
import logging
import typing

import numpy as np
import json
import genericLib
import datetime


def time_to_vect(time: str, ym_only: bool = True) -> typing.List:
    """
    Convert an iso time str to 6 member vector (y,m,d,hr,min,sec)
    :param time: String
    :param ym_only: If True return only year and month -- useful for non standard calendars.
    :return: v
    """
    result = datetime.datetime.fromisoformat(time)  # convert iso-format time to datetime
    result= [result.year, result.month, result.day, result.hour, result.minute, result.second]
    if ym_only:
        result = result[0:2]
    return result

parser = argparse.ArgumentParser(
    description="Copy output from run_simple_model.py to output file"
)
# put your extra stuff here. then call the std_post_process_setup
args, post_process = genericLib.std_post_process_setup(parser)

input_file = args.dir / "model_output.json"

mask_file = post_process.get('mask_file')
if mask_file is not None:
    mask_file = genericLib.expand(mask_file)
mask_name = post_process.get('mask_name')
start_time = time_to_vect(post_process.get('start_time'))

end_time = time_to_vect(post_process.get('end_time'))

if args.OUTPUT.suffix != '.json':
    raise ValueError("Expecting json output. Please modify your post-processing script")

logging.info(f"""
               CWD: {pathlib.Path.cwd()}
               mask_file: {mask_file}
               land_mask: {mask_name}
               start_time: {start_time}
               end_file: {end_time}
               input_file: {input_file}
               output: {args.OUTPUT}
               dir: {args.dir}
""")

# and do the post processing by copying the input file to output file!
# Do this by reading in the json file and checking nothing is NaN. If it is NaN something went wrong...

with open(input_file, 'r') as fp:
    input_obs = json.load(fp)

if any([np.isnan(value) for value in input_obs.values()]):
    raise ValueError(f"Got some nan values in input from {input_file}")
logging.info(f"Read {input_obs} from {input_file}")

with open(args.OUTPUT, 'w') as fp:
    json.dump(input_obs, fp)  # and dump it
logging.info(f"Dumped {input_obs} to {args.OUTPUT}")

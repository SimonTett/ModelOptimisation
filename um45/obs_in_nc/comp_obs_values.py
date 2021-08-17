#!/usr/bin/env python
"""
Compute global and regional mean values for a bunch of input files.
"""
from comp_obs_xarray import means  # want the same fn
import argparse
import xarray
import StudyConfig
import pathlib
import glob
import json
import os

parser = argparse.ArgumentParser(description="""
    Post process observational  data to provide global & regional values. Example use is:

    comp_obs_values.py input.json output.json data/*N48.nc

    Observations are:
    Global mean, Northern Hemisphere Extratropical average (30N-90N), Southern Hemisphere Extra-tropical (90S-30S) mean and 
    Tropical (30S - 30N) mean 
    If variable looks like MSLP then it will be converted to:
      Northern Hemisphere Extra-tropical and Tropical Mean Sea Level Pressure difference from global average
    """
                                 )
parser.add_argument("CONFIG", help="The Name of the Config file")
parser.add_argument("OUTPUT", help="The Name of the Output JSON file")
parser.add_argument("FILES", nargs='*', help='Files to process')
parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
args = parser.parse_args()  # and parse the arguments
verbose = args.verbose
config = StudyConfig.readConfig(args.CONFIG)
output_file = os.path.expandvars(args.OUTPUT)
options = config.getv('postProcess', {})
files = []
for filep in args.FILES:
    if verbose:
        print(f"File pattern {filep}")

    fileList = [pathlib.Path(p) for p in glob.glob(filep)]
    files.extend([file for file in fileList if file.suffix == '.nc'])



if verbose:
    print("Files to process are ", files)

start_time = options['start_time']
end_time = options['end_time']
if verbose:
    print("start_time", start_time)
    print("end_file", end_time)
    if verbose > 1:
        print("options are ", options)
# iterate over files and then the variables in the files.
results = dict()  # where we store results
for file in files:
    ds = xarray.open_dataset(file).sel(time=slice(start_time, end_time))
    file_result = dict() # collect up by file
    # iterate over data variables.
    for v in ds.data_vars:
        try:
            var = ds[v]
        except ValueError:
            continue
        #possibly scale units...
        unit = var.attrs.get('units')
        if (unit == 'm') and ('ERA5' in file.name):
            if verbose:
                print("Scaling ERA5 precip to kg/second from m/day")
            var *= 1000/(24*60*60.)
        if unit == 'mm/month':
            var /= var.time.dt.days_in_month*24*60*60 # convert to kg/sec
        if unit == "degrees Celsius": # convert to K
            var += 273.16
        try:
            latitude_coord = list(var.coords.dims)[2]
            latitude_coord = 'latitude'
        except IndexError:
            continue
        name = v
        mn_values = means(var, name, latitude_coord=latitude_coord)
        # now potentially deal with pressure
        if v in ['msl','mslp']:
            if verbose:
                print(f"Sorting pressure for {file}")
            mn_values.pop(f'{name}_SHX')
            for k in [f'{name}_NHX', f'{name}_TROPICS']:
                mn_values[k + 'DGM'] = mn_values.pop(k) - mn_values[f'{name}_GLOBAL']
        if verbose:
            print(mn_values)
        file_result.update(mn_values)
        # end loop over vars
    results[str(file)]=file_result
    # end loop over files

# now to write out the values.

with open(output_file, 'w') as fp:
    json.dump(results, fp, indent=2)

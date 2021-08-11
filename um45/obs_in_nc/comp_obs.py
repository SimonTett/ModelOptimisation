#!/usr/bin/env python

""" Compute simulated observables 
Observations are:
Global mean, Northern Hemisphere Extratropical average (30N-90N), Southern Hemisphere Extra-tropical (90S-30S) mean and 
Tropical (30S - 30N) mean for:

  Temperature at 500 hPa,
  Relative Humidity at 500 hPa
  Outgoing Longwave Radiation (all and clear sky)
  Reflected Shortwave Radiation (all and clear sky)
  Land air temperature at 1.5m (north of 60S)
  Land precipiation at 1.5m  (north of 60S)
  Cloud effective radius

and Northern Hemisphere Extra-tropical and Tropical Mean Sea Level
Pressure difference from global average

 """
import numpy  # numpy
import datetime  # get datetime module
import iris  # Met Office IRIS library
from iris.time import PartialDateTime
import iris.analysis.cartography
import json  # get JSON library
import copy  # allow copying of objects
import argparse  # parse arguments
import os  # OS support
import pathlib


##


def periodConstraint(cube, start_time=None, end_time=None):
    """
    Generate a time constraint between start_time and end_time. start_time is included while end_time is not included in 
    the times requested. start_time and end_time should be passed in as tuples/or lists 
    """
    Units = cube[0].coord('time').units  # get the units of time so we can convert to them!

    if (start_time is None):
        t1 = -1e20  # large -ve time
    else:
        t1 = Units.date2num(datetime.datetime(*start_time))  # * expands the tupple into hopefully three

    if (end_time is None):
        t2 = 1e20  # large +ve time
    else:
        t2 = Units.date2num(datetime.datetime(*end_time))  #

    return iris.Constraint(time=lambda cell: t1 <= Units.date2num(cell.point) < t2)


def fix_missing_data(cube, field, filename):
    """
    callback for iris.load_cube to fix missing data problems
    :param cube -- cube being loaded and modified
    :param  field -- field being loaded???
    :param filename -- file name
    """

    # extract the data
    data = cube.data
    # missing data problems likely if max/min close to 1e20.
    if (numpy.max(numpy.abs(data)) > 1e19) and (numpy.sum(data.mask) == 0):  # 1e19 looks like reasonable guard value
        cube.data = numpy.ma.masked_outside(data, -1e19, 1e19)


def means(cube, name=None, start_time=None, end_time=None, mask=None, mask_name=None, mask_attribute=None, verbose=0,
          timeMean=True):
    """ 
    Compute means for NH extra tropics, Tropics and SH extra Tropics. 
    Tropics is 30N to 30S. NH extra tropics 30N to 90N and SH extra tropics 90S to 30S 
    Arguments:
    :param cube -- cube to be processed
    :param name )optional -- default =
    :param timeMean If true (defaut) extract and time-mean data.

    """

    ## check inputs for consistency
    if ((mask is not None) and (mask_name is None)):
        raise Exception('Need to specify mask_name when specifying mask')

    if ((mask is not None) and (mask_attribute is None)):
        raise Exception('Need to specify mask_attribute when specifying mask')

    if (name is not None):
        int_name = name
    else:
        int_name = 'VAR'

    cube_internal = cube.copy()  # make a copy of the cube as I want to potentially apply a mask to it
    print(int_name, type(cube_internal.data), type(cube.data))
    ## possibly fix cube long/lat bounds
    for coord in ['longitude', 'latitude']:
        if (not (cube_internal.coord(coord).has_bounds())):  # bounds not set if so guess them.
            cube_internal.coord(coord).guess_bounds()

    ## possibly deal with mask
    if (mask is not None):
        int_name = mask_name + int_name
        if timeMean:
            lmask = iris.util.broadcast_to_shape(numpy.squeeze(mask.data), cube_internal.shape, [1, 2])
        else:
            lmask = numpy.squeeze(mask.data)
        cube_internal.data = numpy.ma.array(cube_internal.data, mask=(lmask == 0))
        cube_internal.attributes['mask_attrib'] = mask_attribute

    ## constraints
    constraints = {
        'GLOBAL': None,
        'NHX': iris.Constraint(coord_values={'latitude': lambda y: y > 30.0}),
        'TROPICS': iris.Constraint(coord_values={'latitude': lambda y: -30 <= y <= 30.0}),
        'SHX': iris.Constraint(coord_values={'latitude': lambda y: y < -30.0}),
    }

    means = []  # empty list  to put results in
    if timeMean:

        time_constraint = periodConstraint(cube_internal, start_time, end_time)  # time range we want
        vars = ['time', 'latitude', 'longitude']
    else:
        time_constraint = iris.Constraint(None)  # no time constraint/
        vars = ['latitude', 'longitude']

    for key in constraints:  # iterate over constraints
        constraint = constraints[key]
        cc = cube_internal.extract(time_constraint & constraint)  # extract to the sub-region and time  we want
        if (verbose >= 3):
            print("CC is size", cc.shape)

        wt = iris.analysis.cartography.area_weights(cc)  # compute the area weights
        mn = cc.collapsed(vars, iris.analysis.MEAN, weights=wt)  # average and store in the hash

        mn.rename(int_name + "_" + key)  # give processed cube a useful name
        means.append(mn)  # append to the list

    means = iris.cube.CubeList(means)  # convert data from list to CubeList
    return means  # means are what we want


###

## parse input arguments

parser = argparse.ArgumentParser(description=
                                 """Post process Unified Model data to provide 26 simulated observations. Example use is:

 comp_obs.py input.json /exports/work/geos_cesd_workspace/OptClim/Runs/st17_reg/01/s0101/A/output.nc.


Observations are:

Global mean, Northern Hemisphere Extratropical average (30N-90N), Southern Hemisphere Extra-tropical (90S-30S) mean and 
Tropical (30S - 30N) mean for:

  Temperature at 500 hPa,

  Relative Humidity at 500 hPa

  Outgoing Longwave Radiation

  Reflected Shortwave Radiation

  Land air temperature at 1.5m (north of 60S)

  Land precipitation at 1.5m  (north of 60S)

  Clear sky outgoing longwave

  Clear sky reflected shortwave

  Effective cloud radius
  

and Northern Hemisphere Extra-tropical and Tropical Mean Sea Level Pressure difference from global average
"""
                                 )
parser.add_argument("JSON_FILE", help="The Name of the JSON file",
                    type=argparse.FileType('r'))
parser.add_argument("-i", "--input", help="The Name of the input file")
parser.add_argument("OUTPUT", help="The Name of the output file")
parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
parser.add_argument("--notmean", help="Do not time mean data", action="store_true")
args = parser.parse_args()  # and parse the arguments
# setup processing

json_file = args.JSON_FILE  # JSON file for reading
json_info = json.load(json_file)  # read the JSON file
json_file.close()  # close file

# work out the files if needed
if args.input is None:
    files = glob.glob(os.path.join('A', 'aps', '*.pp'))
else:
    files = args.input

try:
    options = json_info['postProcess']  # get the options
except KeyError:
    raise Exception("Need to provide post_process_options in " + args.JSON_FILE)

mask_file = options['mask_file']
mask_file = os.path.expanduser(os.path.expandvars(mask_file))
mask_name = options['mask_name']
start_time = options['start_time']
end_time = options['end_time']

verbose = args.verbose

if verbose:  # print out some helpful information..
    print("mask_file", mask_file)
    print("land_mask", mask_name)
    print("start_time", start_time)
    print("end_file", end_time)
    if (verbose > 1):
        print("options are ", options)

output_file = args.OUTPUT
timeMean = not args.notmean
land_mask = iris.load_cube(mask_file, mask_name)  # land/sea mask
mask_attr = 'file: ' + mask_file + ' Name: ' + mask_name

constrain_60S = iris.Constraint(coord_values={'latitude': lambda y: y >= -60})

land_mask_60S = land_mask.extract(constrain_60S)  # mask noth of 60S
mask_attr_60S = 'file: ' + mask_file + ' Name: ' + mask_name + " > 60S"

## now to process data. Below could be simplified with a loop and constraints but not worth the hassle at this point.

## want to add effective liquid cloud  radius. This computed from cld_re_wt/cld_wt
## cld_re_wt=1245; cld_rd=1246

constraint = [iris.Constraint('air_temperature', pressure=500.),

              iris.Constraint('air_temperature', plev=50000.)]


def load_try(files, name, constraints):
    """
    Try and load variables from files using iris and multiple constraints
    :param files -- files for which dta is to be loaded
    :param name -- variable name 
    (see iris.load_cube for full explanation)
    :param constrains -- list of iris constraints. First one matched returned

    
    """
    for constraint in constraints:
        try:
            var = iris.load_cube(files, name & constraint, callback=fix_missing_data)
            return var
        except iris.exceptions.ConstraintMismatchError:
            pass
            # print("Failed to find cube trying next constraint")

    raise Exception("Failed to load cube at all.. Sorry")


constraints = [
    iris.Constraint(pressure=500.),  # HadCM3 where pressure is in hPa
    iris.Constraint(air_pressure=50000.),  # CMIP6 where vertical var is air_pressure in Pa
    iris.Constraint(pressure=50000.)  # CMIP5 where vertical var is pressure in Pa

]

t500 = load_try(files, 'air_temperature', constraints)  ## 500 hPa temp

print("timeMean ", timeMean)
output_data = means(t500, name='TEMP@500', start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean)

rh500 = load_try(files, 'relative_humidity', constraints)  ## 500 hPa RH
output_data.extend(means(rh500, name='RH@500', start_time=start_time, end_time=end_time, timeMean=timeMean))

olr = iris.load_cube(files, 'toa_outgoing_longwave_flux', callback=fix_missing_data)  ## OLR
output_data.extend(means(olr, name='OLR', start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean))

rsr = iris.load_cube(files, 'toa_outgoing_shortwave_flux', callback=fix_missing_data)  ## rsr
output_data.extend(means(rsr, name='RSR', start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean))

insw = iris.load_cube(files, 'toa_incoming_shortwave_flux', callback=fix_missing_data)  ## insw
output_data.extend(
    means(insw, name='insw', start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean))
# clear sky is optional..

try:
    rsrc = iris.load_cube(files, 'toa_outgoing_shortwave_flux_assuming_clear_sky',
                          callback=fix_missing_data)  ## clear rsr
    output_data.extend(
        means(rsrc, name='RSRC', start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean))
except iris.exceptions.ConstraintMismatchError:
    pass

try:
    olrc = iris.load_cube(files, 'toa_outgoing_longwave_flux_assuming_clear_sky',
                          callback=fix_missing_data)  ## clear sky OLR
    output_data.extend(
        means(olr, name='OLRC', start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean))
except iris.exceptions.ConstraintMismatchError:
    pass

for cube in [insw, rsr, olr]:
    try:
        cube.remove_coord('forecast_period')  # remove forecast period co-ord from radn cubes
    except iris.exceptions.CoordinateNotFoundError:
        pass

# then do some arthmetic with them
net_flux = (insw - rsr) - olr

output_data.extend(
    means(net_flux, name='netflux', start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean))

## near sfc temperature north of 60S
constraints = [
    iris.Constraint(height=1.5) & constrain_60S,
    iris.Constraint(height=2.0) & constrain_60S,

]

try:
    sat = load_try(files, 'air_temperature', constraints)
except Exception:
    # now try looping over possible variable names...
    for var in ['near_surface_air_temperature', 'Near-Surface Air Temperature']:
        try:
            sat = iris.load_cube(files, var & constrain_60S, callback=fix_missing_data)
            break  # got the data
        except iris.exceptions.ConstraintMismatchError:
            pass

output_data.extend(means(sat, name='AT',
                         mask=land_mask_60S, mask_name='L', mask_attribute=mask_attr_60S,
                         start_time=start_time, end_time=end_time,
                         verbose=verbose, timeMean=timeMean))

precip = iris.load_cube(files, 'precipitation_flux' & constrain_60S,
                        callback=fix_missing_data)  ## Precipitation north of 60S

output_data.extend(means(precip, name='precip',
                         mask=land_mask_60S, mask_name='L', mask_attribute=mask_attr_60S,
                         start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean))

try:
    mslp = iris.load_cube(files, 'air_pressure_at_sea_level', callback=fix_missing_data)  ## MSLP -- HadAM3
except iris.exceptions.ConstraintMismatchError:
    mslp = iris.load_cube(files, 'air_pressure_at_mean_sea_level', callback=fix_missing_data)  ## MSLP -- CMIP6

mslp_mn = means(mslp, name='MSLP',
                start_time=start_time, end_time=end_time, verbose=verbose, timeMean=timeMean)

## remove the global-mean from the all values -- first we need to find it... 
gm = next(filter(lambda x: x.name() == u'MSLP_GLOBAL', mslp_mn))
# filter (in python 3) is an iterable. Only want the first one so use next
# to get it!
mslp_delta = copy.deepcopy(mslp_mn)  # deep copy the mean mslp as we modify delta
for i in range(len(mslp_delta)):  # iterate over indices here
    mslp_delta[i].data = mslp_mn[i].data - gm.data
    mslp_delta[i].rename(mslp_mn[i].name() + "_DGM")  # rename new version

## then keep only the NHX and TROPICS data
mslp_short = filter(lambda x: (x.name() == u'MSLP_NHX_DGM' or x.name() == u'MSLP_TROPICS_DGM'), mslp_delta)
output_data.extend(mslp_short)  # add it to the list to be appended

if (verbose):  # print out the summary data for all created values
    for cube in output_data:
        print(cube.name(), "=", cube.data)
        if (verbose > 1):
            print(cube)
            print("============================================================")

## now to write the data
iris.save(output_data, output_file)

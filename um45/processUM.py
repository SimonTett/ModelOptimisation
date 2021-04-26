#!/usr/bin/env python
"""
Run mkpph & ummonitor on files.
Then work out where it has got to and save that information
"""
# TODO add some meta-data to output. Life short so have not done so.
# Thing sto include are root dir, cachedirs, name of this code, data ran, input
# example run is  $OPTCLIMTOP/um45/processUM.py studies/coupJac/coupJac14p.json observations.nc


import argparse  # parse arguments
import functools
import json
import os  # OS support
import subprocess

import iris  # Met Office IRIS library
import iris.coord_categorisation
import iris.fileformats.pp as pp  
import iris.util
import netCDF4
import numpy as np  # numpy

import StudyConfig  # needed to parse json file.

maxYear = {}


# bunch of code from grl17.PaperLib
@functools.lru_cache(maxsize=5012)
def readPP(file, realization=None):
    """
    Read in pp data and add some aux co-ords.
    :param dir: name of directory relative to time-cache
    :param name:name of file
    :return: cube
    """
    try:
        cube = iris.load_cube(file) # use normal iris stuff
    except AttributeError:
        cube= read_pp(file) # use hacked pp read.
 
    # add a aux co - ord with the filename
    new_coord = iris.coords.AuxCoord(file, long_name='fileName', units='no_unit')
    cube.add_aux_coord(new_coord)

    addAuxCoord(cube, realization=realization)  # add aux co-ords

    return cube


def readFile(dir, name, realization=None):
    fullname = os.path.join(dir, name)
    pp = readPP(fullname, realization=realization)
    return pp


def addAuxCoord(cube, realization=None):
    """
    Add aux coords to a cube
    :param cube:
    :return:
    """

    for coord in ['time', 'latitude', 'longitude']:
        try:
            cube.coord(coord).guess_bounds()
        except (ValueError, iris.exceptions.CoordinateNotFoundError):
            pass

    # add auxilary information. year & month number
    iris.coord_categorisation.add_year(cube, 'time')
    iris.coord_categorisation.add_month_number(cube, 'time')
    # optionally add ensemble aux_coord -- from iris example doc..
    if realization is not None:
        realization_coord = iris.coords.Coord(realization, 'realization')
        cube.add_aux_coord(realization_coord)
        iris.util.new_axis(cube, scalar_coord='realization')

    return cube


def comp_net(direct):
    """
    Compute net flux from ummonitor cached data
    :param direct: directory where pp files live
    :return: net flux
    """

    # ts_olr = read_pp(os.path.join(direct, 'ts_rtoalwu.pp'))
    # ts_rsr = read_pp(os.path.join(direct, 'ts_rtoaswu.pp'))
    # ts_insw = read_pp(os.path.join(direct, 'ts_rtoaswd.pp'))

    ts_olr = readFile(direct, 'ts_rtoalwu.pp')
    ts_rsr = readFile(direct, 'ts_rtoaswu.pp')
    ts_insw = readFile(direct, 'ts_rtoaswd.pp')
    net = ts_insw - ts_olr - ts_rsr
    net.units = "W m^-2"
    net.long_name = 'Net Flux'
    net = iris.util.squeeze(net)
    return net

def pp_time(tspp, ycoord=True):
    """
    Generate time values for pp field
    :arg tspp timeseries
    """

    if ycoord:
        time = getattr(tspp, 'y', tspp.bzy + tspp.bdy * (1 + np.arange(tspp.data.shape[0])))
        if len(time) == 1 or np.all(time == 0.0):
            time = tspp.bzy + tspp.bdy * (1 + np.arange(tspp.data.shape[0]))
    else:
        time = getattr(tspp, 'x', tspp.bzx + tspp.bdx * (1 + np.arange(tspp.data.shape[1])))
        if len(time) == 1 or np.all(time == 0.0):
            time = tspp.bzx + tspp.bdx * (1 + np.arange(tspp.data.shape[1]))

    return time

def read_pp(file, realization=None):
    """
    Read a pp file  using iris pp library and return it as a cube
    Deals with (eventually) all the brokenness of iris...
    :param file: file to read data from
    :return: pp data  returned
    """

    if file is None: return None
    ts, = pp.load(file)  # extract the timeseries
    # ts=ts[0] # extract from list
    # set the time
    timeValues = (20, 21, 22, 23)
    if ts.lbcode.ix in timeValues:  # x is time-coord
        ts.x = pp_time(ts, ycoord=False)  # this is (I think) the real fix.
    if ts.lbcode.iy in timeValues:  # y is time-coord
        ts.y = pp_time(ts)

    # fix the co-ords if they don't exist
    try:
        x = ts.x
    except AttributeError:
        x = ts.bzx + (np.arange(ts.lbnpt) + 1) * ts.bdx
        ts.x = x

    try:
        y = ts.y
    except AttributeError:
        y = ts.bzy + (np.arange(ts.lbrow) + 1) * ts.bdy
        ts.y = y
    stuff = iris.fileformats.pp_load_rules.convert(ts)  # iris 2.0
    ts.data = np.ma.masked_equal(ts.data, ts.bmdi)
    # fix std name of
    cube = iris.cube.Cube(ts.data, standard_name=stuff.standard_name,
                          long_name=stuff.long_name,
                          # var_name=stuff.var_name,
                          units=stuff.units,
                          attributes=stuff.attributes, cell_methods=stuff.attributes,
                          dim_coords_and_dims=stuff.dim_coords_and_dims,
                          aux_coords_and_dims=stuff.aux_coords_and_dims)  # ,aux_factories=stuff.aux_factories)
    # all to here could be replaced with cube = iris.load_cube(file) though getting hold of the meta-data
    # might be tricky.
    cube.name(file)
    # add co-ords --
    for code, name in zip([5, 13], ['model_level', 'site_number']):
        try:
            if ts.lbcode.ix == code:
                level = iris.coords.DimCoord(ts.x, long_name=name)
                try:
                    cube.add_dim_coord(level, 1)
                except ValueError:
                    pass
        except AttributeError:
            pass
        try:
            if ts.lbcode.iy == code:
                level = iris.coords.DimCoord(ts.y, long_name=name)
                try:
                    cube.add_dim_coord(level, 0)
                except ValueError:
                    pass
        except AttributeError:
            pass


    return cube


parser = argparse.ArgumentParser(description=
                                 """Post process Unified Model to:
  1) Run mkphh on annual average atmospheric data
  2) Run ummonitor (via IDL) on annual average atmospheric data
  3) read in ts_sst.pp & ts_t15.pp from op? and ap? dirs 
    and work out max year
    -- NB this not really used. Subseqent calculations want timeseries. 
  Assume to be running in UM directory
  
  Example use is:
  
   processUM.py input.json outputFile 
  
  which will run makepph & ummonitor on the A/[ao]py dirs.
  """
                                 )
parser.add_argument("JSON_FILE", help="The Name of the JSON file")
parser.add_argument("OUTPUT", help="The Name of the output file which will be written to")
parser.add_argument("--dir", "-d",
                    help="Name of directory to run on. Cached ummonitor files are put here. Default cwd",
                    default=os.getcwd())

parser.add_argument("-s", "--skip", help='Skip data processing. Just read in data', action='store_true')
parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
args = parser.parse_args()  # and parse the arguments

# translation table to translate dir name into avg period string

translate = {'apy': '.000100', 'opy': '.000100', 'apx': '.001000', 'opx': '.001000',
             'aps': '.000030', 'ops': '.000030', 'apm': '.000001', 'opm': '.000001'}

rootDir = os.path.normpath(args.dir)
experName = os.path.basename(rootDir)  # really should work out from pp data but life is short

outputFile = os.path.join(rootDir, args.OUTPUT)
cacheroot = os.path.join(rootDir, 'A')  # where we want to put the data

config = StudyConfig.readConfig(args.JSON_FILE)  # config.
# dirs to process are 
dirs = config.getv('postProcess')['dirs']
gmConstraint = iris.Constraint(site_number=3.0)
cacheInfo = dict()
for mnDir in dirs:
    dir = os.path.join(rootDir, 'A', mnDir)  # full dir path
    if args.skip:  # skip data processing
        continue
    if args.verbose:
        print("Processing data in %s" % mnDir)
    # check dir exists
    if not (os.path.exists(dir) and os.path.isdir(dir)):
        continue 
    # metafunction needs to be set up
    
    metafile = os.path.join(dir, 'metafunction')
    with open(metafile, 'w') as file:
        file.write('pp_filename\n')

    output = subprocess.run(["makepph", dir])  # make the pph file.

    # meta test set up.
    # now to run MIDL. 
    text = """
    u=ss_assoc('%s')
    ummonitor,'*',u,/noplot,cachedir='%s'
    print,'Done making data'
    """ % (dir, cacheroot)

    # midl = subprocess.run('midl', stdout=subprocess.DEVNULL,
    #                       input=text, encoding='ascii',
    #                       stderr=subprocess.STDOUT)  # run midl
    midl = subprocess.run('midl', input=text, encoding='ascii')

    # end of generating data.

    # now read the data -- which is really to give some data to tell model
    # function that model has ran.
    dir = experName + translate[mnDir]
    cachedir = os.path.join(cacheroot, dir)
    file = 'ts_t15.pp'
    data = readFile(cachedir, file).extract(gmConstraint)
    data.coord('year').guess_bounds()
    maxYear = float(np.max(data.coord('year').bounds)  )
    cacheInfo[cachedir] = maxYear
    if args.verbose:
        print("file %s maxYr %5.1f " % (file, maxYear))

# now processed all data write information to JSON file.
with open(outputFile, 'w') as outfile:
    json.dump(cacheInfo, outfile)

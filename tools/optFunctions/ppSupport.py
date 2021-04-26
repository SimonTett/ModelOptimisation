"""
Mini module to provide some additional support for reading pp data.
"""
import functools
import glob
import os
import subprocess
import operator 

import iris
import iris.coord_categorisation
import iris.coords
import iris.exceptions
import iris.fileformats
import iris.fileformats.pp as pp
import iris.util
import numpy as np



def readPP(file, realization=None):
    """
    Read in pp data and add some aux co-ords.
    :param dir: name of directory relative to time-cache
    :param name:name of file
    :return: cube
    """
    # need str as iris does not handle filepath
    try:
        cube = iris.load_cube(str(file))  # use normal iris stuff
    except AttributeError:
        cube = read_pp(str(file))  # use hacked pp read.

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
    ts, = pp.load(str(file))  # extract the timeseries
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


def ummonitorDir(dir, choices=['*'], verbose=False):
    """
    run makepph on directory and then run IDL ummonitor code.
    Requires that makepph, midl in path and ummonitor etc in idl search path.
    :param dir --- a pathLib path for which makepph and ummonitor to be run on
    :param choices -- a list of choices (see ummonitor). Default will make everything.
    :return: *name* of directory (not full path) where ummonitor put the data and
       maximum year of first ts_*.pp file found.
    """

    # translation table to translate dir name into avg period string

    translate = {'apy': '.000100', 'opy': '.000100', 'apx': '.001000', 'opx': '.001000',
                 'aps': '.000003', 'ops': '.000003', 'apm': '.000001', 'opm': '.000001'}
    file_sort_fn = lambda file: file.stat().st_mtime_ns # time in ns for file used for sorting
    if verbose:
        print("Processing data in %s" % dir)

    # Work out if need to make pph file.
    files = sorted(dir.iterdir(),key = file_sort_fn,reverse=True)
    # all files in  reverse time order. (newest to oldest)
    # remove .used if it exists (which is a pain!)
    usedF = dir/ '.used'
    if usedF in files: files.remove(usedF)
    if files[0].name != 'pph': # need to make pph file
        # now make the pph file (and with the -M option the metafunction file)
        output = subprocess.run(["makepph", "-M",dir])  # make the pph file.
        if verbose: print(f"Made pph file for {dir}")

    # now to run ummonitor using MIDL.
    text = f"u=ss_assoc('{dir}')\n"
    try:
        for c in set(choices): # only want unique elements.
            text += f"ummonitor,'{c}',u,/noplot,cachedir='{dir.parent}'\n"
    except TypeError:
        print(f"Nothing to do for {dir}")

    text += f"print,'Done making data for {dir} '\n"
    midl = subprocess.run('midl', input=text, encoding='ascii')
    # end of generating data.

    # now read the data -- which is really to give some data to tell model
    # function that model has ran.
    files = sorted(dir.parent.glob('*'+translate[dir.name]),key =file_sort_fn )
    print("Files are ",files)
    cachedir = files[-1]
    file = sorted(cachedir.glob('ts*.pp'),key=file_sort_fn)[-1] # get newest timeseries file
    data = readPP(file)  # read the first file we find
    if verbose:
        print(f"Read data from {file}")

    try:  # attempt to give data some year bounds.
        data.coord('year').guess_bounds()
        maxYear = float(np.max(data.coord('year').bounds))
    except ValueError:  # points should be monotonically increasing.
        # for month & season this is not the case so just get the max value.
        maxYear = float(np.max(data.coord('year').points))

    if verbose:
        print("file %s maxYr %5.1f " % (file, maxYear))

    return cachedir.name, maxYear

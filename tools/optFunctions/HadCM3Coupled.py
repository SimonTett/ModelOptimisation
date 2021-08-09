"""
Module to support running coupled models to estimate Equilibrium solutions.
 and computing sea-ice.
TOCONSIDER: Convert these into methods -- they could inherit from Submit (which gets renamed). Then would not need to
  pass MODELRUN in.
TODO: Find some way of stuffing values back into model.
TODO: Now have continue handled automatically this may no longer be needed.
"""
import functools
import os
import tempfile
import unittest

import iris
import iris.analysis
import iris.coord_categorisation
import iris.coords
import iris.exceptions
import iris.fileformats.pp as pp
import iris.util
import numpy as np
import numpy.testing as npt
import pandas as pd

import HadCM3
import StudyConfig
import Submit
import optClimLib



def comp_fit(cubeIn, order=2, year=np.array([109, 179]), x=None, bootstrap=None, timeAvg=None, mask=False,
             fit=False, makeCube=False):
    """
    Fit Nth (Default is 2) order fn to time co-ordinate of cube then compute value at specified year
    :param cube: cube to do fitting to.
    :param order  (default 2 )-- the order to fit too
    :param year (default [111, 181]) -- the years to get values for.
    :param x (detault is None) -- if specified regress against this rather than time.
    :param bootstrap (default is None) -- if specified computes bootstrap uncertainties -- assuming guaussin
    :param timeAvg (default is None) -- if specified time-average data with this period.
    :param mask (default is False) -- if True mask data..
    :param fit (default is False) -- if True return fit params
    :param makeCube (default for now if False) -- if True wrap the result as a cube.
    :return: the fitted values at the specified year
    """

    cube = cubeIn
    if timeAvg is not None:
        cube = cubeIn.rolling_window('time', iris.analysis.MEAN, timeAvg)
        # later extract the times in the mid point of the windows..

    # need to figure out which co-ord is time..
    coordNames = [c.standard_name for c in cube.coords()]
    timeAxis = coordNames.index('time')  # work out which axis is time.

    time_points = cube.coord('year').core_points()
    data = cube.data
    if cube.ndim == 1:
        data = data.reshape((len(data), 1))
    timeAxis = cube.coord_dims('time')  # work out which axis is time.
    if timeAxis[0] != 0:  # always want time first...not sure why sometimes a tuple..
        data = np.moveaxis(data, timeAxis, 0)
    shape = data.shape
    npt = np.product(shape[1:])
    data = data.reshape(shape[0], npt)
    if x is None:
        xx = time_points
    else:
        xx = x

    if timeAvg is not None:
        indx = np.arange(0, len(data) + 1, timeAvg)
        xx = xx[indx]
        data = data[indx, :]
    try:
        # watch out when not enough values. Need at least order+1 points.
        try:
            cnt = data.count(0)  # sum all non missing data,
        except AttributeError:
            cnt = np.repeat(shape[0], npt)
        L = cnt > order  # logical where we have enough pts.
        pmsk = np.ma.polyfit(xx, data[:, L], order)  # result on masked subset
        shp = (pmsk.shape[0], data.shape[1])  # shape of result array
        p = np.repeat(np.nan, np.product(shp)).reshape(shp)  # array with ara
        p[:, L] = pmsk
        p = np.ma.masked_invalid(p)
    except ValueError:  # polyfit failed likely because of NaN
        return [None] * order

    if fit:
        return p
    # this fails if year is an int.

    year2 = year
    if len(p.shape) > 1: year2 = np.reshape(year, (np.size(year), 1))
    result = np.polyval(p, year2).squeeze()  # compute values from fit at desired years -- note could extrapolate.
    # reform result..
    rShape = [np.size(year)]
    rShape.extend(shape[1:])
    result = result.reshape(rShape)
    # now mask result
    if mask:
        msk0 = data.mask[0, :].copy()  # mask for first column...
        result = np.ma.masked_array(result, mask=np.broadcast_to(msk0, result.shape))
    # now compute bootstrap...if wanted
    if bootstrap is not None:
        bsValues = []
        for i in range(0, bootstrap):
            npt = len(xx)
            indx = np.random.choice(npt, npt)
            p = np.polyfit(xx[indx], data[indx, :], order)
            bsValues.append(np.polyval(p, year))
        arr = np.array(bsValues)
        var = np.var(arr, 0)
        result = (result, np.sqrt(var))  # append var to the result

    if makeCube:  # wrap data as a cube.
        # easiest approach is to select required years then overwrite the data..
        # won't work if years don't actually exist.
        # should figure out how to create an iris cube from another one.
        tempCube = cube.extract(iris.Constraint(year=year))
        tempCube.data = result.squeeze()
        result = tempCube
    return result


def compTCR(force, ctl, file, scale=1, year=111):
    """
    Compute transient values
    :param force: dir path to forced simulation
    :param ctl: dir path to control simulation
    :param file: name of time_Cache file
    :param scale (optional): scaling to apply -- default 1.
    :return: estimated values at specified year
    """

    delta = readDelta(force, ctl, file)
    if delta is None:
        return [None]*len(year)  # no data so return None x no of years
    delta = delta[:, -1]  # just want the global-mean value.
    transient = comp_fit(delta, 2, year=year) * scale
    return transient.squeeze()


def compCtl(ctl,file,scale=1,year=181):
    """
    Compute Ctl value at specified year 
    :param ctl: dir path to control simulation
    :param file: name of time_Cache file
    :param scale (optional): scaling to apply -- default 1.
    :param year (optional): year at which to compute -- default is 181.
    :return: estimated values at specified year 
    """
    if file == 'netflux':
        ctlD= readNetFlux(ctl)
    else:
        ctlD = readFile(ctl, file)
    if ctlD is None:
        return [None]*len(year)  # no data so return None x no of years
    ctlD = ctlD[:, -1]  # just want the global-mean value.
    estValue = comp_fit(ctlD, 2, year=year) * scale
    return estValue.squeeze()

    

def compEq(delta, netFlux):
    """
    Compute equilibrium value from linear regression on net flux.
    :param delta:
    :param netFlux:
    :return: equilibrium value(s)
    """
    netFluxData = netFlux.data.squeeze()
    deltaData = np.squeeze(delta.data)
    if deltaData.ndim == 1:  # one-d -- make it two d.
        deltaData = np.reshape(deltaData, (-1, 1))
    npts = deltaData.shape[1]
    eqValue = np.zeros(npts)
    for k in range(0, npts):
        reg = np.polyfit(deltaData[:, k], netFluxData, 1)
        roots = np.roots(reg)
        eqValue[k] = roots[np.isreal(roots)][0]  # find first **real** root.

    try:
        eqValue = np.asscalar(eqValue)  # if a single value want to return as a scalar value.
    except ValueError:  # not a single value so just squeeze it.
        eqValue = eqValue.squeeze()
    
    return eqValue


## lots of code to support reading in PP timeseries.
@functools.lru_cache(maxsize=5012)
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
    # try: # sometimes masking breaks things..
    #     ts.data = np.ma.masked_equal(ts.data, ts.bmdi)
    # except  ValueError:
    #     pass
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

    # add auxiliary information for year
    for fn in [iris.coord_categorisation.add_year]:  # ,iris.coord_categorisation.add_month):
        try:
            fn(cube, 'time')
        except ValueError:
            pass
    # optionally add ensemble aux_coord -- from iris example doc..
    if realization is not None:
        realization_coord = iris.coords.Coord(realization, 'realization')
        try:
            cube.add_aux_coord(realization_coord)
            iris.util.new_axis(cube, scalar_coord='realization')
        except ValueError:
            pass

    return cube


def readFile(dir, name, realization=None):
    fullname = os.path.join(dir, name + '.nc')
    if not os.path.exists(fullname):
        fullname = os.path.join(dir, name + '.pp')
    #print(f"Reading in data from {dir} {name}")
    with np.errstate(divide='ignore', invalid='ignore'):  #
        # get Nans in some reads from iris pp conversion. The with means we ignore them!
        try:
            cube = iris.load_cube(fullname)
        except AttributeError:  # damm iris fails withs some PP data
            cube = read_pp(fullname)

    addAuxCoord(cube, realization=realization)
    x=cube.data[-1] # needed to make iris load data which stops some error. FIXME: Understand why needed
    return cube


def readDelta(dir1, dir2, file):
    """
    Compute difference between file in dir1 from file in dir2
    :param dir1: Directory (as in readFile) where first data is
    :param dir2: Directory (as in readFile) where second data is
    :param file:File being read in
    :return: difference between two datasets.
    """
    if file == 'netflux':
        pp1 = readNetFlux(dir1)
        pp2 = readNetFlux(dir2)
    else:
        pp1 = readFile(dir1, file)
        pp2 = readFile(dir2, file)

    delta = comp_delta(pp1, pp2)
    return (delta)


def comp_delta(cube1, cube2):
    """
    Compute difference between two cubes. cube2 will be interpolated to same times as cube1
    :param cube1:  first cube
    :param cube2: 2nd cube
    :return: difference after interpolating cube2 to same times as cube1.
    """
    # check min and max times of cube2 are consistent with cube1
    if cube2.coord('time').bounds.min() > cube1.coord('time').bounds.min():
        # print("Extrapolating below min time -- returning None for ", cube2.name())
        raise Exception("Extrapolating below min c2= %5.1f c1=%5.1f" %
                        (cube2.coord('time').bounds.min(), cube1.coord('time').bounds.min()))

    if cube2.coord('time').bounds.max() < cube1.coord('time').bounds.max():
        # print("Extrapolating above max time -- returning None for", cube2.name())
        raise Exception("Extrapolating above max")

    interp = cube2.interpolate([('time', cube1.coord('time').points)], iris.analysis.Linear())
    try:
        diff = (cube1 - interp)
    except ValueError:
        diff = cube1.copy()
        diff.data = cube1.data - interp.data
        # print "Fix ",exper.name,cube1.name()

    diff.units = cube1.units  # quite why this needs to be done I don't know why..
    diff.long_name = cube2.long_name
    return diff


def readNetFlux(direct):
    """
    Compute net flux from ummonitor cached data
    :param direct: directory where pp files live
    :return: net flux
    """

    # ts_olr = read_pp(os.path.join(direct, 'ts_rtoalwu.pp'))
    # ts_rsr = read_pp(os.path.join(direct, 'ts_rtoaswu.pp'))
    # ts_insw = read_pp(os.path.join(direct, 'ts_rtoaswd.pp'))

    ts_olr = readFile(direct, 'ts_rtoalwu')
    ts_rsr = readFile(direct, 'ts_rtoaswu')
    ts_insw = readFile(direct, 'ts_rtoaswd')
    net = ts_insw - ts_olr - ts_rsr
    net.units = "W m^-2"
    net.long_name = 'Net Flux'
    return net


## end of code for reading in data.

def annDir(model):
    """
    work out the name of the annual directory and return it & any values (year) associated with it
    :param model: a model like object for which the directories are searched.
    :return:
    """
    obs = model.getObs(justRead=True)
    if obs is None:
        # not actually run anything so raise appropriate error
        raise Submit.runModelError
    kAnn = [k for k in obs.keys() if '.000100' in k]
    if len(kAnn) != 1:
        raise ValueError("Got %i keys", len(kAnn))
    ctlDir = kAnn[0]
    o = obs[ctlDir]  # the obs
    dir = ctlDir
    ## to turn of evil hack uncomment below
    # return (dir,o)
    # TEMP evil hack -- convert obsDir to relative path.
    dir = os.path.dirname(model._configFilePath)  # dir path to config.
    # want last two paths...
    dirs = []
    for indx in range(0, 2):
        ctlDir, base = os.path.split(ctlDir)
        dirs.append(base)

    dir = os.path.join(dir, *tuple(dirs[::-1]))
    return (dir, o)


def monDir(model):
    """
    work out the name of the monthly-mean directory and return it & any values (year) associated with it
    :param model: a model like object for which the directories are searched.
    :return: the name of the directory and the
    """
    obs = model.getObs(justRead=True)
    if obs is None:
        # not actually run anything so raise appropriate error
        raise Submit.runModelError
    kMon = [k for k in obs.keys() if '.000001' in k]
    if len(kMon) != 1:
        raise ValueError("Got %i keys", len(kMon))
    ctlDir = kMon[0]
    o = obs[ctlDir]  # the obs
    dir = ctlDir
    ## to turn of evil hack uncomment below
    # return (dir,o)
    # TEMP evil hack -- convert obsDir to relative path.
    dir = os.path.dirname(model._configFilePath)  # dir path to config.
    # want last two elements in path paths...
    dirs = []
    for indx in range(0, 2):
        ctlDir, base = os.path.split(ctlDir)
        dirs.append(base)

    dir = os.path.join(dir, *tuple(dirs[::-1]))
    return (dir, o)

def EQ4(params, ensembleMember=None,  MODELRUN=None, *args, **kwargs):
    """
    Compute Equilibrium values  for 4xCO2 -- locked to HadCM3 (for the moment).
    Code, currently, is really example which could, eventually, be generalised
    Code aimed to be used for computing Jacobian.
    Code tricky as dependencies between running ctl and 4xCo2. Then need to read data..
    It could in principle be used to find parameters that produce high (or low) climate sensitivities...
    :param: params -- a numpy array with the parameter values.
    :param: ensembleMember -- ensemble member for this case.
    :param failNan -- fail if any nans in result. (default is True)
    :param args: -- positional arguments
    :param kwargs: -- keyword arguments
    :return: Value of ECS4xCO2
    """
    if MODELRUN is None:  # check have something for modelrun.
        raise Exception("Supply value for MODELRUN")

    rootDir = os.path.join(os.environ["OPTCLIMTOP"], 'Configurations')
    HadCM3ctl = os.path.join(rootDir, 'xnmea')  # configuration for HadCM3 ctl.
    # TODO make this part of the configuration.
    HadCM34xCO2 = os.path.join(rootDir, "xnmed")  # configuration for HadCM3 4xCO2.
    # TODO make this part of the configuration.
    paramNames = MODELRUN.paramNames()
    nObs = len(MODELRUN.obsNames())
    fileLookup = {
        'sat': 'ts_t15',
        'precip_land': 'ts_tppn_land',
        'precip': 'ts_tppn'
    }
    # lookup table that translates obs name into file to read.

    # params can be a 2D array...
    if params.ndim == 1:
        use_params = params.reshape(1, -1)
    elif params.ndim == 2:
        use_params = params
    else:
        raise Exception("params should be 1 or 2d ")

    nsim = use_params.shape[0]
    nparams = use_params.shape[1]
    if nparams != len(paramNames):
        raise Exception(
            "No of parameters %i not consistent with no of varying parameters %i\n" %
            (nparams, len(paramNames)) +
            "Params: " + repr(use_params) + "\n paramNames: " + repr(paramNames))

    Eq4xCO2 = np.full((nsim, nObs), np.nan)  # array of np.nan for equilibrium soln.
    # loop over param array.
    for indx in range(0, nsim):  # iterate over the simulations.
        pDict = dict(zip(paramNames, use_params[indx, :]))  # create dict with names and values.
        if ensembleMember is not None:
            pDict.update(ensembleMember=ensembleMember)

        # step 1 -- create a ctl run.
        pDict.update(refDir=HadCM3ctl)
        ctl = MODELRUN.model(pDict, update=True)

        if ctl is not None:  #
            # find annual mean directory
            (ctlDir, ctlYr) = annDir(ctl)
            if ctlYr > 40:
                # reached year 40 -- so have a dump for 4xCO2
                # need to do something different with this...
                # Work out start files and set them.
                ASTART = os.path.join(ctl.dirPath, 'A', 'dumps', ctl.name() + "a@da041c1")
                OSTART = os.path.join(ctl.dirPath, 'A', 'dumps', ctl.name() + "o@da041c1")
                pDict.update(ASTART=ASTART, OSTART=OSTART, refDir=HadCM34xCO2)  # params for 4xCO2
                CO2x4 = MODELRUN.model(pDict, update=True)  # generate 4xCO2.
                if ctlYr < 81:  # not yet finished.. modify existing model..
                    ctl.continueSimulation()  # update the control.
                    MODELRUN.rerunModel(ctl)  # push modified ctl on list to be reran
                elif CO2x4 is not None:  # can now data process!
                    (CO2x4dir, CO2x4yr) = annDir(CO2x4)
                    # lets process the data
                    deltaNet = readDelta(CO2x4dir, ctlDir, 'netflux')
                    obsNames = MODELRUN.obsNames()
                    for oindx, obs in enumerate(obsNames):
                        file = fileLookup[obs]  # allow error to be triggered if not present.
                        delta = readDelta(CO2x4dir, ctlDir, file)  # difference between two timeseries.
                        Eq4xCO2[indx, oindx] = compEq(delta, deltaNet)[-1]  # just want global mean.
                    # add info on EQ4 to both models....A bit of a hack...

                    s = pd.Series(Eq4xCO2[indx, :], index=obsNames).rename(ctl.name())
                    MODELRUN.paramObs(pd.Series(pDict), s)  # store the params and obs.

                else:
                    pass
                    # nothing to do...
        # dealt with one case
    # finished iterating over parameter cases.
    return np.squeeze(Eq4xCO2)  # return the data..


def TCR(params, ensembleMember=None,  MODELRUN=None, *args, **kwargs):
    """
    Compute Transient Climate Response  at 4 & 2xCO2 -- locked to HadCM3 (for the moment).
    Code, currently, is really example which could, eventually, be generalised
    Code aimed to be used for computing Jacobian.
    Code tricky as dependencies between running ctl and 1%. Then need to read data..
    It could in principle be used to find parameters that produce high (or low) climate sensitivities...
    :param: params -- a numpy array with the parameter values.
    :param: ensembleMember -- ensemble member for this case.
    :param failNan -- fail if any nans in result. (default is True)
    :param args: -- positional arguments
    :param kwargs: -- keyword arguments
    :return: 2nd order fit at year 110 & yr 180 = 2xCO2 & 4xCO2 respectively
    """
    import re
    if MODELRUN is None:  # check have something for modelrun.
        raise Exception("Supply value for MODELRUN")

    rootDir = os.path.join(os.environ["OPTCLIMTOP"], 'Configurations')
    HadCM3ctl = os.path.join(rootDir, 'xnmea')  # configuration for HadCM3 ctl.
    # TODO make this part of the configuration.
    HadCM3OnePercent = os.path.join(rootDir, "xnmeb")  # configuration for HadCM3 1% run.
    # TODO make this part of the configuration.
    paramNames = MODELRUN.paramNames()

    fileLookup = {
        'sat': 'ts_t15',
        'sat_land': 'ts_t15_land',
        'precip_land': 'ts_tppn_land',
        'precip': 'ts_tppn',
        'ice_extent': 'ts_extent',
        'ocean_temp':'ts_ot',
        'nao':'ts_nao',
        'cet':'ts_cet'
    }
    # lookup table that translates obs name into file to read.
    obsNames = MODELRUN.obsNames()
    
    # params can be a 2D array...
    if params.ndim == 1:
        use_params = params.reshape(1, -1)
    elif params.ndim == 2:
        use_params = params
    else:
        raise Exception("params should be 1 or 2d ")

    nsim = use_params.shape[0]
    nparams = use_params.shape[1]
    if nparams != len(paramNames):
        raise Exception(
            "No of parameters %i not consistent with no of varying parameters %i\n" %
            (nparams, len(paramNames)) +
            "Params: " + repr(use_params) + "\n paramNames: " + repr(paramNames))

    TCR = np.full((nsim, len(obsNames)), np.nan)  # array of np.nan for TCR values soln.
    # loop over param array.
    for indx in range(0, nsim):  # iterate over the simulations.
        pDict = dict(zip(paramNames, use_params[indx, :]))  # create dict with names and values.
        if ensembleMember is not None:
            pDict.update(ensembleMember=ensembleMember)
        pDict1per = pDict.copy()
        # step 1 -- create a ctl run.
        pDict.update(refDir=HadCM3ctl)
        ctl = MODELRUN.model(pDict, update=True)
        if ctl is None:
            continue  # not actually ran but will have a bunch of simulations to run!
        # find annual mean directory
        (ctlDir, ctlYr) = annDir(ctl)
        if ctlYr < 181:
            # not yet finished.. so continue ctl.
            ctl.continueSimulation()  # update the control.
            MODELRUN.rerunModel(ctl)  # push modified ctl on list to be reran
        if ctlYr > 40:
            # reached year 40 -- so now have a dump for 1%
            # Work out start files and set them.
            ASTART = os.path.join(ctl.dirPath, 'A', 'dumps', ctl.name() + "a@da041c1")
            OSTART = os.path.join(ctl.dirPath, 'A', 'dumps', ctl.name() + "o@da041c1")
            pDict1per.update(ASTART=ASTART, OSTART=OSTART, refDir=HadCM3OnePercent)  # params for 1%
            CO2Oneper = MODELRUN.model(pDict1per, update=True)  # generate 1% run
            if CO2Oneper is None:
                print("Started 1% with ctl year = ", ctlYr)
                continue  # not actually ran but will have a bunch of simulations to run!
            # continue the run if necessary
            (CO2OnePerDir, CO2OnePerYr) = annDir(CO2Oneper)
            if CO2OnePerYr < 181:  # started running but not yet finished so continue the run
                CO2Oneper.continueSimulation()  # update the 1% run.
                MODELRUN.rerunModel(CO2Oneper)  # push modified 1% run on list to be reran

            if (CO2OnePerYr > 180) and (ctlYr > 180):
                # both ctl & 1% are finished so process the data
                for oindx, obs in enumerate(obsNames):
                    # remove the TCR/CTL part from the name so can find the file.
                    if re.search('_TCR.*$',obs): # TCR cases
                        file = fileLookup[re.sub('_TCR.*$', '', obs)]
                        # work out year we want -- basically last bit of the TCR tells us this.
                        if re.match('.*_TCR4$',obs):
                            year=179 # 4xCO2 year
                        elif re.match('.*_TCR$',obs):
                            year=109 # 2xCO2 year
                        else:
                            raise Exception("Unknown pattern for %s"%(obs))

                        TCR[indx,oindx] = compTCR(CO2OnePerDir, ctlDir, file,year = year) 
                    elif re.search('_CTL$',obs): # CTL cases
                        file = fileLookup[re.sub('_CTL$', '', obs)]
                        TCR[indx,oindx] = compCtl(ctlDir, file,year = 179) 
                    else:
                        raise Exception("Unknown pattern for %s"%(obs))

                s = pd.Series(TCR[indx, :], index=obsNames).rename(ctl.name())
                MODELRUN.paramObs(pd.Series(pDict), s)  # store the params and obs.
        # dealt with one case
    # finished iterating over parameter cases.


    return np.squeeze(TCR)  # return the data..

def shortNHiceCtl(params, ensembleMember=None,  MODELRUN=None, *args, **kwargs):
    """
    Compute NH ice min & max values from a short ctl.
    :param: params -- a numpy array with the parameter values.
    :param: ensembleMember -- ensemble member for this case.
    :param args: -- positional arguments
    :param kwargs: -- keyword arguments
    :return: NH 10 year average min and max ice extent.
    """
    if MODELRUN is None:  # check have something for modelrun.
        raise Exception("Supply value for MODELRUN")



def fakeTCR(model, studyCfg, verbose=False):
    """
    fake TCR results
      just read in dta from reference experiment and write it out for appropriate period. Scale a bit depending on
      parameter
    :return:
    """

    umMonOut = {
        'xnmea': 'xhivd',  # control
        'xnmeb': 'xjgbb'  # 1% run.
    }
    files = ['ts_sst', 'ts_t15', 'ts_t15_land', 'ts_tppn', 'ts_tppn_land']
    name = os.path.basename(model.refDirPath())
    monDir = os.path.join(os.environ['OPTCLIMTOP'], 'Configurations', 'time_cache', umMonOut[name])
    param = model.getParams(series=True)  # read in the params
    obs = model.getObs(justRead=True, verbose=False)  # read obs
    # print("Obs is ",model.name(),obs)
    if (obs is None) or (len(obs) == 0):  # no obs so make some.
        obs = {}

        startYr = model.readNameList(['START_TIME'])['START_TIME'][0]
        # for extension in ['.000100', '.001000']:
        for extension in ['.000100']:  # only need annual mean data.
            dirName = os.path.join(model.dirPath, 'A', model.name() + extension)
            obs[dirName] = startYr

    # end dealing with no obs.
    for dirName, yr in obs.items():
        obs[dirName] += 40  # running in 40 year chunks.
        tgtYear = obs[dirName]
        try:  # create dir where fake ummonitor files going.
            os.makedirs(dirName, exist_ok=True)
        except FileExistsError:  # already exists so keep going..
            pass
        refDir = monDir + dirName[-7:]
        for file in files:  # copy all the files across
            # need to read in file and perturb it.
            # Assume everything on VF1 and increase values by 1+(VF1-1)
            ppFld = readFile(refDir, file)
            selFn = lambda yr: yr <= tgtYear
            ppFld = ppFld.extract(iris.Constraint(year=selFn))  # extract to desired year
            if file not in ['ts_rtoalwu', 'ts_rtoaswd', 'ts_rtoaswu']:
                scale = (1 + (param.VF1 - 1))
                # print("pp file is ", file, refDir, dirName, ppFld.data.shape)
                ppFld *= scale
            iris.save(ppFld, os.path.join(dirName, file + '.nc'))  # save it
    # done processing all fake ummonitor files
    # print("Writing obs ",model.name(),obs)
    model.writeObs(obs)  # and write data out!
    # TODO something about reading in pp data nad doing various things to it then writing it out again
    # means it cannot be read in first time...
    return None

def fakeEQ4(model, studyCfg, verbose=False):
    """
    fake ECS4 results
    :return:
    """

    umMonOut = {
        'xnmea': 'xhivd',  # control
        'xnmed': 'xnilb'  # 4xCO2
    }
    files = ['ts_sst', 'ts_t15', 'ts_t15_land', 'ts_tppn', 'ts_tppn_land',
             'ts_rtoalwu', 'ts_rtoaswd', 'ts_rtoaswu'  # needed for net flux computations.
             ]
    name = os.path.basename(model.refDirPath())
    monDir = os.path.join(os.environ['OPTCLIMTOP'], 'Configurations', 'time_cache', umMonOut[name])
    param = model.getParams(series=True)  # read in the params
    obs = model.getObs(justRead=True, verbose=False)  # read obs
    if (obs is None) or (len(obs) == 0):  # no obs so make some.
        obs = {}

        startYr = model.readNameList(['START_TIME'])['START_TIME'][0]
        for extension in ['.000100', '.001000']:
            dirName = os.path.join(model.dirPath, 'A', model.name() + extension)
            obs[dirName] = startYr

    # end dealing with no obs.
    for dirName, yr in obs.items():
        obs[dirName] += 40  # running in 40 year chunks.
        tgtYear = obs[dirName]
        try:  # create dir where fake ummonitor files going.
            os.makedirs(dirName, exist_ok=True)
        except FileExistsError:  # already exists so keep going..
            pass
        refDir = monDir + dirName[-7:]
        for file in files:  # copy all the files across
            # need to read in file and perturb it.
            # Assume everything on VF1 and increase values by 1+(VF1-1)
            ppFld = readFile(refDir, file)
            selFn = lambda yr: yr <= tgtYear
            ppFld = ppFld.extract(iris.Constraint(year=selFn))  # extract to desired year
            if file not in ['ts_rtoalwu', 'ts_rtoaswd', 'ts_rtoaswu']:
                scale = (1 + (param.VF1 - 1))
                # print("pp file is ", file, refDir, dirName, ppFld.data.shape)
                ppFld *= scale
            iris.save(ppFld, os.path.join(dirName, file + '.nc'))  # save it
    # done processing all fake ummonitor files
    model.writeObs(obs)  # and write data out!
    # TODO something about reading in pp data nad doing various things to it then writing it out again
    # means it cannot be read in first time...



# modify the lookup tables in config.
#config.optFunctions.update(HadCM3ECS4=EQ4)
#config.fakeFunctions.update(HadCM3ECS4=fakeEQ4)

#config.optFunctions.update(HadCM3TCR=TCR)
#config.fakeFunctions.update(HadCM3TCR=fakeTCR)


## test code. Included in module as only two routines to be tested.


class testHadCM3(unittest.TestCase):
    """
    Test cases for HadCM3Coupled. Tests fakeTCR

    """

    def checkDir(self, dirname, tgtYear, rootname='jc00'):
        """
        Convenience fn to check dir as expected
        :param dirname: naem of directory to check
        :return:
        """

        dirs = []  # list of model dirs to run.
        with os.scandir(dirname) as dirIter:
            dirContents = [entry for entry in dirIter]
        dirContents.sort(key=lambda entry: entry.path)
        for entry in dirContents:
            if entry.is_dir():
                mdir = entry.path
                dirs.append(mdir)
                self.assertEqual(entry.name, rootname + str(len(dirs)))
                # read it in as a model and then get obs
                m = HadCM3.HadCM3(mdir)
                obs = m.getObs(justRead=True)
                # verify that obs year is 41
                for k, v in obs.items():
                    self.assertEqual(v, tgtYear, 'Failed for dir %s got year %i expected %i' % (k, v, tgtYear))

        return dirs



    ################################
    # test code.
    def setUp(self):
        """
        Standard setup for all test cases
        :return:
        """
        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = self.tmpDir.name
        self.testDir = testDir

        rootDir = os.path.join(os.environ["OPTCLIMTOP"], 'Configurations')
        self.rootDir = rootDir

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        optClimLib.delDirContents(self.testDir)  # remove all files and dir!

    def test_rwRadn(self):
        """
        iris screws up in some way when it reads then writes toa fluxes...

        :return:
        """
        monDir = os.path.join(os.environ['OPTCLIMTOP'], 'Configurations',
                              'time_cache', 'xhivd') + '.000100'
        fileName = os.path.join(monDir, 'ts_rtoalwu.pp')
        # fileName = os.path.join(monDir, 'ts_sst.pp')
        outFile = os.path.join('c://', 'users', 'stett2', 'tmp.nc')
        ts = readFile(monDir, 'ts_rtoalwu')
        iris.save(ts, outFile)
        ts2 = iris.load_cube(outFile)

    def test_fakeEQ4(self):
        """
        Test that running fake works
        :return:
        """
        configFile = os.path.join(os.environ['OPTCLIMTOP'], 'Configurations', 'example_coupledJac.json')
        self.Config = StudyConfig.readConfig(configFile)
        HadCM3ctl = os.path.join(self.rootDir, 'xnmea')  # configuration for HadCM3 ctl.
        HadCM34xCO2 = os.path.join(self.rootDir, "xnmed")  # configuration for HadCM3 4xCO2.
        # make 4 models...
        nameRoot = 'dir0'
        models = []
        paramsBase = {'VF1': 1.0, 'RHCRIT': 2.0}
        for refDir in [HadCM3ctl, HadCM34xCO2]:
            for p, v in zip(['VF1', 'RHCRIT'], [1.1, 3.0]):
                params = paramsBase.copy()
                params[p] = v
                dirname = nameRoot + str(len(models))
                model = HadCM3.HadCM3(os.path.join(self.testDir, dirname), create=True,
                                      refDirPath=refDir, name=dirname, ppOutputFile='observations.json',
                                      parameters=params, verbose=False)
                models.append(model)
        # run on each model.
        for m in models:
            fakeEQ4(m, self.Config, verbose=False)

        # now check values are as expected.

        for m in models:
            obs = m.getObs()
            refDir = m.refDirPath()
            expectYr = 41
            if 'xnmed' in refDir:
                expectYr = 81

            for k, v in obs.items():
                self.assertEqual(v, expectYr, msg='Expected %i for %s got %i' % (expectYr, k, v))
                path = os.path.join(m.dirPath, 'A', m.name() + '.000100')
                self.assertTrue(os.path.isdir(path), msg='Failed to find %s' % (path))
                # and verify key gives directory too..
                self.assertTrue(os.path.isdir(k), msg='Failed to find %s' % (path))

    # end of check dir
    def test_EQ4(self):
        """
        Test that submit method (with fake fn ) works.
        :return:
        """
        configFile = os.path.join(os.environ['OPTCLIMTOP'], 'Configurations', 'example_coupledJac.json')
        self.Config = StudyConfig.readConfig(configFile)
        MODELRUN = Submit.ModelSubmit(self.Config, HadCM3.HadCM3, None, EQ4,
                                      fakeFn=fakeEQ4, rootDir=self.testDir, verbose=False)
        # now run EQ4 twice with different parameter sets.
        paramsRef = self.Config.beginParam()
        paramsRef.VF1 = 1.0
        # gen numpy array for running
        arr = np.zeros((2, len(paramsRef)))
        for indx, (k, v) in enumerate(zip(['RHCRIT', 'VF1'], [0.65, 1.1])):
            params = paramsRef.copy()
            params.loc[k] = v
            arr[indx, :] = params.values

        with self.assertRaises(Submit.runModelError):
            result = EQ4(arr, MODELRUN=MODELRUN)  # should fail.

        # now fakeSubmit the models
        MODELRUN.submit()
        # expect to have two directories names jc001 to jc002
        # count dirs
        dirs = self.checkDir(MODELRUN.rootDir, 41)

        self.assertEqual(2, len(dirs))  # expect two directories (Ctl to 40 years)
        # now run again. Should have 4 directories and all times should be 80.
        MODELRUN = Submit.ModelSubmit(self.Config, HadCM3.HadCM3, None, EQ4, modelDirs=dirs,
                                      fakeFn=fakeEQ4, rootDir=self.testDir, verbose=False)
        with self.assertRaises(Submit.runModelError):
            result = EQ4(arr, MODELRUN=MODELRUN)  # should fail.
        # TODO add code to verify that ASTART & OSTART works...
        # now fakeSubmit the models
        MODELRUN.submit()
        # expect to have four directories names jc001 to jc004
        # count dirs
        dirs = self.checkDir(MODELRUN.rootDir, 81)

        # now run again. Should have 4 directories and all times should be 80.
        MODELRUN = Submit.ModelSubmit(self.Config, HadCM3.HadCM3, None, EQ4, modelDirs=dirs,
                                      fakeFn=fakeEQ4, rootDir=self.testDir, verbose=False)
        # and final read should give values.
        result = EQ4(arr, MODELRUN=MODELRUN)  # should succeed
        # result d/VF1 should be 1.1 times larger than d/RHCRIT
        ratio = result[1, :] / result[0, :]
        npt.assert_allclose(ratio, 1.1, rtol=1e-4, atol=1e-4)

    def test_TCR(self):
        """
        Test that submit method (with fake fn ) for TCR works.
        :return:
        """
        configFile = os.path.join(os.environ['OPTCLIMTOP'], 'Configurations', 'example_coupledJacTCR.json')
        self.Config = StudyConfig.readConfig(configFile)
        MODELRUN = Submit.ModelSubmit(self.Config, HadCM3.HadCM3, None, TCR,
                                      fakeFn=fakeTCR, rootDir=self.testDir, verbose=False)
        # now run TCR twice with different parameter sets.
        paramsRef = self.Config.beginParam()
        paramsRef.VF1 = 1.0
        # gen numpy array for running
        arr = np.zeros((2, len(paramsRef)))
        for indx, (k, v) in enumerate(zip(['RHCRIT', 'VF1'], [0.65, 1.1])):
            params = paramsRef.copy()
            params.loc[k] = v
            arr[indx, :] = params.values

        with self.assertRaises(Submit.runModelError):
            result = TCR(arr, MODELRUN=MODELRUN)  # should fail.

        # now fakeSubmit the models
        MODELRUN.submit()
        # expect to have two directories named jt001 to jc002
        # count dirs
        expectYr = 41
        dirs = self.checkDir(MODELRUN.rootDir, expectYr, rootname='jt00')
        self.assertEqual(2, len(dirs))  # expect two directories (Ctl to 40 years)
        # now run again with times increasing by 40 each time.
        # Should have 4 directories and all times should be expectYr.
        while expectYr < 181:
            # need to generate a new MODELRUN bu reading in all directories containing run info
            modelDirs = next(os.walk(self.testDir))[1]  # all sub-directories
            modelDirs = sorted(modelDirs)
            MODELRUN = Submit.ModelSubmit(self.Config, HadCM3.HadCM3, None, TCR,
                                          fakeFn=fakeTCR, rootDir=self.testDir, verbose=False, modelDirs=modelDirs)
            expectYr += 40
            with self.assertRaises(Submit.runModelError):
                result = TCR(arr, MODELRUN=MODELRUN)  # should fail.
            MODELRUN.submit()  # run it.
            dirs = self.checkDir(MODELRUN.rootDir, expectYr, rootname='jt00')
            self.assertEqual(4, len(dirs), msg='Expecting %i directories got %i' % (4, len(dirs)))

        # and final read should give values.
        result = TCR(arr, MODELRUN=MODELRUN)  # should succeed
        # result d/VF1 should be 1.1 times larger than d/RHCRIT
        ratio = result[1, :] / result[0, :]
        npt.assert_allclose(ratio, 1.1, rtol=1e-4, atol=1e-4)
        config = MODELRUN.runConfig(Config)
        with pd.option_context('max_rows', None, 'max_columns', None, 'precision', 2,
                               'expand_frame_repr', True, 'display.width', 120):
            print("params \n", config.parameters(normalise=True).T)
            print("simObs \n", config.simObs().T)

if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  # actually run the test cases

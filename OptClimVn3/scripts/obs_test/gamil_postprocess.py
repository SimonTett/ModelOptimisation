#!/usr/bin/env python

""" Compute simulated observables using xarray.
xarray appears to be a lot faster than iris!
 Based on comp_obs. Hopefully faster 
Observations generated are in genProcess (or help for script)

 """

import argparse  # parse arguments
import json  # get JSON library
import os  # OS support
import pathlib
import re
import typing
import numpy as np
import xarray

import json
# import Ngl

# for moleculr wts.
Avogadro_constant = 6.022E23
dry_air_gas_constant = 287.05
# Mol masses need to covert from moles to mass. (and for conversion)
# Taken from Molar Mass Calculator, https://www.webqc.org/mmcalc.php, 12/20/2022

mole_wt = dict(CO2=44.010, CH4=16.042, N2O=44.013, CFC12eq=120.91,
               HFC134aeq=102.03, O3=47.9982,
               S=32.0650, SO2=64.0638, DMS=62.1340, DryAir=28.9644,
               C=12.0107, OH=17.00734, O2H=33.00674, H2O2=34.01468)  # these values are in g.
# and convert to kg as everything should be SI!
for k in mole_wt.keys():
    mole_wt[k] /= 1000

moles_to_mass = dict()  # conversion factor to convert moles substance/moles air into kg substance/kg air
DU = dict()  # conversion factors for DU from mass mixing ratio. Only really useful for O3 and SO2
for k, mass in mole_wt.items():  # convert from mass of mole of material to mass mixing ratio. Assumes small mmr
    moles_to_mass[k] = mass / mole_wt['DryAir']
    DU[k] = 2.241e3 / mass  # DU conversion factor.


def failCleanMissing(function):
    """
    Decorator function to produce a clean fail if data is not present
    :param function: function to be decorated.
    :return: decorated function
    """
    def mod_func(*args, **kwargs):
        try:
            result=function(*args, **kwargs)
        except KeyError:
            result=None
        return result
    return mod_func

def guess_lat_lon_vert_names(dataArray):
    """
    Guess the names of latitude,longitude & vertical co-ords in the dataArray
    starts with latitude/longitude, then lat/lon, then latitude_N/longitude_N then lat_N,lon_N
    and with atmosphere_hybrid_sigma_pressure_coordinate, altitude, air_pressure for vertical

    :param dataArray:
    :return: latitude, longitude & vertical co-ord names
    """

    def find_name(dimensions, patterns):
        name = None
        for pattern in patterns:
            reg = re.compile("^" + pattern + r"(_\d+)?" + "$")
            for d in dimensions:
                if reg.match(d):
                    name = d
                    break
        return name

    dims = dataArray.dims
    lat_patterns = ['latitude', 'lat']
    lon_patterns = ['longitude', 'lon']
    vert_patterns = ['lev', 'altitude', 'air_pressure']   #liangwj
    lat_name = find_name(dims, lat_patterns)
    lon_name = find_name(dims, lon_patterns)
    vert_name = find_name(dims, vert_patterns)

    return lat_name, lon_name, vert_name


def model_delta_p(pstar, a_bounds, b_bounds):
    """
    Compute the delta_pressures for each model level for UM hybrid-sigma grid
    For which pressure is a+b*pstar.
    :param pstar: Surface pressure
    :param a_bounds: bounds of "a" co-efficients
    :param b_bounds: bounds of "b" co-efficients
    :return: Pressure thicknesses as a function of vertical co-ord for each model grid cell
    """
    #print(a_bounds.dims) #('time', 'lev')
    bnds_coord = [d for d in a_bounds.dims if d.startswith('lev')][0] #liangwj
    #print(bnds_coord)
    delta = lambda bnds: bnds.sel({bnds_coord: 1},method='nearest') - bnds.sel({bnds_coord: 0},method='nearest')  #liangwj
    #print(delta)
    delta_a = delta(a_bounds)
    delta_b = delta(b_bounds)
    delta_p = delta_a + pstar * delta_b
    #print(delta_p)

    return delta_p


def total_column(data, delta_p, scale=None,vertical_coord=None):
    """
    Compute total column of substance (kg/m^2) assuming hydrostatic atmosphere.
        (Atmospheric mass in layer is \Delta P/g)
    :param data: dataArray of the variable for which column calculation is being done.
        Assumed to be a mass mixing ratio (kg/kg)
    :param delta_p: dataArray of pressure thicknesses (Pa) for each level.
    :param scale: real scale factor.If None no scaling is done
    :param vertical_coord: default is None. If None then will be guessed using guess_lat_long_vert_names()
    :return: total_column of substance
    """
    if data is None:
        return None
    lat, lon, vertical_coord = guess_lat_lon_vert_names(data)
    mass = (data * delta_p / 9.81).sum(vertical_coord)  # work out mass by integrating.
    if scale is not None:
        mass *= scale

    return mass


def names(dataset, name=None):
    """
     Return dictionary of standard (or long)  names for each variable in a dataset.

    :param dataset: xarray data
    :param name: what you want to return (None, 'standard','long'). If None (default) then standard_name will be returned
    :return: dict of standard names
    """

    if name is None or name == 'standard':
        key = 'standard_name'
    elif name == 'long':
        key = 'long_name'
    else:
        raise Exception(f"Do not know what to do with {name}")

    lookup = {}
    for var in dataset.variables:
        try:
            name = dataset[var].attrs[key]
            lookup[name] = var  # if not present then won't update lookup
        except KeyError:
            pass

    return lookup





def genProcess(dataset, land_mask, latitude_coord=None):
    from typing import List
    """
    Setup the processing information
    :param dataset: the dataset containing the data
    :param land_mask -- land mask as a dateArray.
    :param latitude_coord -- name of latitude co-ord (optional; default None)
      If set to None then will be guessed using
    :return: dict containing data to be processed.
    """
    # create long & standard name lookup
    lookup_std = names(dataset)

    lookup_long = names(dataset, name='long')

    def name_fn(name: typing.Union[str,List[str]],  #liangwj
                dataset: xarray.Dataset,
                *args,name_type=None,**kwargs) -> (xarray.DataArray,None):
        """
        Lookup name/variable and then return datarray corresponding to it.
        If list provided iterate over. Any None return None; and then sum values.
        If not present return None
        :param long_name:
        :param **kwargs: Remaining kw args passed to select
        :param dataset:
        :return:
        """
        # handle list
        if isinstance(name,list): # loop over vars calling name_fn and then add
            results=[] 
            for n in name:
                var = name_fn(n,dataset,*args,name_type=name_type,**kwargs)
                if var is None:
                    return None
                results.append(var)

            result=results[0].squeeze().copy()
            for r in results[1:]:
                result += r.squeeze()
            return result # computed result
            
                
        if name_type is None or name_type=='name':
            if name in dataset.variables:
                var = name
            else:
                var=None
                name_type='name'
        elif name_type=='long':
            var = lookup_long.get(name)
        elif name_type=='standard':
            var = lookup_std.get(name)

        else:
            raise ValueError(f"Do not know what to do with name_type {name_type}")

        if var is None: #failed to find name so return None
            print(f"Failed to find name {name} of type {name_type}")
            return None
        da = dataset[var]
        if (len(args) >0) or (len(kwargs) > 0):
            da=da.sel(*args,**kwargs,method='nearest')
        return da


    def reff(dataset):
        reffwt=dataset.get('UM_m01s01i245_vn405.0')
        wt=dataset.get('UM_m01s01i246_vn405.0')
        if (reffwt is None) or (wt is None):
            print("Failed to find reffwt or wt")
            return None
        return reffwt/wt

    if latitude_coord is None:
        latitude_coord, lon, vert = guess_lat_lon_vert_names(dataset[lookup_long['Temperature at 2m height']]) #liangwj
    constrain_60S = dataset[latitude_coord] >= -60.
    # need to extract the actual values...
    constrain_60S = constrain_60S.lat[constrain_60S]  #liangwj

    coord_500hPa = dict(lev=0.00133607736846712)  # co-ord value for 500 hPa  liangwj???
    coord_50hPa = dict(lev=0.00513899915929914)  # co-ord value for 50# hPa    liangwj
    # set up the data to be meaned. Because of xarray's use of dask no loading happens till
    # data actually processed (below)
    #print(dataset.hyam)
    delta_p = model_delta_p(dataset.PS,
                            dataset.hyam,
                            dataset.hybm) #liangwj
    SO2_scale = mole_wt['SO2'] / mole_wt['S'] # convert from S to SO2

    #============liagnwj==============
    # arrayinterp = Ngl.vinth2p(np.array(dataset.PS), np.array(dataset.hyam), np.array(dataset.hybm), np.array([500]),
    #                     dataset.PS, 2, 1000, 1, False)
    # print(arrayinterp.shape)
    # ============liagnwj==============

    #condition = (land_mask > 0) & (land_mask <= 1)
    #result = dataset[lookup_long['Temperature at 2m height']] * np.where(condition, land_mask, np.nan)
    process = {
        #'TEMP@50': name_fn('Temperature',dataset,coord_50hPa,name_type='long'),
        'TEMP@500': name_fn('Temperature',dataset,coord_500hPa,name_type='long'),
        'RH@500': name_fn('Relative humidity',dataset,coord_500hPa,name_type='long'),
        'OLR': dataset[lookup_long['Long wave upward flux at top of atmosphere']],
        'OLRC': name_fn('Long wave clearsky upward flux at top of atmosphere',dataset,name_type='long'),
        'RSR': dataset[lookup_long['Short wave upward flux at top of atmosphere']],
        'RSRC': name_fn('Short wave upward flux at top of atmosphere_C',dataset,name_type='long'),
        'INSW': dataset[lookup_long['Short wave downward flux at top of atmosphere']],
        # 'LAT': dataset[lookup_long['Temperature at 2m height']] * np.where(condition, land_mask, np.nan).sel(
        #     {latitude_coord: constrain_60S}),
        'LAT': xarray.where(land_mask, dataset[lookup_long['Temperature at 2m height']], np.nan).sel(
            {latitude_coord: constrain_60S}),
        'Lprecip': xarray.where(land_mask, dataset[lookup_long['Total (convective and large-scale) precipitation rate']], np.nan).sel(
            {latitude_coord: constrain_60S}),
        'MSLP': dataset[lookup_long['Sea level pressure']],
        #'Reff': dataset[lookup_long['Average Cloud Top droplet effective radius']], #reff(dataset),  liangwj
#         # HadXM3 carries all S compounds as mass of S. So SO2 & DMS need conversion to SO2 & DMS for
#         # comparison with obs.
#         'SO2_col': total_column(name_fn('mass_fraction_of_sulfur_dioxide_in_air',dataset,name_type='name'),
#                                 delta_p,scale=SO2_scale),
#         'dis_col': total_column(name_fn("SO4 DISSOLVED AEROSOL AFTER TSTEP",dataset,name_type='long'),delta_p),
#         'aitkin_col': total_column(dataset[lookup_long['SO4 AITKEN MODE AEROSOL AFTER TSTEP']], delta_p),
#         'accum_col': total_column(name_fn('SO4 ACCUM. MODE AEROSOL AFTER TSTEP',dataset,name_type='long'), delta_p),
#         'DMS_col': total_column(name_fn('mass_fraction_of_dimethyl_sulfide_in_air',dataset,name_type='name'),
#                                         delta_p, scale=mole_wt['DMS'] / mole_wt['S']) ,
#         'O3_col_DU': total_column(name_fn('UM_m01s02i260_vn405.0',dataset,name_type='name'),
#                                   delta_p,scale=DU['O3']),
#         'Trop_SW_up': name_fn('tropopause_upwelling_shortwave_flux',dataset),
#         'Trop_SW_net': name_fn('tropopause_net_downward_shortwave_flux',dataset),
#         'Trop_LW_up': name_fn('tropopause_upwelling_longwave_flux', dataset),
#         'Trop_LW_net': name_fn('tropopause_net_downward_longwave_flux', dataset),
# ## deposition rates
# # dry deposition rates
#         'Dry_SO2':name_fn("SO2 SURFACE DRY DEP FLUX KG/M2/S",dataset,name_type='long'),
#         'Dry_ait':name_fn("SO4 AIT SURF DRY DEP FLUX KG/M2/S",dataset,name_type='long'),
#         'Dry_acc':name_fn("SO4 ACC SURF DRY DEP FLUX KG/M2/S",dataset,name_type='long'),
#         'Dry_dis':name_fn("SO4 DIS SURF DRY DEP FLUX KG/M2/S",dataset,name_type='long'),
# # Wet depositions are sum of ls  rain and convection.
#         'Wet_SO2':name_fn(["SO2 SCAVENGED BY LS PPN KG/M2/S",
#                            "SO2 SCAVENGED BY CONV PPN KG/M2/SEC"], dataset,name_type='long'),
#         'Wet_ait':name_fn(["SO4 AIT SCAVNGD BY LS PPN KG/M2/S",
#                            "SO4 AIT SCAVNGD BY CONV PPN KG/M2/S"],dataset,name_type='long'),
#         'Wet_acc':name_fn(["SO4 ACC SCAVNGD BY LS PPN KG/M2/S",
#                            "SO4 ACC SCAVNGD BY CONV PPN KG/M2/S"],dataset,name_type='long'),
#         'Wet_dis':name_fn(["SO4 DIS SCAVNGD BY LS PPN KG/M2/S",
#                            "SO4 DIS SCAVNGD BY CONV PPN KG/M2/S"],dataset,name_type='long'),
#
    }

    process['netflux'] = process['INSW'] - process['RSR'] - process['OLR']
    #dataset[lookup_long['Short wave net flux at top of atmosphere']]-process['OLR']
    #process['INSW'] - process['RSR'] - process['OLR']
    #process['RSRC'] = process['INSW'] - process['RSRC']

    return process


def means(dataArray, name, latitude_coord=None):
    """ 
    Compute means for NH extra tropics, Tropics and SH extra Tropics. 
    Tropics is 30N to 30S. NH extra tropics 30N to 90N and SH extra tropics 90S to 30S 
    Arguments:
        :param dataArray -- dataArray to be processed
        :param name -- name to call mean.
        :param latitude_coord: name of the latitude co-ord. Default is NOne.
            If not set then guess_lat_long_vert_names() will be used


    """

    if latitude_coord is None:
        latitude_coord, lon, vert = guess_lat_lon_vert_names(dataArray)

    wt = np.cos(np.deg2rad(dataArray[latitude_coord]))  # simple cos lat weighting.
    # constraints and names for regions.
    constraints = {
        'GLOBAL': None,
        'NHX': lambda y: y > 30.0,
        'TROPICS': lambda y: (y >= -30) & (y <= 30.0),
        'SHX': lambda y: y < -30.0,
    }
    means = dict()
    for rgn_name, rgn_fn in constraints.items():
        if rgn_fn is None:
            v = dataArray.squeeze().weighted(wt).mean()
        else:
            msk = rgn_fn(dataArray[latitude_coord])  # T where want data
            v = dataArray.where(msk, np.nan).squeeze().weighted(wt.where(msk, 0.0)).mean()
        means[name + '_' + rgn_name] = float(v.load().squeeze().values)
        # store the actual values -- losing all meta-data

    return means  # means are what we want

#================liangwj================
def extract_year_month(file_path):
    # Split the file path by '/' and take the last element
    file_name = file_path.split('/')[-1]
    # Split the file name by '.' and take the second-to-last element
    year_month = file_name.split('.')[-2]
    # Split the year and month using '-'
    year, month = map(int, year_month.split('-'))
    return (year, month)
#================liangwj================

def do_work():
    # parse input arguments

    parser = argparse.ArgumentParser(description="""
    Post process Unified Model data to provide 32 simulated observations. Example use is:
    
    #comp_sim_obs.py input.json /exports/work/geos_cesd_workspace/OptClim/Runs/st17_reg/01/s0101/A/output.json nc/apm/*.nc
    ./comp_sim_obs_new.py input.json -d /BIGDATA2/sysu_atmos_wjliang_1/FG3/run/amip1d_nudging/run/atmhist 
    
    Observations are:
    Global mean, Northern Hemisphere Extra-tropical average (30N-90N), Southern Hemisphere Extra-tropical (90S-30S) mean and 
    Tropical (30S - 30N) mean for:
      Temperature at 500 hPa,
      Relative Humidity at 500 hPa
      Outgoing Longwave Radiation
      Outgoing Clearsky Longwave Radiation
      Reflected Shortwave Radiation
      Clear sky reflected shortwave
      Land air temperature at 1.5m (north of 60S)
      Land precipitation at 1.5m  (north of 60S)
      Effective cloud radius
      total column SO2 
      total column DMS
      Netflux 
    
      Northern Hemisphere Extra-tropical and Tropical Mean Sea Level Pressure difference from global average
    """
                                     )
    parser.add_argument("CONFIG", help="The Name of the Config file. Should be a json file with a postProcess entry.")
    parser.add_argument("-d", "--dir", help="The Name of the input directory")
    parser.add_argument("OUTPUT", nargs='?', default=None,help="The name of the output file. Will override what is in the config file")
    parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
    args = parser.parse_args()  # and parse the arguments
    # setup processing
    with open(args.CONFIG,'rt') as fp:
        config = json.load(fp)

    options = config.get('postProcess', {})
    path = options.get('netcdf_path', 'atm/hist')

    # work out the files if needed
    if args.dir is None:
        cwd = pathlib.Path.cwd()
        rootdir = pathlib.Path("/BIGDATA2/sysu_atmos_wjliang_1/FG3/run")/cwd.name/path #pathlib.Path("/BIGDATA2/sysu_atmos_wjliang_1/FG3/run/amip1d_nudging/atm/hist")#self.model_dir  #pathlib.Path.cwd()/path   #liangwj
    else:
        rootdir = pathlib.Path(args.dir)
    files = list(rootdir.glob('*gamil.h0.*.nc'))#[3:] #从201101开始
    files_1=[str(i) for i in files]
    # print(str(files[0])[-10:-6])
    #print(files_1)

    # #================liangwj================
    # sorted_file_paths = sorted(files_1, key=lambda x: extract_year_month(x))
    # files=[pathlib.Path(i) for i in sorted_file_paths][3:]
    # # ================liangwj================
    # print(files,len(files),"PP info")
    
    files=[pathlib.Path(i) for i in files_1]
    # ================liangwj================
    print(files,len(files),"PP info")


    #print(files,"datafiles in progress")
      
    mask_file = options['mask_file']
    mask_file = pathlib.Path(os.path.expandvars(mask_file)).expanduser()
    mask_name = options['mask_name']
    start_time = options['start_time'] #日期晚一个月   '2010-02-01 00:00:00'  #liangwj
    end_time = options['end_time']    #日期晚一个月   '2010-02-01 00:00:00'  #liangwj
    if args.OUTPUT is None:
        output_file = options['outputPath']  # better be defined so throw error if not
    else:
        output_file = args.OUTPUT

    output_file = os.path.expanduser(os.path.expandvars(output_file) )    # deal with any env vars and expand user

    verbose = args.verbose

    #if verbose:  # print out some helpful information..
    print("============PP===========")
    print(pathlib.Path.cwd(), path, "path")
    print(args.dir, "args.dir")
    print("dir", rootdir)
    print("mask_file", mask_file)
    print("land_mask", mask_name)
    print("start_time", start_time)
    print("end_file", end_time)
    print("output", output_file)
    if verbose > 1:
        print("options are ", options)
    print("============PP===========")

    land_mask = xarray.load_dataset(mask_file)[mask_name].squeeze()  # land/sea mask
    #print(land_mask)
    latitude_coord = options.get('latitude_coord', None)  #liangwj
    #print(latitude_coord)
    # code below does not work when data is on my M drive on my laptop...
    dataset = xarray.open_mfdataset(files,engine="netcdf4",parallel=True).sortby('time')  #liangwj  sortby is really important as want co-ords to be monotonic
    dataset = dataset.sel(time=slice(start_time, end_time))

    process = genProcess(dataset, land_mask, latitude_coord=latitude_coord)

    # now to process all the data making output.
    results = dict()
    for name, dataArray in process.items():
        if dataArray is None:  # no dataarray for this name
            print(f"{name} is None. Not processing")
            continue
        mean = means(dataArray, name, latitude_coord=latitude_coord)  # compute the means
        results.update(mean)  # and stuff them into the results dict.
        if verbose > 1:
            print(f"Processed {name} and got {mean}")

    # now fix the MSLP values. Need to remove the global mean from values and the drop the SHX value.
    results.pop('MSLP_SHX')
    for k in ['MSLP_NHX', 'MSLP_TROPICS']:
        results[k + '_DGM'] = results.pop(k) - results['MSLP_GLOBAL']

    results = {key: value for key, value in results.items() if "_GLOBAL" not in key or key == "netflux_GLOBAL"} #liangwj
    results = {key: value for key, value in results.items() if "netflux_" not in key or key == "netflux_GLOBAL"}  # liangwj
    results = {key: value for key, value in results.items() if "INSW_" not in key or key == "netflux_GLOBAL"}  # liangwj

    if verbose:  # print out the summary data for all created values
        for name, value in results.items():
            print(f"{name}: {value:.4g}")
        print("============================================================")

    # now to write the data
    with open(output_file, 'w') as fp:
        json.dump(results, fp, indent=2)


# TODO add  some test cases...

import unittest


class testComp_obs_xarray(unittest.TestCase):
    """
    Test cases for comp_obs_xarray.

    Some cases to try:
    1) That means fn works -- set to 1 > 30N; 2 for lat beteen 0S & 30N; 3 for less than 30S.
    2) That global mean value is close (5%) to the simple mean but not the same...
    3) That for LAT & LPrecip values have the expected number of points..  (basically we are missing Ant. & land only)

    """

    def setUp(self):
        """
        Standard setup for all test cases
        :return: nada
        """


if __name__ == "__main__":
    do_work()

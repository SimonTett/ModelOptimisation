#!/usr/bin/env python

""" Compute simulated observables using xarray.
Xarray appears to be a lot faster than iris!
 Based on comp_obs. Hopefully faster 
Observations are:
Global mean, Northern Hemisphere Extratropical average (30N-90N), Southern Hemisphere Extra-tropical (90S-30S) mean and 
Tropical (30S - 30N) mean for:

  Temperature at 500 hPa,
  Relative Humidity at 500 hPa
  Outgoing Longwave Radiation (all and clear sky)
  Reflected Shortwave Radiation (all and clear sky)
  Land air temperature at 1.5m (north of 60S)
  Land precipitation at 1.5m  (north of 60S)
  Cloud effective radius

and Northern Hemisphere Extra-tropical and Tropical Mean Sea Level
Pressure difference from global average

 """

import argparse  # parse arguments
import json  # get JSON library
import os  # OS support
import pathlib

import numpy as np
import xarray

import StudyConfig

def genProcess(dataset, land_mask, latitude_coord='latitude',):
    """
    Setup the processing information
    :param dataArray: the dataArray containing the data
    :param land_mask -- land mask as a dateArray.
    :param latitude_coord -- name of latitude co-ord (optional; default latitude)
    :return: dict containing data to be processed.
    """
    # create long & standard name lookup
    lookup_std = {}
    lookup_long = {}
    for var in dataset.variables:
        try:
            lookup_std[dataset[var].standard_name] = var
        except AttributeError:
            pass
        try:
            lookup_long[dataset[var].long_name] = var
        except AttributeError:
            pass
    constrain_60S = dataset[latitude_coord] >= -60.
    coord_500hPa = dict(air_pressure=500)  # co-ord value for 500 hPa
    # set up the data to be meaned. Because of xarray's use of dask no loading happens till
    # data actually processed (below)
    process = {
        'TEMP@500': dataset[lookup_long['TEMPERATURE ON PRESSURE LEVELS']].sel(coord_500hPa, method='nearest'),
        'RH@500': dataset[lookup_long['RELATIVE HUMIDITY ON PRESSURE LEVELS']].sel(coord_500hPa, method='nearest'),
        'OLR': dataset[lookup_std['toa_outgoing_longwave_flux']],
        'OLRC': dataset[lookup_std['toa_outgoing_longwave_flux_assuming_clear_sky']],
        'RSR': dataset[lookup_std['toa_outgoing_shortwave_flux']],
        'RSRC': dataset[lookup_std['toa_outgoing_shortwave_flux_assuming_clear_sky']],
        'INSW': dataset[lookup_std['toa_incoming_shortwave_flux']],
        'LAT': xarray.where(land_mask, dataset[lookup_long['TEMPERATURE AT 1.5M']], np.nan)[constrain_60S],
        'Lprecip': xarray.where(land_mask, dataset[lookup_std['precipitation_flux']], np.nan)[constrain_60S],
        'MSLP': dataset[lookup_std['air_pressure_at_sea_level']],
        'REFF': dataset['UM_m01s01i245_vn405.0'] / dataset['UM_m01s01i246_vn405.0'],
    }

    process['netflux'] = process['INSW'] - process['RSR'] - process['OLR']

    return process

def means(dataArray, name, latitude_coord='latitude'):
    """ 
    Compute means for NH extra tropics, Tropics and SH extra Tropics. 
    Tropics is 30N to 30S. NH extra tropics 30N to 90N and SH extra tropics 90S to 30S 
    Arguments:
        :param dataArray -- dataArray to be processed
    :param name -- name to call mean.
    :param latitude_coord: name of the latitude co-ord. Default is 'latitude


    """

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
        means[name + '_' + rgn_name] = float(v.load().squeeze().values)  # store the actual values -- losing all meta-data
    return means  # means are what we want


def do_work():
    # parse input arguments

    parser = argparse.ArgumentParser(description="""
    Post process Unified Model data to provide 32 simulated observations. Example use is:
    
    comp_obs_xarray.py input.json /exports/work/geos_cesd_workspace/OptClim/Runs/st17_reg/01/s0101/A/output.json nc/apm/*.nc
    
    Observations are:
    Global mean, Northern Hemisphere Extratropical average (30N-90N), Southern Hemisphere Extra-tropical (90S-30S) mean and 
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
      Netflux 
    
      Northern Hemisphere Extra-tropical and Tropical Mean Sea Level Pressure difference from global average
    """
                                     )
    parser.add_argument("CONFIG", help="The Name of the Config file")
    parser.add_argument("-d", "--dir", help="The Name of the input directory")
    parser.add_argument("-o", "--output", help="The name of the output file. Will override what is in the config file")
    parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
    args = parser.parse_args()  # and parse the arguments
    # setup processing

    config = StudyConfig.readConfig(args.CONFIG)
    options = config.getv('postProcess', {})
    path = options.get('netcdf_path', 'nc/apm')

    # work out the files if needed
    if args.dir is None:
        files = list(pathlib.Path(path).glob('*.nc'))
    else:
        files = list(pathlib.Path(args.dir).glob('*.nc'))  # if have input files specified.

    mask_file = options['mask_file']
    mask_file = pathlib.Path(os.path.expandvars(mask_file)).expanduser()
    mask_name = options['mask_name']
    start_time = options['start_time']
    end_time = options['end_time']
    if args.output is None:
        output_file = config['postProcess']  # better be defined so throw error if not
    else:
        output_file = args.output
    # and deal with any env vars
    output_file = os.path.expandvars(output_file)


    verbose = args.verbose

    if verbose:  # print out some helpful information..
        print("mask_file", mask_file)
        print("land_mask", mask_name)
        print("start_time", start_time)
        print("end_file", end_time)
        print("output", output_file)
        if verbose > 1:
            print("options are ", options)

    land_mask = xarray.load_dataset(mask_file)[mask_name].squeeze()  # land/sea mask
    latitude_coord = options.get('latitude_coord', 'latitude')

    dataset = xarray.open_mfdataset(files, combine='by_coords').sel(time=slice(start_time, end_time))


    # now to process data.
    process = genProcess(dataset,land_mask,latitude_coord=latitude_coord)

    # now to process all the data making output.
    results = dict()
    for name, dataArray in process.items():
        mean = means(dataArray, name, latitude_coord=latitude_coord)  # compute the means
        results.update(mean)  # and stuff them into the results dict.
        if verbose > 1:
            print(f"Processed {name} and got {mean}")

    # now fix the MSLP values. Need to remove the global mean from values and the drop the SHX value.
    results.pop('MSLP_SHX')
    for k in ['MSLP_NHX', 'MSLP_TROPICS']:
        results[k + 'DGM'] = results.pop(k) - results['MSLP_GLOBAL']

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

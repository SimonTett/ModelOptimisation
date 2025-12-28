#!/usr/bin/env python
# *****************************COPYRIGHT*******************************
# (C) Crown copyright Met Office. All rights reserved.
# For further details please refer to the file COPYRIGHT.txt
# which you should have received as part of this distribution.
# *****************************COPYRIGHT*******************************

''' Script to test increment budget in a UM experiment '''

import os

# Must be here to run on batch systems
import matplotlib as mpl
if not os.getenv('DISPLAY'):
    mpl.use('AGG')

import inc_code

Q_PHYSICS = ['eastward_wind', 'northward_wind', 'upward_air_velocity',
             'air_temperature', 'specific_humidity',
             'mass_fraction_of_cloud_liquid_water_in_air',
             'mass_fraction_of_cloud_ice_in_air',
             'mass_fraction_of_rain_in_air',
             'cloud_area_fraction_in_atmosphere_layer',
             'liquid_water_cloud_area_fraction_in_atmosphere_layer',
             'ice_cloud_area_fraction_in_atmosphere_layer']

M_PHYSICS = ['eastward_wind', 'northward_wind', 'upward_air_velocity',
             'air_temperature', 'humidity_mixing_ratio',
             'cloud_liquid_water_mixing_ratio', 'cloud_ice_mixing_ratio',
             'rain_mixing_ratio', 'cloud_area_fraction_in_atmosphere_layer',
             'liquid_water_cloud_area_fraction_in_atmosphere_layer',
             'ice_cloud_area_fraction_in_atmosphere_layer']


def get_bool_env(opt):
    ''' Method to extract boolean value from text value in os.environ'''
    opt_str = os.environ.get(opt, "false").lower()
    if opt_str not in ("false", "true"):
        msg = f'{opt} option must be "false" or "true"'
        raise ValueError(msg)
    return {'false': False, 'true': True}.get(opt_str)


def write_passed(tests, outdir):
    ''' Test to see if rose ana should pass '''
    with open(os.path.join(outdir, 'inc_tests.txt'), 'w') as ofile:
        for (key, value) in tests.items():
            if value is None:
                msg = f'[INFO]     No test:      None = {key}'
            elif value < 1.0e-10:
                msg = f'[ OK ] Test passed: {value:.3e} = {key}'
            else:
                msg = f'[FAIL] Test failed: {value:.3e} = {key}'
            print(msg, file=ofile, flush=True)


def test_incs(infile, outdir, mr_physics=False, plot_inc=False):
    ''' Actual increments tests '''
    # Set plot options
    kwargs = dict(hgt=False, fixed=True)

    # Determine moist physics type
    if mr_physics:
        prognostics = M_PHYSICS
    else:
        prognostics = Q_PHYSICS

    # Load data
    incs = inc_code.Increments(infile,
                               prognostics=prognostics,
                               mr_physics=mr_physics)

    # Loop over prognostics
    for prognostic in incs.prognostics:
        incs.set_prognostic(prognostic)

        # Check if file has increments for prognostic
        if incs.has_increments():

            # Calculate normalised total residual rms
            incs.set_rms_dict('total_residual', norm=True)

            # Create plots
            if plot_inc:
                incs.plot_budget_page(outdir, **kwargs)

    return incs.get_rms_dict(fulldict=True)


def main():
    infile = os.environ['FILE']
    outdir = f'{os.getcwd()}'
    lplot = get_bool_env('PLOT_INCS')
    lmrphy = get_bool_env('MR_PHYSICS')
    if lplot:
        print(f'[INFO] Creating plots in {outdir}')
    tests = test_incs(infile, outdir, mr_physics=lmrphy, plot_inc=lplot)
    write_passed(tests, outdir)


if __name__ == '__main__':
    main()

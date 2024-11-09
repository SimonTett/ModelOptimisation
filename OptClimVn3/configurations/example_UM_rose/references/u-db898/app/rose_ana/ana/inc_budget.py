#!/usr/bin/env python
# *****************************COPYRIGHT*******************************
# (C) Crown copyright Met Office. All rights reserved.
# For further details please refer to the file COPYRIGHT.txt
# which you should have received as part of this distribution.
# *****************************COPYRIGHT*******************************

# Must be here to run on batch systems
import os
import matplotlib as mpl
if not os.getenv('DISPLAY'):
    mpl.use('AGG')
mpl.rcParams['figure.figsize'] = (17, 12)
mpl.rcParams['axes.titlesize'] = 'x-small'
mpl.rcParams['axes.labelsize'] = 'x-small'
mpl.rcParams['xtick.labelsize'] = 'xx-small'
mpl.rcParams['ytick.labelsize'] = 'xx-small'

from rose.apps.rose_ana import AnalysisTask

import re

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_col
import numpy as np
import scipy.interpolate as spint

import cartopy.mpl.gridliner as cmgridl
import cf_units
import iris
import iris.analysis.cartography as iac
import iris.plot as iplt

iris.FUTURE.clip_latitudes = True


class BudgetTest(AnalysisTask):
    """Analysis task to test increment budgets"""

    def run_analysis(self):
        self.infile = self.options['file']
        self.lplot = self.boolean_option('plot_incs')
        if self.lplot:
            msg = 'Creating plots in {0}'.format(os.getcwd())
        tests = self.increments()
        self.passed = self.has_passed(tests)

    def boolean_option(self, opt):
        opt_str = self.options.pop(opt, ".false.")
        if opt_str not in (".false.", ".true."):
            msg = "{} option must be '.false.' or '.true.'".format(opt)
            raise ValueError(msg)
        return {'.false.': False, '.true.': True}.get(opt_str)

    def has_passed(self, tests):
        'Test to see if rose ana should pass'
        test = True
        for (key, value) in tests.items():
            if value is None:
                msg = '    No test:      None = {0}'
                prefix = '[INFO] '
            elif value < 1.0e-10:
                msg = 'Test passed: {1:.3e} = {0}'
                prefix = '[ OK ] '
            else:
                msg = 'Test failed: {1:.3e} = {0}'
                prefix = '[FAIL] '
                test = False
            self.parent.reporter(msg.format(key, value), prefix=prefix)
        return test

    def increments(self):
        'Calculate increment residiual and plot if requested'
        allcubes = iris.load(self.infile, callback=inc_callback)

        progs_to_check = ['eastward_wind',
                          'northward_wind',
                          'upward_air_velocity',
                          'air_temperature',
                          'specific_humidity',
                          'mass_fraction_of_cloud_liquid_water_in_air',
                          'mass_fraction_of_cloud_ice_in_air',
                          'mass_fraction_of_rain_in_air',
                          'cloud_area_fraction_in_atmosphere_layer',
                          'liquid_water_cloud_area_fraction_in_atmosphere_layer',
                          'ice_cloud_area_fraction_in_atmosphere_layer']

        prog_res_rms = {prog: None for prog in progs_to_check}
        for prog in progs_to_check:
            # Extract cubes for given prognostic field
            name_lambda = lambda cube: cube.name().startswith('tendency_of_'+prog)
            name_cons = iris.Constraint(cube_func=name_lambda)
            cubes = allcubes.extract(name_cons)

            if cubes:

                # Get increment total if it exists
                try:
                    name_lambda = lambda cube: '_due_to_' not in cube.name()
                    name_cons = iris.Constraint(cube_func=name_lambda)
                    totcube = cubes.extract_strict(name_cons).copy()
                except iris.exceptions.ConstraintMismatchError:
                    totcube = None

                # Remove total from cube list if it exists
                if totcube:
                    name_lambda = lambda cube: '_due_to_' in cube.name()
                    name_cons = iris.Constraint(cube_func=name_lambda)
                    cubes = cubes.extract(name_cons)

                # Sum total of component increments
                name = join_cf_stdname(prog, 'derived_model_total')
                sumcube = sumcubes(cubes, name)

                # Calculate residual if total exists
                if totcube and sumcube:
                    rescube = totcube - sumcube
                    rescube.rename(totcube.name())
                    prog_res_rms[prog] = \
                        (area_rms(rescube.copy(), vert='model_level_number') /
                         area_rms(totcube.copy(), vert='model_level_number'))

                if self.lplot:
                    self.plot(prog, cubes, totcube, sumcube,
                              rescube, prog_res_rms[prog])

        return prog_res_rms

    def plot(self, prog, cubes, totcube, sumcube, rescube, resrms):
        'Plot model increments with derived totals'
        # Common plotting routine kwargs
        kwargs = dict(hgt=False, fixed=True, rows=9, columns=3)

        # Extract "physics" increment cube
        name = join_cf_stdname(prog, 'derived_model_physics')
        phycube = sumcubes(cubes, name)

        # Extract "dynamics" increment cube
        name = join_cf_stdname(prog, 'derived_model_dynamics')
        dyncube = sumcubes(cubes, name)

        # Calculate dynamics increment as residual of total - physics
        if totcube and phycube:
            name = join_cf_stdname(prog, 'model_dynamics_as_residual')
            resdyncube = totcube - phycube
            resdyncube.rename(name)
        else:
            resdyncube = None

        # Extract AP1 increment cubes
        name = join_cf_stdname(prog, 'derived_ap1')
        ap1cube = sumcubes(cubes, name)

        # Extract AP2 increment cubes
        name = join_cf_stdname(prog, 'derived_ap2')
        ap2cube = sumcubes(cubes, name)

        # Extract AP3 increment cubes
        name = join_cf_stdname(prog, 'derived_ap3')
        ap3cube = sumcubes(cubes, name)

        # Set up plot
        fig = plt.figure()

        # Plot increment total
        if totcube:
            plot_increment(totcube.copy(), head=1, **kwargs)

        # Plot increment sum
        if sumcube:
            plot_increment(sumcube.copy(), **kwargs)

        # Plot residual if total exists
        if rescube:
            plot_increment(rescube.copy(), head=3, **kwargs)

        # Plot physics and dynamics totals
        if phycube:
            plot_increment(phycube.copy(), **kwargs)
        if dyncube:
            plot_increment(dyncube.copy(), **kwargs)
        if resdyncube:
            plot_increment(resdyncube.copy(), **kwargs)
        if ap1cube:
            plot_increment(ap1cube.copy(), **kwargs)
        if ap2cube:
            plot_increment(ap2cube.copy(), **kwargs)
        if ap3cube:
            plot_increment(ap3cube.copy(), **kwargs)

        # Plot increment components
        for cube in cubes:
            plot_increment(cube.copy(), **kwargs)

        # Add super title
        fig.suptitle(plot_suptitle(prog, resrms))

        # Adjust spacing between plots
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9,
                            bottom=0.05, wspace=0.3, hspace=0.6)

        # Save image
        proglab = PROGNOSTICS[prog]['label'].lower()
        fig.savefig(proglab+'_incs.png')
        plt.close()


# Below is a lot of supplementary routines for the above
# TODO: Work out how to best tidy this up
#
# Notes:
#  1) While we use the tendency standard_names, none of the model increments
#     are strict tendencies (rates), they are amounts. Therefore they need to
#     lose the s-1 from the CF Standard Name table and then have their rates
#     calculated from the time mean metadata.
#  2) Only use increment diagnostics that have been accumulated over a period
#     (even if one timestep!) so that cube metadata works in the above.
#  3) Not all increments exist as yet, both in the CF standard name table and
#     from the model. The list below will need to be updated as new increments
#     are added.
#  4) I prefer to use eastward/northwards instead of x/y in winds
#  5) A better set of contour intervals is required for moist quantities.
#     Currently using a log-style list, but is linear better?
#  6) Could we get the unit_latex routine in cf_units?
#  7) Is there a better colour map function available?


STASH_CF_STDNAME = {
'm01s00i002': 'eastward_wind',
'm01s00i003': 'northward_wind',
'm01s00i004': 'air_potential_temperature',
'm01s00i010': 'specific_humidity',
'm01s00i012': 'mass_fraction_of_cloud_ice_in_air',
'm01s00i150': 'upward_air_velocity',
'm01s00i254': 'mass_fraction_of_cloud_liquid_water_in_air',
'm01s00i266': 'cloud_area_fraction_in_atmosphere_layer',
'm01s00i267': 'liquid_water_cloud_area_fraction_in_atmosphere_layer',
'm01s00i268': 'ice_cloud_area_fraction_in_atmosphere_layer',
'm01s00i271': 'mass_fraction_of_cloud_ice2_in_air',
'm01s00i272': 'mass_fraction_of_rain_in_air',
'm01s00i273': 'mass_fraction_of_graupel_in_air',
'm01s00i388': 'virtual_potential_temperature',
'm01s00i389': 'air_density',
'm01s00i391': 'humidity_mixing_ratio',
'm01s00i392': 'cloud_liquid_water_mixing_ratio',
'm01s00i393': 'cloud_ice_mixing_ratio',
'm01s00i394': 'rain_mixing_ratio',
'm01s00i395': 'graupel_mixing_ratio',
'm01s00i396': 'cloud_ice2_mixing_ratio',
'm01s01i181': 'tendency_of_air_temperature_due_to_shortwave_heating',
'm01s01i182': 'tendency_of_specific_humidity_due_to_shortwave_heating',
'm01s01i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_shortwave_heating',
'm01s01i192': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_shortwave_heating',
'm01s01i193': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_shortwave_heating',
'm01s02i181': 'tendency_of_air_temperature_due_to_longwave_heating',
'm01s02i182': 'tendency_of_specific_humidity_due_to_longwave_heating',
'm01s02i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_longwave_heating',
'm01s02i192': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_longwave_heating',
'm01s02i193': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_longwave_heating',
'm01s03i181': 'tendency_of_air_temperature_due_to_boundary_layer_mixing',
'm01s03i182': 'tendency_of_specific_humidity_due_to_boundary_layer_mixing',
'm01s03i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_boundary_layer_mixing',
'm01s03i184': 'tendency_of_mass_fraction_of_cloud_ice_in_air_due_to_boundary_layer_mixing',
'm01s03i185': 'tendency_of_eastward_wind_due_to_boundary_layer_mixing',
'm01s03i186': 'tendency_of_northward_wind_due_to_boundary_layer_mixing',
'm01s03i187': 'tendency_of_upward_air_velocity_due_to_boundary_layer_mixing',
'm01s03i192': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_boundary_layer_mixing',
'm01s03i193': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_boundary_layer_mixing',
'm01s03i194': 'tendency_of_ice_cloud_area_fraction_in_atmosphere_layer_due_to_boundary_layer_mixing',
'm01s04i141': 'tendency_of_air_temperature_due_to_pc2_checks',
'm01s04i142': 'tendency_of_specific_humidity_due_to_pc2_checks',
'm01s04i143': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_pc2_checks',
'm01s04i144': 'tendency_of_mass_fraction_of_cloud_ice_in_air_due_to_pc2_checks',
'm01s04i152': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_checks',
'm01s04i153': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_checks',
'm01s04i154': 'tendency_of_ice_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_checks',
'm01s04i181': 'tendency_of_air_temperature_due_to_stratiform_precipitation',
'm01s04i182': 'tendency_of_specific_humidity_due_to_stratiform_precipitation',
'm01s04i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_stratiform_precipitation',
'm01s04i184': 'tendency_of_mass_fraction_of_cloud_ice_in_air_due_to_stratiform_precipitation',
'm01s04i189': 'tendency_of_mass_fraction_of_rain_in_air_due_to_stratiform_precipitation',
'm01s04i190': 'tendency_of_mass_fraction_of_cloud_ice2_in_air_due_to_stratiform_precipitation',
'm01s04i191': 'tendency_of_mass_fraction_of_graupel_in_air_due_to_stratiform_precipitation',
'm01s04i192': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_stratiform_precipitation',
'm01s04i193': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_stratiform_precipitation',
'm01s04i194': 'tendency_of_ice_cloud_area_fraction_in_atmosphere_layer_due_to_stratiform_precipitation',
'm01s04i982': 'tendency_of_specific_humidity_due_to_methane_oxidation',
'm01s05i181': 'tendency_of_air_temperature_due_to_convection',
'm01s05i182': 'tendency_of_specific_humidity_due_to_convection',
'm01s05i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_convection',
'm01s05i184': 'tendency_of_mass_fraction_of_cloud_ice_in_air_due_to_convection',
'm01s05i185': 'tendency_of_eastward_wind_due_to_convection',
'm01s05i186': 'tendency_of_northward_wind_due_to_convection',
'm01s05i192': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_convection',
'm01s05i193': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_convection',
'm01s05i194': 'tendency_of_ice_cloud_area_fraction_in_atmosphere_layer_due_to_convection',
'm01s06i181': 'tendency_of_air_temperature_due_to_gravity_wave_drag',
'm01s06i185': 'tendency_of_eastward_wind_due_to_gravity_wave_drag',
'm01s06i186': 'tendency_of_northward_wind_due_to_gravity_wave_drag',
'm01s09i181': 'tendency_of_air_temperature_due_to_large_scale_cloud',
'm01s09i182': 'tendency_of_specific_humidity_due_to_large_scale_cloud',
'm01s09i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_large_scale_cloud',
'm01s10i181': 'tendency_of_air_temperature_due_to_solver',
'm01s10i185': 'tendency_of_eastward_wind_due_to_solver',
'm01s10i186': 'tendency_of_northward_wind_due_to_solver',
'm01s10i187': 'tendency_of_upward_air_velocity_due_to_solver',
'm01s12i181': 'tendency_of_air_temperature_due_to_advection',
'm01s12i182': 'tendency_of_specific_humidity_due_to_advection',
'm01s12i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_advection',
'm01s12i184': 'tendency_of_mass_fraction_of_cloud_ice_in_air_due_to_advection',
'm01s12i185': 'tendency_of_eastward_wind_due_to_advection',
'm01s12i186': 'tendency_of_northward_wind_due_to_advection',
'm01s12i187': 'tendency_of_upward_air_velocity_due_to_advection',
'm01s12i189': 'tendency_of_mass_fraction_of_rain_in_air_due_to_advection',
'm01s12i190': 'tendency_of_mass_fraction_of_graupel_in_air_due_to_advection',
'm01s12i191': 'tendency_of_mass_fraction_of_cloud_ice2_in_air_due_to_advection',
'm01s12i192': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_advection',
'm01s12i193': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_advection',
'm01s12i194': 'tendency_of_ice_cloud_area_fraction_in_atmosphere_layer_due_to_advection',
'm01s12i195': 'tendency_of_humidity_mixing_ratio_due_to_advection',
'm01s12i196': 'tendency_of_cloud_liquid_water_mixing_ratio_due_to_advection',
'm01s12i197': 'tendency_of_cloud_ice_mixing_ratio_due_to_advection',
'm01s12i198': 'tendency_of_rain_mixing_ratio_due_to_advection',
'm01s12i199': 'tendency_of_graupel_mixing_ratio_due_to_advection',
'm01s12i200': 'tendency_of_cloud_ice2_mixing_ratio_due_to_advection',
'm01s12i381': 'tendency_of_air_temperature_due_to_adv_correction',
'm01s12i382': 'tendency_of_specific_humidity_due_to_adv_correction',
'm01s12i383': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_adv_correction',
'm01s12i384': 'tendency_of_mass_fraction_of_cloud_ice_in_air_due_to_adv_correction',
'm01s12i389': 'tendency_of_mass_fraction_of_rain_in_air_due_to_adv_correction',
'm01s12i390': 'tendency_of_mass_fraction_of_cloud_ice2_in_air_due_to_adv_correction',
'm01s12i391': 'tendency_of_mass_fraction_of_graupel_in_air_due_to_adv_correction',
'm01s12i395': 'tendency_of_humidity_mixing_ratio_due_to_adv_correction',
'm01s12i396': 'tendency_of_cloud_liquid_water_mixing_ratio_due_to_adv_correction',
'm01s12i397': 'tendency_of_cloud_ice_mixing_ratio_due_to_adv_correction',
'm01s12i398': 'tendency_of_rain_mixing_ratio_due_to_adv_correction',
'm01s12i399': 'tendency_of_graupel_mixing_ratio_due_to_adv_correction',
'm01s12i400': 'tendency_of_cloud_ice2_mixing_ratio_due_to_adv_correction',
'm01s13i181': 'tendency_of_air_temperature_due_to_diffusion',
'm01s13i182': 'tendency_of_specific_humidity_due_to_diffusion',
'm01s13i185': 'tendency_of_eastward_wind_due_to_diffusion',
'm01s13i186': 'tendency_of_northward_wind_due_to_diffusion',
'm01s13i187': 'tendency_of_upward_air_velocity_due_to_diffusion',
'm01s14i181': 'tendency_of_air_temperature_due_to_energy_correction',
'm01s16i004': 'air_temperature',
'm01s16i161': 'tendency_of_air_temperature_due_to_pc2_initialisation',
'm01s16i162': 'tendency_of_specific_humidity_due_to_pc2_initialisation',
'm01s16i163': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_pc2_initialisation',
'm01s16i164': 'tendency_of_mass_fraction_of_cloud_ice_in_air_due_to_pc2_initialisation',
'm01s16i172': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_initialisation',
'm01s16i173': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_initialisation',
'm01s16i174': 'tendency_of_ice_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_initialisation',
'm01s16i181': 'tendency_of_air_temperature_due_to_pc2_pressure_change',
'm01s16i182': 'tendency_of_specific_humidity_due_to_pc2_pressure_change',
'm01s16i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air_due_to_pc2_pressure_change',
'm01s16i184': 'tendency_of_mass_fraction_of_cloud_ice_in_air_due_to_pc2_pressure_change',
'm01s16i192': 'tendency_of_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_pressure_change',
'm01s16i193': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_pressure_change',
'm01s16i194': 'tendency_of_ice_cloud_area_fraction_in_atmosphere_layer_due_to_pc2_pressure_change',
'm01s30i181': 'tendency_of_air_temperature',
'm01s30i182': 'tendency_of_specific_humidity',
'm01s30i183': 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air',
'm01s30i184': 'tendency_of_mass_fraction_of_cloud_ice_in_air',
'm01s30i185': 'tendency_of_eastward_wind',
'm01s30i186': 'tendency_of_northward_wind',
'm01s30i187': 'tendency_of_upward_air_velocity',
'm01s30i189': 'tendency_of_mass_fraction_of_rain_in_air',
'm01s30i190': 'tendency_of_mass_fraction_of_cloud_ice2_in_air',
'm01s30i191': 'tendency_of_mass_fraction_of_graupel_in_air',
'm01s30i192': 'tendency_of_cloud_area_fraction_in_atmosphere_layer',
'm01s30i193': 'tendency_of_liquid_water_cloud_area_fraction_in_atmosphere_layer',
'm01s30i194': 'tendency_of_ice_cloud_area_fraction_in_atmosphere_layer',
'm01s30i195': 'tendency_of_humidity_mixing_ratio',
'm01s30i196': 'tendency_of_cloud_liquid_water_mixing_ratio',
'm01s30i197': 'tendency_of_cloud_ice_mixing_ratio',
'm01s30i198': 'tendency_of_rain_mixing_ratio',
'm01s30i199': 'tendency_of_graupel_mixing_ratio',
'm01s30i200': 'tendency_of_cloud_ice2_mixing_ratio',
'm01s30i901': 'tendency_of_air_potential_temperature',
'm01s30i902': 'tendency_of_virtual_potential_temperature',
'm01s30i903': 'tendency_of_air_density',
'm01s35i003': 'tendency_of_eastward_wind_due_to_skeb',
'm01s35i004': 'tendency_of_northward_wind_due_to_skeb',
'm01s35i029': 'tendency_of_air_temperature_due_to_spt',
'm01s35i025': 'tendency_of_specific_humidity_due_to_spt',
'm01s35i026': 'tendency_of_eastward_wind_due_to_spt',
'm01s35i027': 'tendency_of_northward_wind_due_to_spt',
}

# Invert the above
CF_STDNAME_STASH = {}
for (k, v) in STASH_CF_STDNAME.items():
    CF_STDNAME_STASH[v] = k

# Define moisture contour levels
Q_LEVS = np.array([-5e-4, -2e-4, -1e-4, -5e-5, -2e-5, -1e-5,
                   -5e-6, -2e-6, -1e-6, -5e-7, -2e-7, -1e-7,
                   -5e-8, -2e-8, -1e-8, 0, 1e-8, 2e-8, 5e-8,
                   1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6,
                   1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4])

# Define prognostic fields to plot
PROGNOSTICS = {
'air_temperature': dict(stash='m01s16i004', label='T', units='K', levels=np.linspace(-5.0, 5.0, 21)),
'specific_humidity': dict(stash='m01s00i010', label='Q', units='kg kg-1', levels=Q_LEVS),
'mass_fraction_of_cloud_liquid_water_in_air': dict(stash='m01s00i254', label='QCL', units='kg kg-1', levels=Q_LEVS),
'mass_fraction_of_cloud_ice_in_air': dict(stash='m01s00i012', label='QCF', units='kg kg-1', levels=Q_LEVS),
'eastward_wind': dict(stash='m01s00i002', label='U', units='m s-1', levels=np.linspace(-1.0, 1.0, 21)),
'northward_wind': dict(stash='m01s00i003', label='V', units='m s-1', levels=np.linspace(-1.0, 1.0, 21)),
'upward_air_velocity': dict(stash='m01s00i150', label='W', units='m s-1', levels=np.linspace(-10.0, 10.0, 21)),
'mass_fraction_of_rain_in_air': dict(stash='m01s00i272', label='QRAIN', units='kg kg-1', levels=Q_LEVS),
'mass_fraction_of_graupel_in_air': dict(stash='m01s00i273', label='QGRAUP', units='kg kg-1', levels=Q_LEVS),
'mass_fraction_of_cloud_ice2_in_air': dict(stash='m01s00i271', label='QCF2', units='kg kg-1', levels=Q_LEVS),
'cloud_area_fraction_in_atmosphere_layer': dict(stash='m01s00i266', label='BCF', units='1', levels=np.linspace(-1.0, 1.0, 21)),
'liquid_water_cloud_area_fraction_in_atmosphere_layer': dict(stash='m01s00i267', label='LCF', units='1', levels=np.linspace(-1.0, 1.0, 21)),
'ice_cloud_area_fraction_in_atmosphere_layer': dict(stash='m01s00i268', label='FCF', units='1', levels=np.linspace(-1.0, 1.0, 21)),
'humidity_mixing_ratio': dict(stash='m01s00i391', label='MV', units='kg kg-1', levels=Q_LEVS),
'cloud_liquid_water_mixing_ratio': dict(stash='m01s00i392', label='MCL', units='kg kg-1', levels=Q_LEVS),
'cloud_ice_mixing_ratio': dict(stash='m01s00i393', label='MCF', units='kg kg-1', levels=Q_LEVS),
'rain_mixing_ratio': dict(stash='m01s00i394', label='MRAIN', units='kg kg-1', levels=Q_LEVS),
'graupel_mixing_ratio': dict(stash='m01s00i395', label='MGRAUP', units='kg kg-1', levels=Q_LEVS),
'cloud_ice2_mixing_ratio': dict(stash='m01s00i396', label='MCF2', units='kg kg-1', levels=Q_LEVS),
'air_potential_temperature': dict(stash='m01s00i004', label='THETA', units='K', levels=np.linspace(-5.0, 5.0, 21)),
'virtual_potential_temperature': dict(stash='m01s00i388', label='THETAV', units='K', levels=np.linspace(-5.0, 5.0, 21)),
'air_density': dict(stash='m01s00i389', label='RHO', units='kg m-3', levels=np.linspace(-0.05, 0.05, 21)),
}

# Define which sections go in which derived totals
SEC_AP1 = ['shortwave_heating', 'longwave_heating', 'stratiform_precipitation', 'gravity_wave_drag', 'methane_oxidation', 'pc2_checks', 'energy_correction']
SEC_AP2 = ['boundary_layer_mixing', 'convection', 'large_scale_cloud', 'spt'] #, 'skeb']
SEC_AP3 = ['pc2_initialisation', 'pc2_pressure_change']
SEC_DYN = ['advection', 'solver', 'adv_correction', 'diffusion']
SEC_PHY = SEC_AP1 + SEC_AP2 + SEC_AP3
SEC_TOT = SEC_DYN + SEC_PHY

# Define physics schemes to plot
PHYSICS = {
'derived_model_total': dict(loc=2, label='Derived Model Total', sections=SEC_TOT),
'derived_model_physics': dict(loc=4, label='Derived Physics', sections=SEC_PHY),
'derived_model_dynamics': dict(loc=5, label='Derived Dynamics', sections=SEC_DYN),
'model_dynamics_as_residual': dict(loc=6, label='Dynamics (Total - Physics)'),
'derived_ap1': dict(loc=7, label='Derived Atmos Physics 1', sections=SEC_AP1),
'derived_ap2': dict(loc=8, label='Derived Atmos Physics 2', sections=SEC_AP2),
'derived_ap3': dict(loc=9, label='Derived Atmos Physics 3', sections=SEC_AP3),
'shortwave_heating': dict(loc=10, label='SW Radiation', fname='swr'),
'longwave_heating': dict(loc=11, label='LW Radiation', fname='lwr'),
'stratiform_precipitation': dict(loc=12, label='LS Precipitation', fname='lsp'),
'gravity_wave_drag': dict(loc=13, label='GWD', fname='gwd'),
'methane_oxidation': dict(loc=14, label='Methane Oxidation', fname='mox'),
'energy_correction': dict(loc=15, label='Energy Correction', fname='ecor'),
'pc2_checks': dict(loc=16, label='PC2 Checks', fname='pc2c'),
'pc2_initialisation': dict(loc=17, label='PC2 Initialisation', fname='pc2i'),
'pc2_pressure_change': dict(loc=18, label='PC2 Pressure Change', fname='pc2p'),
'boundary_layer_mixing': dict(loc=19, label='Boundary Layer', fname='bl'),
'convection': dict(loc=20, label='Convection', fname='cu'),
'advection': dict(loc=21, label='Advection', fname='adv'),
'skeb': dict(loc=22, label='SKEB2', fname='skeb'),
'spt': dict(loc=23, label='SPT', fname='spt'),
'solver': dict(loc=24, label='Solver', fname='slv'),
'adv_correction': dict(loc=25, label='Advection Correction', fname='advcor'),
'diffusion': dict(loc=26, label='Diffusion', fname='diff'),
'large_scale_cloud': dict(loc=27, label='Large Scale Cloud', fname='lsc'),
}

# Timescale for increment rates
TIMESCALE = 'day'


def msi_stashcode(lbuser):
    'Create MSI stashcode from pp header entry lbuser'
    stdname = 'm{0:02d}s{1[0]:02d}i{1[1]:03d}'
    return stdname.format(lbuser[6], divmod(lbuser[3], 1000))


def split_cf_stdname(name):
    'Split up name in the form tendency_of_<prog>[_due_to_<phys>]'
    sname = name.replace('tendency_of_', '').split('_due_to_')
    if len(sname) == 1:
        sname.append(None)
    return tuple(sname)


def join_cf_stdname(prog, phys=None):
    '''
    Join up prognostic name and optional physics scheme in the form
    tendency_of_<prog>[_due_to_<phys>]
    '''
    if phys:
        name = 'tendency_of_{0}_due_to_{1}'.format(prog, phys)
    else:
        name = 'tendency_of_{0}'.format(prog)
    return name


def inc_callback(cube, field, filename):
    'Iris callback to fix names and units of all increment diagnostics'
    # Ignore orography
    if cube.name() != 'surface_altitude':
        # If stash code not in list above then do not accept
        if msi_stashcode(field.lbuser) in STASH_CF_STDNAME:
            stdname = STASH_CF_STDNAME[msi_stashcode(field.lbuser)]
            cube.rename(stdname)
            (prog, phys) = split_cf_stdname(stdname)
            cube.units = cf_units.Unit(PROGNOSTICS[prog]['units'])
        else:
            raise iris.exceptions.IgnoreCubeException


def sumcubes(cubes, name):
    'Sum list of cubes into one cube'
    sumcube = None
    (prog, phys) = split_cf_stdname(name)
    phys_section = PHYSICS[phys]
    if 'sections' in phys_section:
        if len(cubes) > 0:
            sumcube = 0.0
            for cube in cubes:
                (prog, phys) = split_cf_stdname(cube.name())
                if phys in phys_section['sections']:
                    sumcube += cube.copy()
            if sumcube:
                sumcube.rename(name)
    return sumcube


def area_rms(cube, vert=None):
    'Routine to calculate global area-weighted RMS fields'
    cube.coord('longitude').guess_bounds()
    cube.coord('latitude').guess_bounds()
    weights = iac.area_weights(cube)
    meansq = (cube*cube).collapsed(['time', 'longitude', 'latitude'],
                                   iris.analysis.MEAN,
                                   weights=weights)
    # Do vertical average if requested
    if vert:
        meansq = meansq.collapsed(vert, iris.analysis.MEAN)
    return np.sqrt(meansq.data)


def plot_title(head, prog, phys):
    'Determine plot title'
    proglab = PROGNOSTICS[prog]['label']
    title = 'd{} '.format(proglab)
    if head == 1:
        title += 'Total Increment'
    elif head == 3:
        title += 'Residual (Total - Sum)'
    elif head == 1002:
        title += 'Time Diff (End - Begin)'
    elif head == 1003:
        title += 'Residual (Total - Diff)'
    else:
        title += PHYSICS[phys]['label']
    return title


def plot_loc(head, phys):
    'Determine plot location'
    if head:
        loc = head
    else:
        loc = PHYSICS[phys]['loc']
    return loc


def plot_suptitle(prog, resrms):
    proginfo = PROGNOSTICS[prog]
    latex_unit = unit_latex(proginfo['units'])
    latex_unit += '{{\cdot}}{0}^{{-1}}'.format(TIMESCALE)
    suptitle = r'{0} Increment Budget (${1}$)'
    suptitle = suptitle.format(proginfo['label'], latex_unit)
    if resrms is not None:
        suptitle += '\n\nNormalised Residual RMS = {0:.3e}'.format(resrms)
    return suptitle


def calc_deltat(coord):
    if coord.units.is_time_reference():
        diff = coord.bounds[0, 1] - coord.bounds[0, 0]
        unit = str(coord.units).split(' ')[0]
    return '{0:f} {1}'.format(diff, unit)


def modify_time_unit(cube):
    deltat = calc_deltat(cube.coord('time'))
    field_unit = str(cube.units)
    if cube.units != '1':
        cube.units = '{0} ({1})-1'.format(field_unit, deltat)
        cube.convert_units('{0} {1}-1'.format(field_unit, TIMESCALE))
    else:
        cube.units = '({0})-1'.format(deltat)
        cube.convert_units('{0}-1'.format(TIMESCALE))


def plot_increment(cube, rows=0, columns=0, head=0, residual=False,
                   fixed=False, hgt=False):
    'Plot zonal mean increment'

    # Set coordinates to plot against
    if hgt:
        coords = ['latitude', 'level_height']
        ylabel = 'Level Height (km)'
    else:
        coords = ['latitude', 'model_level_number']
        ylabel = 'Model Level'

    # Make alterations to cube to allow plotting
    if cube.coords('surface_altitude'):
        cube.remove_coord('surface_altitude')
    if cube.coords('level_height'):
        cube.coord('level_height').convert_units('km')

    # Create time and zonal mean to plot
    pcube = cube.collapsed(['time', 'longitude'], iris.analysis.MEAN)
    modify_time_unit(pcube)

    # Determine prognostic and physics scheme
    (prog, phys) = split_cf_stdname(cube.name())

    # Create plot axes
    ax = plt.subplot(rows, columns, plot_loc(head, phys))

    # Plot cube with fixed levels or self-determined
    cmap = mpl_cm.get_cmap('seismic')
    if fixed:
        levels = PROGNOSTICS[prog]['levels']
        norm = mpl_col.BoundaryNorm(levels, cmap.N)
        cf = iplt.contourf(pcube, coords=coords, extend='both',
                           cmap=cmap, norm=norm, levels=levels)
    else:
        cf = iplt.contourf(pcube, coords=coords, extend='both',
                           cmap=cmap)

    # Add colorbar
    cb = plt.colorbar(cf, orientation='vertical', extend='both')

    # Set plot niceties
    ax.set_title(plot_title(head, prog, phys))
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(cmgridl.LATITUDE_FORMATTER)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])

    # If all values are zero then plaster a big "= 0" on plot
    if not cube.data.any():
        ax.text(0.5, 0.5, 'ZERO', ha='center', va='center',
                transform=ax.transAxes, fontsize='xx-large',
                bbox=dict(edgecolor='black', facecolor='yellow'))


def unit_latex(unit):
    ''' Convert cf_units.Unit object into a latex consumable string '''
    def conv_latex(unitstr, regexp):
        match = regexp.match(unitstr)
        if match:
            # group(0) is matching string
            # group(2) is caret if it exists
            (unit, power) = match.group(1, 3)
            # Replace micro with \mu, only prefix that requires this
            if unit.startswith('micro'):
                newunitstr = '{\mu}{' + unit.replace('micro','') + '}'
            else:
                newunitstr = '{' + unit + '}'
            # Add power to string if it exists
            if power:
                newunitstr += '^{' + power + '}'
        else:
            # Return original string if it doesn't match regexp
            newunitstr = '{' + unitstr + '}'
        return newunitstr
    # Must use str(unit) here as unit.definition resolves units to
    # closest form, e.g. kg kg^-1 => 1
    re_unit = re.compile('(^[^-+\d\^]+)([\^]?)([-+]?[\d]*)$')
    latex_str = [conv_latex(unit, re_unit) for unit in str(unit).split()]
    # Join using a central dot
    return '{\cdot}'.join(latex_str)

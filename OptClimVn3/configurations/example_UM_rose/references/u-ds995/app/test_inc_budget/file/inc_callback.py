# *****************************COPYRIGHT*******************************
# (C) Crown copyright Met Office. All rights reserved.
# For further details please refer to the file COPYRIGHT.txt
# which you should have received as part of this distribution.
# *****************************COPYRIGHT*******************************

"""
Module to define callbacks that set CF standard names and units for STASH codes

This should not be required if all the correct mappings are in place in Iris
"""

import re

import cf_units

import iris

# Define all units for prognostic fields
UNITS = {
    'air_temperature': 'K',
    'specific_humidity': 'kg kg-1',
    'mass_fraction_of_cloud_liquid_water_in_air': 'kg kg-1',
    'mass_fraction_of_cloud_ice_in_air': 'kg kg-1',
    'eastward_wind': 'm s-1',
    'northward_wind': 'm s-1',
    'upward_air_velocity': 'm s-1',
    'mass_fraction_of_rain_in_air': 'kg kg-1',
    'mass_fraction_of_graupel_in_air': 'kg kg-1',
    'mass_fraction_of_cloud_ice2_in_air': 'kg kg-1',
    'cloud_volume_fraction_in_atmosphere_layer': '1',
    'liquid_water_cloud_volume_fraction_in_atmosphere_layer': '1',
    'ice_cloud_volume_fraction_in_atmosphere_layer': '1',
    'humidity_mixing_ratio': 'kg kg-1',
    'cloud_liquid_water_mixing_ratio': 'kg kg-1',
    'cloud_ice_mixing_ratio': 'kg kg-1',
    'rain_mixing_ratio': 'kg kg-1',
    'graupel_mixing_ratio': 'kg kg-1',
    'cloud_ice2_mixing_ratio': 'kg kg-1',
    'air_potential_temperature': 'K',
    'virtual_potential_temperature': 'K',
    'air_density': 'kg m-3',
}

# Define stash codes that can be specific or mixing ratio
MIXING_RATIO_SWITCH = {
    'm01s01i182': 'change_over_time_in_humidity_mixing_ratio_due_to_shortwave_heating',
    'm01s01i183': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_shortwave_heating',
    'm01s02i182': 'change_over_time_in_humidity_mixing_ratio_due_to_longwave_heating',
    'm01s02i183': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_longwave_heating',
    'm01s03i182': 'change_over_time_in_humidity_mixing_ratio_due_to_boundary_layer_mixing',
    'm01s03i183': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_boundary_layer_mixing',
    'm01s03i184': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_boundary_layer_mixing',
    'm01s04i142': 'change_over_time_in_humidity_mixing_ratio_due_to_pc2_checks',
    'm01s04i143': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_pc2_checks',
    'm01s04i144': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_pc2_checks',
    'm01s04i182': 'change_over_time_in_humidity_mixing_ratio_due_to_stratiform_precipitation',
    'm01s04i183': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_stratiform_precipitation',
    'm01s04i184': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_stratiform_precipitation',
    'm01s04i189': 'change_over_time_in_rain_mixing_ratio_due_to_stratiform_precipitation',
    'm01s04i190': 'change_over_time_in_cloud_ice2_mixing_ratio_due_to_stratiform_precipitation',
    'm01s04i191': 'change_over_time_in_graupel_mixing_ratio_due_to_stratiform_precipitation',
    'm01s04i882': 'change_over_time_in_humidity_mixing_ratio_due_to_parallel_physics',
    'm01s04i883': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_parallel_physics',
    'm01s04i884': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_parallel_physics',
    'm01s04i889': 'change_over_time_in_rain_mixing_ratio_due_to_parallel_physics',
    'm01s04i890': 'change_over_time_in_cloud_ice2_mixing_ratio_due_to_parallel_physics',
    'm01s04i891': 'change_over_time_in_graupel_mixing_ratio_due_to_parallel_physics',
    'm01s04i982': 'change_over_time_in_humidity_mixing_ratio_due_to_methane_oxidation',
    'm01s05i182': 'change_over_time_in_humidity_mixing_ratio_due_to_convection',
    'm01s05i183': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_convection',
    'm01s05i184': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_convection',
    'm01s09i182': 'change_over_time_in_humidity_mixing_ratio_due_to_large_scale_cloud',
    'm01s09i183': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_large_scale_cloud',
    'm01s16i162': 'change_over_time_in_humidity_mixing_ratio_due_to_pc2_initialisation',
    'm01s16i163': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_pc2_initialisation',
    'm01s16i164': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_pc2_initialisation',
    'm01s16i182': 'change_over_time_in_humidity_mixing_ratio_due_to_pc2_pressure_change',
    'm01s16i183': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_pc2_pressure_change',
    'm01s16i184': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_pc2_pressure_change',
    'm01s35i025': 'change_over_time_in_humidity_mixing_ratio_due_to_stochastic_perturbation_of_tendencies',
    'm01s53i182': 'change_over_time_in_humidity_mixing_ratio_due_to_idealised',
}

# Define all increment stashcodes CF standard names
STASH_CF_STDNAME = {
    'm01s00i002': 'eastward_wind',
    'm01s00i003': 'northward_wind',
    'm01s00i004': 'air_potential_temperature',
    'm01s00i010': 'specific_humidity',
    'm01s00i012': 'mass_fraction_of_cloud_ice_in_air',
    'm01s00i150': 'upward_air_velocity',
    'm01s00i254': 'mass_fraction_of_cloud_liquid_water_in_air',
    'm01s00i266': 'cloud_volume_fraction_in_atmosphere_layer',
    'm01s00i267': 'liquid_water_cloud_volume_fraction_in_atmosphere_layer',
    'm01s00i268': 'ice_cloud_volume_fraction_in_atmosphere_layer',
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
    'm01s01i181': 'change_over_time_in_air_temperature_due_to_shortwave_heating',
    'm01s01i182': 'change_over_time_in_specific_humidity_due_to_shortwave_heating',
    'm01s01i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_shortwave_heating',
    'm01s01i192': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_shortwave_heating',
    'm01s01i193': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_shortwave_heating',
    'm01s02i181': 'change_over_time_in_air_temperature_due_to_longwave_heating',
    'm01s02i182': 'change_over_time_in_specific_humidity_due_to_longwave_heating',
    'm01s02i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_longwave_heating',
    'm01s02i192': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_longwave_heating',
    'm01s02i193': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_longwave_heating',
    'm01s03i181': 'change_over_time_in_air_temperature_due_to_boundary_layer_mixing',
    'm01s03i182': 'change_over_time_in_specific_humidity_due_to_boundary_layer_mixing',
    'm01s03i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_boundary_layer_mixing',
    'm01s03i184': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_boundary_layer_mixing',
    'm01s03i185': 'change_over_time_in_eastward_wind_due_to_boundary_layer_mixing',
    'm01s03i186': 'change_over_time_in_northward_wind_due_to_boundary_layer_mixing',
    'm01s03i187': 'change_over_time_in_upward_air_velocity_due_to_boundary_layer_mixing',
    'm01s03i192': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_boundary_layer_mixing',
    'm01s03i193': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_boundary_layer_mixing',
    'm01s03i194': 'change_over_time_in_ice_cloud_volume_fraction_in_atmosphere_layer_due_to_boundary_layer_mixing',
    'm01s04i141': 'change_over_time_in_air_temperature_due_to_pc2_checks',
    'm01s04i142': 'change_over_time_in_specific_humidity_due_to_pc2_checks',
    'm01s04i143': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_pc2_checks',
    'm01s04i144': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_pc2_checks',
    'm01s04i152': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_checks',
    'm01s04i153': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_checks',
    'm01s04i154': 'change_over_time_in_ice_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_checks',
    'm01s04i181': 'change_over_time_in_air_temperature_due_to_stratiform_precipitation',
    'm01s04i182': 'change_over_time_in_specific_humidity_due_to_stratiform_precipitation',
    'm01s04i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_stratiform_precipitation',
    'm01s04i184': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_stratiform_precipitation',
    'm01s04i189': 'change_over_time_in_mass_fraction_of_rain_in_air_due_to_stratiform_precipitation',
    'm01s04i190': 'change_over_time_in_mass_fraction_of_cloud_ice2_in_air_due_to_stratiform_precipitation',
    'm01s04i191': 'change_over_time_in_mass_fraction_of_graupel_in_air_due_to_stratiform_precipitation',
    'm01s04i192': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_stratiform_precipitation',
    'm01s04i193': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_stratiform_precipitation',
    'm01s04i194': 'change_over_time_in_ice_cloud_volume_fraction_in_atmosphere_layer_due_to_stratiform_precipitation',
    'm01s04i880': 'change_over_time_in_air_potential_temperature_due_to_parallel_physics',
    'm01s04i881': 'change_over_time_in_air_temperature_due_to_parallel_physics',
    'm01s04i882': 'change_over_time_in_specific_humidity_due_to_parallel_physics',
    'm01s04i883': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_parallel_physics',
    'm01s04i884': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_parallel_physics',
    'm01s04i885': 'change_over_time_in_eastward_wind_due_to_parallel_physics',
    'm01s04i886': 'change_over_time_in_northward_wind_due_to_parallel_physics',
    'm01s04i889': 'change_over_time_in_mass_fraction_of_rain_in_air_due_to_parallel_physics',
    'm01s04i890': 'change_over_time_in_mass_fraction_of_cloud_ice2_in_air_due_to_parallel_physics',
    'm01s04i891': 'change_over_time_in_mass_fraction_of_graupel_in_air_due_to_parallel_physics',
    'm01s04i892': 'change_over_time_in_cloud_area_fraction_in_atmosphere_layer_due_to_parallel_physics',
    'm01s04i893': 'change_over_time_in_liquid_water_cloud_area_fraction_in_atmosphere_layer_due_to_parallel_physics',
    'm01s04i894': 'change_over_time_in_ice_cloud_area_fraction_in_atmosphere_layer_due_to_parallel_physics',
    'm01s04i982': 'change_over_time_in_specific_humidity_due_to_methane_oxidation',
    'm01s05i181': 'change_over_time_in_air_temperature_due_to_convection',
    'm01s05i182': 'change_over_time_in_specific_humidity_due_to_convection',
    'm01s05i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_convection',
    'm01s05i184': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_convection',
    'm01s05i185': 'change_over_time_in_eastward_wind_due_to_convection',
    'm01s05i186': 'change_over_time_in_northward_wind_due_to_convection',
    'm01s05i192': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_convection',
    'm01s05i193': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_convection',
    'm01s05i194': 'change_over_time_in_ice_cloud_volume_fraction_in_atmosphere_layer_due_to_convection',
    'm01s06i181': 'change_over_time_in_air_temperature_due_to_gravity_wave_drag',
    'm01s06i185': 'change_over_time_in_eastward_wind_due_to_gravity_wave_drag',
    'm01s06i186': 'change_over_time_in_northward_wind_due_to_gravity_wave_drag',
    'm01s09i181': 'change_over_time_in_air_temperature_due_to_large_scale_cloud',
    'm01s09i182': 'change_over_time_in_specific_humidity_due_to_large_scale_cloud',
    'm01s09i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_large_scale_cloud',
    'm01s10i181': 'change_over_time_in_air_temperature_due_to_pressure_solver',
    'm01s10i185': 'change_over_time_in_eastward_wind_due_to_pressure_solver',
    'm01s10i186': 'change_over_time_in_northward_wind_due_to_pressure_solver',
    'm01s10i187': 'change_over_time_in_upward_air_velocity_due_to_pressure_solver',
    'm01s12i181': 'change_over_time_in_air_temperature_due_to_advection',
    'm01s12i182': 'change_over_time_in_specific_humidity_due_to_advection',
    'm01s12i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_advection',
    'm01s12i184': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_advection',
    'm01s12i185': 'change_over_time_in_eastward_wind_due_to_advection',
    'm01s12i186': 'change_over_time_in_northward_wind_due_to_advection',
    'm01s12i187': 'change_over_time_in_upward_air_velocity_due_to_advection',
    'm01s12i189': 'change_over_time_in_mass_fraction_of_rain_in_air_due_to_advection',
    'm01s12i190': 'change_over_time_in_mass_fraction_of_graupel_in_air_due_to_advection',
    'm01s12i191': 'change_over_time_in_mass_fraction_of_cloud_ice2_in_air_due_to_advection',
    'm01s12i192': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_advection',
    'm01s12i193': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_advection',
    'm01s12i194': 'change_over_time_in_ice_cloud_volume_fraction_in_atmosphere_layer_due_to_advection',
    'm01s12i195': 'change_over_time_in_humidity_mixing_ratio_due_to_advection',
    'm01s12i196': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_advection',
    'm01s12i197': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_advection',
    'm01s12i198': 'change_over_time_in_rain_mixing_ratio_due_to_advection',
    'm01s12i199': 'change_over_time_in_graupel_mixing_ratio_due_to_advection',
    'm01s12i200': 'change_over_time_in_cloud_ice2_mixing_ratio_due_to_advection',
    'm01s12i381': 'change_over_time_in_air_temperature_due_to_advection_corrections',
    'm01s12i382': 'change_over_time_in_specific_humidity_due_to_advection_corrections',
    'm01s12i383': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_advection_corrections',
    'm01s12i384': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_advection_corrections',
    'm01s12i389': 'change_over_time_in_mass_fraction_of_rain_in_air_due_to_advection_corrections',
    'm01s12i390': 'change_over_time_in_mass_fraction_of_cloud_ice2_in_air_due_to_advection_corrections',
    'm01s12i391': 'change_over_time_in_mass_fraction_of_graupel_in_air_due_to_advection_corrections',
    'm01s12i395': 'change_over_time_in_humidity_mixing_ratio_due_to_advection_corrections',
    'm01s12i396': 'change_over_time_in_cloud_liquid_water_mixing_ratio_due_to_advection_corrections',
    'm01s12i397': 'change_over_time_in_cloud_ice_mixing_ratio_due_to_advection_corrections',
    'm01s12i398': 'change_over_time_in_rain_mixing_ratio_due_to_advection_corrections',
    'm01s12i399': 'change_over_time_in_graupel_mixing_ratio_due_to_advection_corrections',
    'm01s12i400': 'change_over_time_in_cloud_ice2_mixing_ratio_due_to_advection_corrections',
    'm01s13i181': 'change_over_time_in_air_temperature_due_to_diffusion',
    'm01s13i182': 'change_over_time_in_specific_humidity_due_to_diffusion',
    'm01s13i185': 'change_over_time_in_eastward_wind_due_to_diffusion',
    'm01s13i186': 'change_over_time_in_northward_wind_due_to_diffusion',
    'm01s13i187': 'change_over_time_in_upward_air_velocity_due_to_diffusion',
    'm01s14i181': 'change_over_time_in_air_temperature_due_to_energy_correction',
    'm01s16i004': 'air_temperature',
    'm01s16i161': 'change_over_time_in_air_temperature_due_to_pc2_initialisation',
    'm01s16i162': 'change_over_time_in_specific_humidity_due_to_pc2_initialisation',
    'm01s16i163': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_pc2_initialisation',
    'm01s16i164': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_pc2_initialisation',
    'm01s16i172': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_initialisation',
    'm01s16i173': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_initialisation',
    'm01s16i174': 'change_over_time_in_ice_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_initialisation',
    'm01s16i181': 'change_over_time_in_air_temperature_due_to_pc2_pressure_change',
    'm01s16i182': 'change_over_time_in_specific_humidity_due_to_pc2_pressure_change',
    'm01s16i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_pc2_pressure_change',
    'm01s16i184': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_pc2_pressure_change',
    'm01s16i192': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_pressure_change',
    'm01s16i193': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_pressure_change',
    'm01s16i194': 'change_over_time_in_ice_cloud_volume_fraction_in_atmosphere_layer_due_to_pc2_pressure_change',
    'm01s30i181': 'change_over_time_in_air_temperature',
    'm01s30i182': 'change_over_time_in_specific_humidity',
    'm01s30i183': 'change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air',
    'm01s30i184': 'change_over_time_in_mass_fraction_of_cloud_ice_in_air',
    'm01s30i185': 'change_over_time_in_eastward_wind',
    'm01s30i186': 'change_over_time_in_northward_wind',
    'm01s30i187': 'change_over_time_in_upward_air_velocity',
    'm01s30i189': 'change_over_time_in_mass_fraction_of_rain_in_air',
    'm01s30i190': 'change_over_time_in_mass_fraction_of_cloud_ice2_in_air',
    'm01s30i191': 'change_over_time_in_mass_fraction_of_graupel_in_air',
    'm01s30i192': 'change_over_time_in_cloud_volume_fraction_in_atmosphere_layer',
    'm01s30i193': 'change_over_time_in_liquid_water_cloud_volume_fraction_in_atmosphere_layer',
    'm01s30i194': 'change_over_time_in_ice_cloud_volume_fraction_in_atmosphere_layer',
    'm01s30i195': 'change_over_time_in_humidity_mixing_ratio',
    'm01s30i196': 'change_over_time_in_cloud_liquid_water_mixing_ratio',
    'm01s30i197': 'change_over_time_in_cloud_ice_mixing_ratio',
    'm01s30i198': 'change_over_time_in_rain_mixing_ratio',
    'm01s30i199': 'change_over_time_in_graupel_mixing_ratio',
    'm01s30i200': 'change_over_time_in_cloud_ice2_mixing_ratio',
    'm01s30i901': 'change_over_time_in_air_potential_temperature',
    'm01s30i902': 'change_over_time_in_virtual_potential_temperature',
    'm01s30i903': 'change_over_time_in_air_density',
    'm01s35i003': 'change_over_time_in_eastward_wind_due_to_stochastic_kinetic_energy_backscatter',
    'm01s35i004': 'change_over_time_in_northward_wind_due_to_stochastic_kinetic_energy_backscatter',
    'm01s35i024': 'change_over_time_in_air_potential_temperature_due_to_stochastic_perturbation_of_tendencies',
    'm01s35i025': 'change_over_time_in_specific_humidity_due_to_stochastic_perturbation_of_tendencies',
    'm01s35i026': 'change_over_time_in_eastward_wind_due_to_stochastic_perturbation_of_tendencies',
    'm01s35i027': 'change_over_time_in_northward_wind_due_to_stochastic_perturbation_of_tendencies',
    'm01s35i029': 'change_over_time_in_air_temperature_due_to_stochastic_perturbation_of_tendencies',
    'm01s53i181': 'change_over_time_in_air_temperature_due_to_idealised',
    'm01s53i182': 'change_over_time_in_specific_humidity_due_to_idealised',
    'm01s53i185': 'change_over_time_in_eastward_wind_due_to_idealised',
    'm01s53i186': 'change_over_time_in_northward_wind_due_to_idealised',
}


def msi_stashcode(lbuser):
    ''' Create MSI stashcode from pp header entry lbuser '''
    (model, (section, item)) = (lbuser[6], divmod(lbuser[3], 1000))
    return f'm{model:02d}s{section:02d}i{item:03d}'


def split_cf_stdname(name):
    ''' Split up increment standard name into prognostic and physics scheme '''
    inc_regexp = r'^(?P<type>change_over_time_in_|tendency_of_|)(?P<prognostic>\w+?)(?:_due_to_(?P<scheme>\w+))?$'
    return re.match(inc_regexp, name).group('type', 'prognostic', 'scheme')


def inc_callback(cube, field, filename):
    '''
    Call back routine to make sure we only allow known fields from list
    STASH_CF_STDNAME. Also make sure all fields have been renamed to either
    CF-Compliant names and units, or something close!

    This includes all known prognostic fields, and their parameterisation
    increments. Orography (surface_altitude) is an exception as it will always
    be allowed through.
    '''
    # Ignore orography
    if cube.name() != 'surface_altitude':
        # If stash code not in list above then do not accept
        if msi_stashcode(field.lbuser) in STASH_CF_STDNAME:
            stdname = STASH_CF_STDNAME[msi_stashcode(field.lbuser)]
            cube.rename(stdname)
            (_, prognostic, _) = split_cf_stdname(stdname)
            cube.units = cf_units.Unit(UNITS[prognostic])
        else:
            raise iris.exceptions.IgnoreCubeException


def inc_callback_mr(cube, field, filename):
    '''
    Call back routine to make sure we only allow known fields from list
    STASH_CF_STDNAME. Also make sure all fields have been renamed to either
    CF-Compliant names and units, or something close!

    This includes all known prognostic fields, and their parameterisation
    increments. Orography (surface_altitude) is an exception as it will always
    be allowed through.
    '''
    # Ignore orography
    if cube.name() != 'surface_altitude':
        # If stash code not in list above then do not accept
        if msi_stashcode(field.lbuser) in STASH_CF_STDNAME:
            stdname = STASH_CF_STDNAME[msi_stashcode(field.lbuser)]
            # If STASH code is actually mixing ratio then change name
            if msi_stashcode(field.lbuser) in MIXING_RATIO_SWITCH:
                stdname = MIXING_RATIO_SWITCH[msi_stashcode(field.lbuser)]
            cube.rename(stdname)
            (_, prognostic, _) = split_cf_stdname(stdname)
            cube.units = cf_units.Unit(UNITS[prognostic])
        else:
            raise iris.exceptions.IgnoreCubeException

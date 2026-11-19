# *****************************COPYRIGHT*******************************
# (C) Crown copyright Met Office. All rights reserved.
# For further details please refer to the file COPYRIGHT.txt
# which you should have received as part of this distribution.
# *****************************COPYRIGHT*******************************

"""
Module to allow for the calculation of increments from UM data
"""

import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_col
import numpy as np

import cartopy.crs as ccrs
import cartopy.mpl.gridliner as cmgridl

import iris
import iris.analysis.cartography as iac
import iris.plot as iplt

import inc_callback

# Timescale for increment plotting
DEFAULT_TIMESCALE = 'day'

# Define contour levels
U_LEVS = np.linspace(-10.0, 10.0, 21)
V_LEVS = np.linspace(-10.0, 10.0, 21)
W_LEVS = np.linspace(-0.1, 0.1, 21)
Q_LEVS = np.linspace(-1.0, 1.0, 21)
QC_LEVS = np.linspace(-0.1, 0.1, 21)
CF_LEVS = np.linspace(-1.0, 1.0, 21)
T_LEVS = np.linspace(-5.0, 5.0, 21)
RHO_LEVS = np.linspace(-0.05, 0.05, 21)

# Define prognostic fields to plot
PROGNOSTICS = {
    'air_temperature': dict(stash='m01s16i004', label='T', units='K', levels=T_LEVS, split=60),
    'specific_humidity': dict(stash='m01s00i010', label='Q', units='kg kg-1', punits='g kg-1', levels=Q_LEVS),
    'mass_fraction_of_cloud_liquid_water_in_air': dict(stash='m01s00i254', label='QCL', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'mass_fraction_of_cloud_ice_in_air': dict(stash='m01s00i012', label='QCF', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'eastward_wind': dict(stash='m01s00i002', label='U', units='m s-1', levels=U_LEVS),
    'northward_wind': dict(stash='m01s00i003', label='V', units='m s-1', levels=V_LEVS),
    'upward_air_velocity': dict(stash='m01s00i150', label='W', units='m s-1', levels=W_LEVS),
    'mass_fraction_of_rain_in_air': dict(stash='m01s00i272', label='QRAIN', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'mass_fraction_of_graupel_in_air': dict(stash='m01s00i273', label='QGRAUP', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'mass_fraction_of_cloud_ice2_in_air': dict(stash='m01s00i271', label='QCF2', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'cloud_volume_fraction_in_atmosphere_layer': dict(stash='m01s00i266', label='BCF', units='1', levels=CF_LEVS),
    'liquid_water_cloud_volume_fraction_in_atmosphere_layer': dict(stash='m01s00i267', label='LCF', units='1', levels=CF_LEVS),
    'ice_cloud_volume_fraction_in_atmosphere_layer': dict(stash='m01s00i268', label='FCF', units='1', levels=CF_LEVS),
    'humidity_mixing_ratio': dict(stash='m01s00i391', label='MV', units='kg kg-1', punits='g kg-1', levels=Q_LEVS),
    'cloud_liquid_water_mixing_ratio': dict(stash='m01s00i392', label='MCL', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'cloud_ice_mixing_ratio': dict(stash='m01s00i393', label='MCF', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'rain_mixing_ratio': dict(stash='m01s00i394', label='MRAIN', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'graupel_mixing_ratio': dict(stash='m01s00i395', label='MGRAUP', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'cloud_ice2_mixing_ratio': dict(stash='m01s00i396', label='MCF2', units='kg kg-1', punits='g kg-1', levels=QC_LEVS),
    'air_potential_temperature': dict(stash='m01s00i004', label='THETA', units='K', levels=T_LEVS),
    'virtual_potential_temperature': dict(stash='m01s00i388', label='THETAV', units='K', levels=T_LEVS),
    'air_density': dict(stash='m01s00i389', label='RHO', units='kg m-3', levels=RHO_LEVS),
}

# Define which sections go in which derived totals
SEC_AP1 = ['shortwave_heating', 'longwave_heating', 'stratiform_precipitation', 'gravity_wave_drag', 'methane_oxidation', 'pc2_checks', 'energy_correction']
SEC_AP2 = ['boundary_layer_mixing', 'convection', 'large_scale_cloud', 'stochastic_perturbation_of_tendencies'] #, 'stochastic_kinetic_energy_backscatter']
SEC_AP3 = ['pc2_initialisation', 'pc2_pressure_change']
SEC_DYN = ['advection', 'pressure_solver', 'advection_corrections', 'diffusion']
SEC_PHY = SEC_AP1 + SEC_AP2 + SEC_AP3
SEC_TOT = SEC_DYN + SEC_PHY

# Define physics schemes (and combinations there-of) to plot
# Need to add line plot colours and styles!!!
PHYSICS = {
    None: dict(loc=1, label='Total Increment', fname=None, lines=('black', 'solid')),
    'derived_model_total': dict(loc=2, label='Derived Model Total', fname='tot', lines=('black', 'dotted'), sections=SEC_TOT),
    'total_residual': dict(loc=3, label='Residual (Total - Sum)', fname='res', lines=('black', 'dotted'), scale_levels=0.1, diff=[None, 'derived_model_total']),
    'derived_model_physics': dict(loc=4, label='Derived Physics', fname='phy', lines=('black', 'dashed'), sections=SEC_PHY),
    'derived_model_dynamics': dict(loc=5, label='Derived Dynamics', fname='dyn', lines=('black', 'dotted'), sections=SEC_DYN),
    'model_dynamics_as_residual': dict(loc=6, label='Dynamics (Total - Physics)', fname='resdyn', lines=('black', 'dotted'), diff=[None, 'derived_model_physics']),
    'parallel_physics': dict(loc=7, label='Atmos Physics 1', fname='ap1', lines=('red', 'dashdot')),
    'derived_parallel_physics': dict(loc=8, label='Derived Atmos Physics 1', fname='ap1der', lines=('red', 'dashdot'), sections=SEC_AP1),
    'parallel_physics_residual': dict(loc=9, label='Residual (AP1 - SumAP1)', fname='ap1res', lines=('red', 'dashdot'), scale_levels=0.1, diff=['parallel_physics', 'derived_parallel_physics']),
    'sequential_physics': dict(loc=10, label='Atmos Physics 2', fname='ap2', lines=('blue', 'dashdot')),
    'derived_sequential_physics': dict(loc=11, label='Derived Atmos Physics 2', fname='ap2der', lines=('blue', 'dashdot'), sections=SEC_AP2),
    'sequential_physics_residual': dict(loc=12, label='Residual (AP2 - SumAP2)', fname='ap2res', lines=('blue', 'dashdot'), scale_levels=0.1, diff=['sequential_physics', 'derived_sequential_physics']),
    'further_physics': dict(loc=13, label='Atmos Physics 3', fname='ap3', lines=('limegreen', 'dashdot')),
    'derived_further_physics': dict(loc=14, label='Derived Atmos Physics 3', fname='ap3der', lines=('limegreen', 'dashdot'), sections=SEC_AP3),
    'further_physics_residual': dict(loc=15, label='Residual (AP3 - SumAP3)', fname='ap3res', lines=('limegreen', 'dashdot'), scale_levels=0.1, diff=['further_physics', 'derived_further_physics']),
    'shortwave_heating': dict(loc=16, label='SW Radiation', fname='swr', lines=('red', 'solid')),
    'longwave_heating': dict(loc=17, label='LW Radiation', fname='lwr', lines=('blue', 'solid')),
    'stratiform_precipitation': dict(loc=18, label='LS Precipitation', fname='lsp', lines=('limegreen', 'solid')),
    'gravity_wave_drag': dict(loc=19, label='GWD', fname='gwd', lines=('yellow', 'solid')),
    'methane_oxidation': dict(loc=20, label='Methane Oxidation', fname='mox', lines=('darkcyan', 'solid')),
    'energy_correction': dict(loc=21, label='Energy Correction', fname='ecor', lines=('darkorange', 'solid')),
    'pc2_checks': dict(loc=22, label='PC2 Checks', fname='pc2c', lines=('cyan', 'solid')),
    'pc2_initialisation': dict(loc=23, label='PC2 Initialisation', fname='pc2i', lines=('steelblue', 'solid')),
    'pc2_pressure_change': dict(loc=24, label='PC2 Pressure Change', fname='pc2p', lines=('cornflowerblue', 'solid')),
    'boundary_layer_mixing': dict(loc=25, label='Boundary Layer', fname='bl', lines=('palevioletred', 'solid')),
    'convection': dict(loc=26, label='Convection', fname='cu', lines=('tan', 'solid')),
    'advection': dict(loc=27, label='Advection', fname='adv', lines=('red', 'dashed')),
    'stochastic_kinetic_energy_backscatter': dict(loc=28, label='SKEB2', fname='skeb', lines=('lightseagreen', 'solid')),
    'stochastic_perturbation_of_tendencies': dict(loc=29, label='SPT', fname='spt', lines=('midnightblue', 'solid')),
    'pressure_solver': dict(loc=30, label='Solver', fname='slv', lines=('blue', 'dashed')),
    'advection_corrections': dict(loc=31, label='Advection Correction', fname='advcor', lines=('limegreen', 'dashed')),
    'diffusion': dict(loc=32, label='Diffusion', fname='diff', lines=('yellow', 'dashed')),
    'large_scale_cloud': dict(loc=33, label='Large Scale Cloud', fname='lsc', lines=('lightpink', 'solid')),
}

OTHER_DIAGS = {
    'time_difference': dict(loc=2, label='Time Diff (End - Begin)', fname='timediff', time_delta=True),
    'time_residual': dict(loc=3, label='Residual (Total - Diff)', fname='timeres', scale_levels=0.1, diff=[None, 'time_difference']),
    'idealised': dict(loc=10, label='Idealised', fname='idl'),
}


class Increments():
    ''' Class to manage increments '''

    # Set defaults
    inctype = 'change_over_time_in_'
    rms_thresh = 1.0e-10
    prognostics = PROGNOSTICS.keys()
    physics = PHYSICS.keys()
    cubes = None
    prognostic = None
    prog_cube = None
    prog_cubes = None
    difference = False

    kw_plot_increment_lines = ('rows', 'columns', 'loc', 'split', 'hgt',
                               'timescale', 'longitude', 'latitude', 'schemes')

    kw_plot_increment_zm = ('rows', 'columns', 'loc', 'fixed', 'hgt',
                            'timescale', 'cb_orientation', 'cb_units')

    kw_plot_increment_map = ('rows', 'columns', 'loc', 'fixed', 'model_level',
                            'timescale', 'cb_orientation', 'cb_units')

    linear_scale = ('air_temperature', 'upward_air_velocity',
                    'cloud_volume_fraction_in_atmosphere_layer',
                    'liquid_water_cloud_volume_fraction_in_atmosphere_layer',
                    'ice_cloud_volume_fraction_in_atmosphere_layer',
                    'air_potential_temperature', 'air_density',
                    'virtual_potential_temperature')

    def __init__(self, umfile=None, ncfile=None, forceload=False,
                 mr_physics=False, prognostics=None, quiet=True):
        ''' Initialisation of class '''

        # Deal with inputs
        self.umfile = umfile
        self.ncfile = ncfile
        self.lmrphy = mr_physics
        self.quiet = quiet

        # Use only selected prognostics
        if prognostics:
            self.prognostics = prognostics

        # Initialise RMS dictionary
        self.rms_dict = {p: None for p in self.prognostics}

        # Load data
        self.load_data(forceload=forceload)

    def __isub__(self, other):
        ''' Overload inplace subtraction, used to create differences '''
        assert isinstance(other, Increments), \
            'Must try to subtract another Increments instance'
        cubes = iris.cube.CubeList()
        for scube in self.cubes:
            try:
                name_cons = iris.Constraint(scube.name())
                dcube = scube - other.cubes.extract_cube(name_cons).copy()
                dcube.rename(scube.name())
                cubes.append(dcube)
            except iris.exceptions.ConstraintMismatchError:
                pass
        self.cubes = cubes
        self.difference = True
        return self

    def load_um_file(self, mr_physics=False):
        ''' Method to read a UM file '''
        if not self.quiet:
            print('>> Reading UM file ' + self.umfile)
        if mr_physics:
            if not self.quiet:
                print('>> Using mixing ratio fix!!')
            # callback = mixrat_callback
            callback = inc_callback.inc_callback_mr
        else:
            # callback = None
            callback = inc_callback.inc_callback
        self.cubes = iris.load(self.umfile, callback=callback)

    def load_nc_file(self):
        ''' Method to read a NetCDF file '''
        if not self.quiet:
            print('>> Reading NetCDF file ' + self.ncfile)
        self.cubes = iris.load(self.ncfile)

    def save_nc_file(self):
        ''' Method to write a NetCDF file '''
        if not self.quiet:
            print('>> Writing NetCDF file ' + self.ncfile)
        iris.save(self.cubes, self.ncfile)

    def load_data(self, forceload=False):
        ''' Method to load increment data '''

        # Set defaults, assume data in NetCDF file to be loaded
        loadum = False
        savenc = False
        loadnc = True

        # If NetCDF file is specified, then load UM and save NetCDF file only
        #  if we need to force loading or NetCDF file does not exist
        if self.ncfile:
            if forceload or not os.path.exists(self.ncfile):
                loadum = True
                savenc = True
        # If NetCDF file is not specified then must load UM file only
        else:
            loadum = True
            loadnc = False

        # Load UM file if required
        if loadum:
            self.load_um_file(mr_physics=self.lmrphy)

        # Save to NetCDF file
        if savenc:
            self.save_nc_file()

        # Load from NetCDF file
        if loadnc:
            self.load_nc_file()

    def set_prog_cube(self):
        ''' Method to extract cube for a given prognostic '''
        try:
            name_cons = iris.Constraint(self.prognostic)
            self.prog_cube = self.cubes.extract_cube(name_cons)
        except iris.exceptions.ConstraintMismatchError:
            self.prog_cube = None

    def set_prog_cubes(self):
        ''' Method to extract increment cubes for a given prognostic '''
        try:
            name = self.inctype + self.prognostic
            name_lam = lambda cube: cube.name().startswith(name)
            name_con = iris.Constraint(cube_func=name_lam)
            self.prog_cubes = self.cubes.extract(name_con)
        except iris.exceptions.ConstraintMismatchError:
            self.prog_cubes = None

    def set_prognostic(self, prognostic):
        ''' Method to set the current prognostic '''
        self.prognostic = prognostic
        self.set_prog_cube()
        self.set_prog_cubes()

    def has_increments(self):
        ''' Method to determine if given prognostic is in master cubelist '''
        return len(self.prog_cubes) > 0

    def get_name(self, scheme):
        ''' Method to create an increment name for given physics scheme '''
        return self.join_cf_stdname(self.inctype, self.prognostic, scheme)

    def create_instant_delta(self):
        '''
        Method to create a time difference between prognostic fields based
        on time bounds in total increment field.
        '''
        try:
            if self.prog_cube:
                inccube = self.get_increment(None)
                name = self.get_name('time_difference')
                tcoord = inccube.coord('time').copy()
                (time0, time1) = tcoord.cell(0).bound
                inst0 = self.prog_cube.extract(iris.Constraint(time=time0)).copy()
                inst1 = self.prog_cube.extract(iris.Constraint(time=time1)).copy()
                cube = self.diffcubes(inst1, inst0, name)
                cube.add_aux_coord(tcoord)
            else:
                cube = None
        except iris.exceptions.ConstraintMismatchError:
            cube = None
        return cube

    def create_increment(self, scheme):
        ''' Method to create derived increment for given physics scheme '''
        if scheme in OTHER_DIAGS:
            physinfo = OTHER_DIAGS[scheme]
        else:
            physinfo = PHYSICS[scheme]
        name = self.get_name(scheme)

        # Create cube as sum of other physics sections
        if 'sections' in physinfo:
            cubes = iris.cube.CubeList()
            for subscheme in physinfo['sections']:
                subcube = self.get_increment(subscheme)
                if subcube:
                    cubes.append(subcube)
            cube = self.sumcubes(cubes, name)

        # Create cube as difference of two other physics sections
        elif 'diff' in physinfo:
            cube0 = self.get_increment(physinfo['diff'][0])
            cube1 = self.get_increment(physinfo['diff'][1])
            cube = self.diffcubes(cube0, cube1, name)

        elif 'time_delta' in physinfo:
            if physinfo['time_delta']:
                cube = self.create_instant_delta()
            else:
                cube = None

        # No instructions - so no can do!
        else:
            cube = None

        # Add to main cube list
        if cube:
            if not self.quiet:
                print('>> Created increment ' + name)
            self.cubes.append(cube)
            self.set_prog_cubes()
        else:
            if not self.quiet:
                print('>> Failed to create increment ' + name)

    def get_increment(self, scheme):
        ''' Method to extract increment for given physics scheme '''
        name = self.get_name(scheme)
        name_cons = iris.Constraint(name)
        try:
            cube = self.prog_cubes.extract_cube(name_cons)
        except iris.exceptions.ConstraintMismatchError:
            cube = None
            # Create increment from other increments if possible
            self.create_increment(scheme)
            try:
                cube = self.prog_cubes.extract_cube(name_cons)
            except iris.exceptions.ConstraintMismatchError:
                pass
        if not self.quiet:
            if cube:
                print('>> Obtained increment ' + name)
            else:
                print('>> Failed to obtain increment ' + name)
        return cube

    def set_rms_dict(self, scheme, norm=False):
        '''
        Method to set RMS dict with RMS of increment for given physics scheme
        '''
        self.rms_dict[self.prognostic] = self.get_rms(scheme, norm=norm)

    def get_rms_dict(self, fulldict=False):
        ''' Method to return RMS dict '''
        if fulldict:
            result = self.rms_dict
        else:
            result = self.rms_dict[self.prognostic]
        return result

    def get_rms(self, scheme, norm=False):
        '''
        Method to calculate RMS of increment for given physics scheme

        Option to normalise to total increment
        '''
        cube = self.get_increment(scheme)
        if norm:
            normcube = self.get_increment(None)
        else:
            normcube = None
        return vol_rms(cube, norm=normcube)

    def display_rms(self):
        ''' Method to display RMS values '''
        print('\nNormalised Residual RMS Values:\n')
        for (prog, rms) in self.rms_dict.items():
            if rms is None:
                msg = f'    No test:      None = {prog}'
            elif rms < self.rms_thresh:
                msg = f'Test passed: {rms:.3e} = {prog}'
            else:
                msg = f'Test failed: {rms:.3e} = {prog}'
            print(msg)

    def plot_increment_lines(self, rows=0, columns=0, loc=0,
                             split=False, hgt=False,
                             timescale=DEFAULT_TIMESCALE,
                             longitude=(0, 360), latitude=(-90, 90),
                             schemes=[None]+SEC_TOT):
        ''' Plot vertical average increment line plots '''

        # Get prognostic information
        proginfo = PROGNOSTICS[self.prognostic]

        # Create plot axes
        axes = plt.subplot(rows, columns, loc)
        if split:
            axes2 = axes.twiny()
            split_level = proginfo.get('split', 50)

        # Set coordinates to plot against
        if hgt:
            coord = 'altitude'
            ylabel = 'Altitude (km)'
        else:
            coord = 'model_level_number'
            ylabel = 'Model Level'

        # Loop over schemes
        for scheme in schemes:
            cube = self.get_increment(scheme)
            if cube:
                pcube = cube.copy()

                # Get physics scheme information
                if scheme in OTHER_DIAGS:
                    physinfo = OTHER_DIAGS[scheme]
                else:
                    physinfo = PHYSICS[scheme]

                # Make alterations to cube to allow plotting
                # - Cannot do zonal mean with this in
                orog_coord_name = 'surface_altitude'
                if pcube.coords(orog_coord_name):
                    pcube.remove_coord(orog_coord_name)

                # Create time and zonal mean to plot
                pcube = pcube.intersection(longitude=longitude,
                                           latitude=latitude)
                pcube = area_avg(pcube)

                # Put level heights in km for better axis labels
                hgt_coord_name = 'atmosphere_hybrid_height_coordinate'
                if pcube.coords(hgt_coord_name):
                    pcube.coord(hgt_coord_name).convert_units('km')

                # Modify plot units
                if 'punits' in proginfo:
                    pcube.convert_units(proginfo['punits'])
                self.scale_time_unit(pcube, timescale)

                # Extract line color and style
                (lcolor, lstyle) = physinfo.get('lines', ('black', 'solid'))
                plot_kw = dict(label=physinfo['label'], color=lcolor,
                               linestyle=lstyle, linewidth=1.0)
                if split:
                    pcube1 = pcube[1:split_level]
                    pcube2 = pcube[split_level:]
                    iplt.plot(pcube1, pcube1.coord(coord), axes=axes, **plot_kw)
                    iplt.plot(pcube2, pcube2.coord(coord), axes=axes2, **plot_kw)
                else:
                    iplt.plot(pcube, pcube.coord(coord), axes=axes, **plot_kw)

        axes.legend(loc=0, fontsize='small')
        axes.set_ylabel(ylabel)
        line_kw = dict(color='black', linestyle='dotted', linewidth=0.5)
        axes.axvline(0.0, **line_kw)

        # Set up x-axis scaling
        lim = axes.get_xlim()
        xlim = max(max(lim), -min(lim))
        axes.set_xlim((-xlim, xlim))

        if self.prognostic in self.linear_scale:
            axes.set_xscale('linear')
        else:
            axes.set_xscale('symlog')

        if split:
            axes.axhline(split_level+0.5, **line_kw)

            lim = axes2.get_xlim()
            xlim2 = max(max(lim), -min(lim))
            axes2.set_xlim((-xlim2, xlim2))

            if self.prognostic in self.linear_scale:
                axes2.set_xscale('linear')
            else:
                axes.set_xscale('symlog', linthreshx=xlim2)
                axes2.set_xscale('linear')

    def plot_increment_zm(self, cube, rows=0, columns=0, loc=0,
                          fixed=False, hgt=False, timescale=DEFAULT_TIMESCALE,
                          cb_orientation='vertical', cb_units=False):
        ''' Plot zonal mean increment '''

        # Make alterations to cube to allow plotting
        # - Cannot do zonal mean with this in
        orog_coord_name = 'surface_altitude'
        if cube.coords(orog_coord_name):
            cube.remove_coord(orog_coord_name)

        # Create time and zonal mean to plot
        pcube = cube.collapsed('longitude', iris.analysis.MEAN)
        if pcube.coords('time', dim_coords=True):
            pcube = pcube.collapsed('time', iris.analysis.MEAN)

        # Put level heights in km for better axis labels
        hgt_coord_name = 'atmosphere_hybrid_height_coordinate'
        if pcube.coords(hgt_coord_name):
            pcube.coord(hgt_coord_name).convert_units('km')

        # Determine prognostic and physics scheme
        (_, prognostic, scheme) = self.split_cf_stdname(cube.name())
        proginfo = PROGNOSTICS[prognostic]
        if scheme in OTHER_DIAGS:
            physinfo = OTHER_DIAGS[scheme]
        else:
            physinfo = PHYSICS[scheme]

        # Set coordinates to plot against
        if hgt:
            coords = ['latitude', 'altitude']
            ylabel = 'Altitude (km)'
        else:
            coords = ['latitude', 'model_level_number']
            ylabel = 'Model Level'

        # Modify plot units
        if 'punits' in proginfo:
            pcube.convert_units(proginfo['punits'])
        self.scale_time_unit(pcube, timescale)

        # Create plot axes
        if loc < 1:
            loc = physinfo['loc']
        axes = plt.subplot(rows, columns, loc)

        # Plot cube with fixed levels or self-determined
        cmap = mpl_cm.get_cmap('RdBu_r')
        if fixed:
            scale_levels = physinfo.get('scale_levels', 1.0)
            levels = scale_levels * proginfo['levels']
            # if self.difference:
            #    levels *= 0.1
            norm = mpl_col.BoundaryNorm(levels, cmap.N, clip=True)
            ctf = iplt.contourf(pcube, axes=axes, coords=coords, extend='both',
                                cmap=cmap, norm=norm, levels=levels)
        else:
            ctf = iplt.contourf(pcube, axes=axes, coords=coords, extend='both',
                                cmap=cmap)

        # Fix for unsightly "contour lines"
        # https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills
        for coll in ctf.collections:
            coll.set_edgecolor("face")

        # Add colorbar
        cbar = plt.colorbar(ctf, orientation=cb_orientation, extend='both',
                            spacing='uniform')
        if cb_units:
            cbar.set_label(rf'${{unit_latex(pcube.units)}}$')

        # Set plot niceties
        plot_title = f'd{proginfo["label"]} {physinfo["label"]}'
        axes.set_title(plot_title)
        axes.set_ylabel(ylabel)
        axes.xaxis.set_major_formatter(cmgridl.LATITUDE_FORMATTER)
        axes.set_xticks([-90, -60, -30, 0, 30, 60, 90])

        # If all values are zero then plaster a big "= 0" on plot
        if not cube.data.any():
            axes.text(0.5, 0.5, 'ZERO', ha='center', va='center',
                      transform=axes.transAxes, fontsize='xx-large',
                      bbox=dict(edgecolor='black', facecolor='yellow'))

    def plot_increment_map(self, cube, rows=0, columns=0, loc=0, model_level=1,
                           fixed=False, timescale=DEFAULT_TIMESCALE,
                           cb_orientation='horizontal', cb_units=False):
        ''' Plot increment on given model level '''

        # Extract model level and create time mean
        pcube = cube.extract(iris.Constraint(model_level_number=model_level))
        if pcube.coords('time', dim_coords=True):
            pcube = pcube.collapsed('time', iris.analysis.MEAN)

        # Determine prognostic and physics scheme
        (_, prognostic, scheme) = self.split_cf_stdname(cube.name())
        proginfo = PROGNOSTICS[prognostic]
        if scheme in OTHER_DIAGS:
            physinfo = OTHER_DIAGS[scheme]
        else:
            physinfo = PHYSICS[scheme]

        # Modify plot units
        if 'punits' in proginfo:
            pcube.convert_units(proginfo['punits'])
        self.scale_time_unit(pcube, timescale)

        # Create plot axes
        if loc < 1:
            loc = physinfo['loc']
        axes = plt.subplot(rows, columns, loc, projection=ccrs.PlateCarree())

        # Plot cube with fixed levels or self-determined
        cmap = mpl_cm.get_cmap('RdBu_r')
        if fixed:
            scale_levels = physinfo.get('scale_levels', 1.0)
            levels = scale_levels * proginfo['levels']
            # if self.difference:
            #    levels *= 0.1
            norm = mpl_col.BoundaryNorm(levels, cmap.N, clip=True)
            ctf = iplt.pcolormesh(pcube, axes=axes, cmap=cmap, norm=norm)
        else:
            ctf = iplt.pcolormesh(pcube, axes=axes, cmap=cmap)

        # Add colorbar
        cbar = plt.colorbar(ctf, orientation=cb_orientation, extend='both',
                            spacing='uniform')
        if cb_units:
            cbar.set_label(rf'${{unit_latex(pcube.units)}}$')

        # Set plot niceties
        plot_title = f'd{proginfo["label"]} {physinfo["label"]}'
        axes.set_title(plot_title)
        axes.coastlines()
        gl = axes.gridlines(draw_labels=True, linestyle='dotted', alpha=0.5)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = cmgridl.LONGITUDE_FORMATTER
        gl.yformatter = cmgridl.LATITUDE_FORMATTER

        # If all values are zero then plaster a big "= 0" on plot
        if not cube.data.any():
            axes.text(0.5, 0.5, 'ZERO', ha='center', va='center',
                      transform=axes.transAxes, fontsize='xx-large',
                      bbox=dict(edgecolor='black', facecolor='yellow'))

    def plot_section30(self, plotdir, **kwargs):
        ''' Method to plot section 30 increment against actual change '''
        kwargs['rows'] = 1
        kwargs['columns'] = 3
        map_plot = kwargs.pop('map_plot', False)

        # Filter kwargs for plotting routine
        if map_plot:
            plot_kw = self.filtdict(kwargs, self.kw_plot_increment_map)
        else:
            plot_kw = self.filtdict(kwargs, self.kw_plot_increment_zm)

        # Set Matplotlib defaults
        mpl.rcParams['axes.titlesize'] = 'x-small'
        mpl.rcParams['axes.labelsize'] = 'x-small'
        mpl.rcParams['xtick.labelsize'] = 'xx-small'
        mpl.rcParams['ytick.labelsize'] = 'xx-small'

        # Get data to plot
        totcube = self.get_increment(None)
        indcube = self.get_increment('time_difference')
        rescube = self.get_increment('time_residual')
        resrms = self.get_rms('time_residual', norm=True)

        # Only create plot if main two cubes present
        # - implies residual will exist as well
        if totcube and indcube:

            # Set up plot
            fig = plt.figure(figsize=(17, 5))

            # Plot cubes
            if map_plot:
                self.plot_increment_map(totcube.copy(), **plot_kw)
                self.plot_increment_map(indcube.copy(), **plot_kw)
                self.plot_increment_map(rescube.copy(), **plot_kw)
            else:
                self.plot_increment_zm(totcube.copy(), **plot_kw)
                self.plot_increment_zm(indcube.copy(), **plot_kw)
                self.plot_increment_zm(rescube.copy(), **plot_kw)

            # Add super title
            suptitle = self.plot_suptitle(self.prognostic, resrms)
            fig.suptitle(suptitle)

            # Adjust spacing between plots
            # fig.tight_layout()
            fig.subplots_adjust(left=0.05, right=0.95, top=0.7,
                                bottom=0.15, wspace=0.3, hspace=0.6)

            # Save image
            proglab = PROGNOSTICS[self.prognostic]['label'].lower()
            if map_plot:
                proglab = 'map_' + proglab
            else:
                proglab = 'zm_' + proglab
            fig.savefig(os.path.join(plotdir, proglab+'_sec30.png'))
            plt.close()

    def plot_lines(self, plotdir, **kwargs):
        ''' Method to plot vertical lines '''
        kwargs['rows'] = 1
        kwargs['columns'] = 1
        kwargs['loc'] = 1
        toplev = kwargs.pop('toplev', False)

        filetag = ''
        if toplev:
            filetag = '_toplev'
            kwargs['schemes'] = [None,
                                 'derived_model_physics',
                                 'derived_model_dynamics',
                                 'derived_parallel_physics',
                                 'derived_sequential_physics',
                                 'derived_further_physics']

        # Filter kwargs for plotting routine
        plot_kw = self.filtdict(kwargs, self.kw_plot_increment_lines)

        # Set up plot
        fig = plt.figure(figsize=(16.0, 8.0))

        # Plot cubes
        self.plot_increment_lines(**plot_kw)

        # Add super title
        suptitle = self.plot_suptitle(self.prognostic, None)
        fig.suptitle(suptitle)

        # Save image
        proglab = PROGNOSTICS[self.prognostic]['label'].lower()

        fig.savefig(os.path.join(plotdir, proglab+'_lines'+filetag+'.png'))
        plt.close()

    def plot_budget_top(self, plotdir, **kwargs):
        ''' Method to plot top level residual summary '''
        kwargs['rows'] = 1
        kwargs['columns'] = 3
        map_plot = kwargs.pop('map_plot', False)

        # Filter kwargs for plotting routine
        if map_plot:
            plot_kw = self.filtdict(kwargs, self.kw_plot_increment_map)
        else:
            plot_kw = self.filtdict(kwargs, self.kw_plot_increment_zm)

        # Set Matplotlib defaults
        mpl.rcParams['axes.titlesize'] = 'x-small'
        mpl.rcParams['axes.labelsize'] = 'x-small'
        mpl.rcParams['xtick.labelsize'] = 'xx-small'
        mpl.rcParams['ytick.labelsize'] = 'xx-small'

        # Get data to plot
        totcube = self.get_increment(None)
        sumcube = self.get_increment('derived_model_total')
        rescube = self.get_increment('total_residual')
        resrms = self.get_rms_dict()

        # Only create plot if main two cubes present
        # - implies residual will exist as well
        if totcube and sumcube:

            # Set up plot
            fig = plt.figure(figsize=(17, 5))

            # Plot cubes
            if map_plot:
                self.plot_increment_map(totcube.copy(), **plot_kw)
                self.plot_increment_map(sumcube.copy(), **plot_kw)
                self.plot_increment_map(rescube.copy(), **plot_kw)
            else:
                self.plot_increment_zm(totcube.copy(), **plot_kw)
                self.plot_increment_zm(sumcube.copy(), **plot_kw)
                self.plot_increment_zm(rescube.copy(), **plot_kw)

            # Add super title
            suptitle = self.plot_suptitle(self.prognostic, resrms)
            fig.suptitle(suptitle)

            # Adjust spacing between plots
            # fig.tight_layout()
            fig.subplots_adjust(left=0.05, right=0.95, top=0.7,
                                bottom=0.15, wspace=0.3, hspace=0.6)

            # Save image
            proglab = PROGNOSTICS[self.prognostic]['label'].lower()
            if map_plot:
                proglab = 'map_' + proglab
            else:
                proglab = 'zm_' + proglab
            fig.savefig(os.path.join(plotdir, proglab+'_toplev.png'))
            plt.close()

    def plot_budget_page(self, plotdir, **kwargs):
        ''' Method to plot all increments on a single page '''
        kwargs['rows'] = 11
        kwargs['columns'] = 3
        map_plot = kwargs.pop('map_plot', False)

        # Filter kwargs for plotting routine
        if map_plot:
            plot_kw = self.filtdict(kwargs, self.kw_plot_increment_map)
        else:
            plot_kw = self.filtdict(kwargs, self.kw_plot_increment_zm)

        # Set Matplotlib defaults
        mpl.rcParams['axes.titlesize'] = 'small'
        mpl.rcParams['axes.labelsize'] = 'xx-small'
        mpl.rcParams['xtick.labelsize'] = 'xx-small'
        mpl.rcParams['ytick.labelsize'] = 'xx-small'

        # Set up plot
        fig = plt.figure(figsize=(17, 12))

        # for cube in self.prog_cubes:
        for scheme in self.physics:
            cube = self.get_increment(scheme)
            if cube:
                if map_plot:
                    self.plot_increment_map(cube.copy(), **plot_kw)
                else:
                    self.plot_increment_zm(cube.copy(), **plot_kw)

        # Add super title
        resrms = self.get_rms_dict()
        suptitle = self.plot_suptitle(self.prognostic, resrms)
        fig.suptitle(suptitle)

        # Adjust spacing between plots
        # fig.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9,
                            bottom=0.05, wspace=0.3, hspace=1.0)

        # Save image
        proglab = PROGNOSTICS[self.prognostic]['label'].lower()
        if map_plot:
            proglab = 'map_' + proglab
        else:
            proglab = 'zm_' + proglab
        fig.savefig(os.path.join(plotdir, proglab+'_incs.png'))
        plt.close()

    def plot_individual(self, plotdir, **kwargs):
        ''' Method to plot individual increments on a single page '''
        kwargs['rows'] = 1
        kwargs['columns'] = 1
        kwargs['loc'] = 1
        kwargs['cb_orientation'] = 'horizontal'
        kwargs['cb_units'] = True
        map_plot = kwargs.pop('map_plot', False)

        # Filter kwargs for plotting routine
        if map_plot:
            plot_kw = self.filtdict(kwargs, self.kw_plot_increment_map)
        else:
            plot_kw = self.filtdict(kwargs, self.kw_plot_increment_zm)

        # Set Matplotlib defaults
        mpl.rcParams['axes.titlesize'] = 'large'
        mpl.rcParams['axes.labelsize'] = 'medium'
        mpl.rcParams['xtick.labelsize'] = 'medium'
        mpl.rcParams['ytick.labelsize'] = 'medium'

        # for cube in self.prog_cubes:
        for scheme in self.physics:
            cube = self.get_increment(scheme)

            if cube:
                # Set up plot
                fig = plt.figure(figsize=(8, 6))

                if map_plot:
                    self.plot_increment_map(cube.copy(), **plot_kw)
                else:
                    self.plot_increment_zm(cube.copy(), **plot_kw)

                # Save image
                proglabel = PROGNOSTICS[self.prognostic]['label'].lower()
                if scheme:
                    physlabel = '_' + PHYSICS[scheme]['fname'].lower()
                else:
                    physlabel = ''
                plotfile = f'{proglabel}_inc{physlabel}.png'
                if map_plot:
                    plotfile = 'map_' + plotfile
                else:
                    plotfile = 'zm_' + plotfile
                plotfile = os.path.join(plotdir, plotfile)
                fig.savefig(plotfile)
                plt.close()

    @staticmethod
    def filtdict(indict, keys):
        return {k: indict[k] for k in set(keys) if k in indict}

    @staticmethod
    def sumcubes(cubes, name):
        ''' Sum list of cubes into one cube '''
        sumcube = None
        if cubes:
            sumcube = 0.0
            for cube in cubes:
                sumcube += cube.copy()
            sumcube.rename(name)
        return sumcube

    @staticmethod
    def diffcubes(cube0, cube1, name):
        ''' Difference list of 2 cubes '''
        diffcube = None
        if cube0 and cube1:
            diffcube = cube0 - cube1
            diffcube.rename(name)
        return diffcube

    @staticmethod
    def join_cf_stdname(inctype, prognostic, scheme):
        '''
        Join up prognostic name and optional physics scheme in the form:

        <inctype><prognostic>[_due_to_<scheme>]
        '''
        name = inctype + prognostic
        if scheme:
            name += '_due_to_' + scheme
        return name

    @staticmethod
    def split_cf_stdname(name):
        '''
        Split up increment standard name into prognostic and physics scheme
        '''
        inc_regexp = r'^(?P<type>change_over_time_in_|tendency_of_|)(?P<prognostic>\w+?)(?:_due_to_(?P<scheme>\w+))?$'
        return re.match(inc_regexp, name).group('type', 'prognostic', 'scheme')

    @staticmethod
    def plot_suptitle(prog, resrms, timescale=DEFAULT_TIMESCALE):
        ''' Method to generate a suitable suptitle for increment plots '''
        (cdot, inv) = (r'{\cdot}', r'{-1}')
        proginfo = PROGNOSTICS[prog]
        proglabel = proginfo['label']
        if 'punits' in proginfo:
            latex_unit = unit_latex(proginfo['punits'])
        else:
            latex_unit = unit_latex(proginfo['units'])
        latex_unit += rf'{cdot}{{{timescale}}}^{inv}'
        suptitle = rf'{{proglabel}} Increment Budget (${{latex_unit}}$)'
        if resrms is not None:
            suptitle += f'\n\nNormalised Residual RMS = {resrms:.3e}'
        return suptitle

    @staticmethod
    def scale_time_unit(cube, timescale):
        '''
        Method to scale increments by timescale given no time dependence in
        cube units. Useful for "change_over_time_in_" fields.
        '''
        tcoord = cube.coord('time')
        period = tcoord.bounds[0, 1] - tcoord.bounds[0, 0]
        coord_units = str(tcoord.units).split(' ')[0]
        deltat = f'{period:f} {coord_units}'
        cube_units = str(cube.units)
        if cube_units == '1':
            cube.units = f'({deltat})-1'
            cube.convert_units(f'{timescale}-1')
        else:
            cube.units = f'{cube_units!s} ({deltat})-1'
            cube.convert_units(f'{cube_units!s} {timescale}-1')


def area_avg(cube, vert=None):
    ''' Routine to calculate global area-weighted mean fields '''
    if cube:
        cube.coord('longitude').guess_bounds()
        cube.coord('latitude').guess_bounds()
        weights = iac.area_weights(cube)
        mean = cube.collapsed(['time', 'longitude', 'latitude'],
                              iris.analysis.MEAN,
                              weights=weights)
        # Do vertical average if requested, using named vertical coordinate
        if vert:
            mean = mean.collapsed(vert, iris.analysis.MEAN)
    else:
        mean = None
    return mean


def area_rms(cube, vert=None, norm=None):
    ''' Routine to calculate global area-weighted RMS fields '''
    if cube:
        mnsq = area_avg(cube * cube, vert=vert)
        if norm:
            mnsq /= area_avg(norm * norm, vert=vert)
        rms = np.sqrt(mnsq.data)
    else:
        rms = None
    return rms


def vol_rms(cube, norm=None):
    '''
    Routine to calculate volume area-weighted RMS fields

    This guesses vertical coordinate and uses area_rms
    '''
    if cube and cube.coords(axis='Z'):
        zcoord = cube.coords(axis='Z')
        rms = area_rms(cube, vert=zcoord[0].name(), norm=norm)
    else:
        rms = None
    return rms


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
                newunitstr = r'{\mu}{' + unit.replace('micro', '') + '}'
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
    re_unit = re.compile(r'(^[^-+\d\^]+)([\^]?)([-+]?[\d]*)$')
    latex_str = [conv_latex(unit, re_unit) for unit in str(unit).split()]
    # Join using a central dot
    return r'{\cdot}'.join(latex_str)


# ==========================================
# Dealing with mixing ratio physics metadata
# ==========================================
# Keeping separate as it involves STASH codes, everything above is CF

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


def msi_stashcode(lbuser):
    ''' Create MSI stashcode from pp header entry lbuser '''
    (model, (section, item)) = (lbuser[6], divmod(lbuser[3], 1000))
    return f'm{model:02d}s{section:02d}i{item:03d}'


def mixrat_callback(cube, field, filename):
    '''
    Callback routine to fix the metadata for mixing ratio diagnostics.

    From the UM they are written out into specific quantity stashcodes, so
    the metadata is wrong.
    '''
    # If STASH code is actually mixing ratio then change name
    if msi_stashcode(field.lbuser) in MIXING_RATIO_SWITCH:
        stdname = MIXING_RATIO_SWITCH[msi_stashcode(field.lbuser)]
        cube.rename(stdname)

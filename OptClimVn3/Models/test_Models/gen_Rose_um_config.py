# generate a ROSE_um config.
import pathlib
import genericLib
from UM_rose import UKESM1_1_c8 # cylc8 version
import shutil
import tempfile
genericLib.setup_env() # set up default env.
name='002case'
model_dir = pathlib.Path('/work/n02/n02/tetts/test')/name
model_dir = tempfile.TemporaryDirectory(prefix='OptClim_test_') # use temp dir for testing.
model_dir = pathlib.Path(model_dir.name) # use temp dir for testing.
try:
    shutil.rmtree(model_dir)
except FileNotFoundError:
    pass

#reference = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898')) # ROSE config
reference = pathlib.Path(genericLib.expand('/home/n02/n02-puma/tetts/roses/roses/u-dr157')) # ROSE config
#reference = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898'))
#reference = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898')) # ROSE config
pp_dir = '/work/n02/shared/tetts/OptClim/st2024/post_process/'
post_process = dict(
    script=pp_dir+'comp_sim_obs_UKESM1_1.py',
    output_file='observations.json',
    mask_file=pp_dir+'landfrac_N96.nc',
    mask_file_comment="Path for landfrac file.",
    mask_fraction=0.5,
    mask_fraction_comment="Critical Fraction. Specify if mask  is a land/sea fraction. Values >= are land < sea. Set to null if mask is a t/f mask",
    start_time=None,
    start_time_comment="Start time as ISO std string. ",
    end_time="2011-12-31",
    end_time_comment="End time as str of ISO std string",
    file_pattern='*a.pm*.pp',
    file_pattern_comment="File pattern to match for post-processing. Use * as wildcard. ",

)

#parameters = dict( dp_corr_strat=500.0,two_d_fsd_factor=2,
#                   ent_fac_dp= 1.0, ai=3e-2,RUN_TARGET='P2M')
# Full list of parameters can be found in the UKESM1_1 class.
parameters = {'dz0v_dh_io': 0.05,
 'f0_io': 0.875,
 'nl0_io': 0.046,
 'rootd_ft_io': 2,
 'tupp_io': 43,
 'u10_max_coare': 22.0,
 'rho_snow_fresh': 109.0,
 'orog_drag_param': 0.15,
 'dec_thres_cloud': 0.1,
 'zhloc_depth_fac': 0.4,
 'dbsdtbs_turb_0': 0.00015,
 'forced_cu_fac': 0.5,
 'ice_width': 0.02,
 'amdet_fac': 3.0,
 'cca_md_knob': 0.1,
 'cca_sh_knob': 0.5,
 'ent_fac_dp': 1.13,
 'ent_fac_md': 0.9,
 'mparwtr': 0.0015,
 'qlmin': 0.0003,
 'r_det': 0.8,
 'fbcd': 4.0,
 'gsharp': 0.5,
 'gwd_frc': 4.0,
 'gwd_fsat': 0.25,
 'nsigma': 2.5,
 'ai': 0.0257,
 'ar': 1.0,
 'c_r_correl': 0.9,
 'mp_dz_scal': 2.0,
 'x1r': 0.22,
 'dp_corr_strat': 10000.0,
 'two_d_fsd_factor': 1.5,
 'biom_aer_ems_scaling': 2.0}

run_info=dict(
    run_info_comment='Information for running system. ',
    runCode='n02-TERRAFIRMA', # run with terafirma
    prebuild=True, # use prebuild from ref model
    prebuild_comment='If True guess from ref model. If string use as path to prebuild. Path should be on puma2',
    use_scratch=True,
    use_scratch_comment='use scratch space on Archer2. Means models get cleaned up after 28 days.',
    runQueue='serial',
    runQueue_comment='Q to run the pp job in',
    runExtraArgs=['--qos=serial'],
    runExtraArgs_comment='List of extra args for submission. For archer2 need to specifiy qos for pp job',
)
model = UKESM1_1_c8(name=name,
                suite_dir=model_dir/'suite',
                reference=reference,
                model_dir=model_dir,
                #post_process=post_process,
                parameters=parameters,
                run_info=run_info)
model.instantiate()
## print out UM functions values in the model.
for name,fn in model.param_info.known_functions.items():
    try:
        result = fn(model,0.0) # call the function with a dummy value.
        if result is None:
            print(f'{fn.__qualname__} returned None')
            continue
        nl_info  = [nl[0] for nl in result]
        values = [model.read_nl_value(nl) for nl in nl_info]
        print(f'{fn.__qualname__} = {values} with  len {len(values[0])}')

    except Exception as e:
        print(f'{fn.__qualname__} failed with {e}')

##model.submit_model() # should submit post-process and model.
pth = pathlib.Path('~')/model.suite_dir.relative_to(model.puma_dir.parent)
print(f'Submited {pth} on puma')
print(model.print_output())

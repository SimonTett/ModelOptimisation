# generate a ROSE_um config.
import pathlib
import genericLib
from UM_rose import UM_rose
import shutil
genericLib.setup_env() # set up default env.
name='002case'
model_dir = pathlib.Path('/work/n02/n02/tetts/test')/name
try:
    shutil.rmtree(model_dir)
except FileNotFoundError:
    pass

#reference = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898')) # ROSE config
reference = pathlib.Path(genericLib.expand('/home/n02/n02-puma/tetts/roses/u-db898')) # ROSE config
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

parameters = dict( dp_corr_strat=500.0,two_d_fsd_factor=2,
                   ent_fac_dp= 1.0, ai=3e-2,RUN_TARGET='P2M')

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
model = UM_rose(name=name,
                reference=reference,
                model_dir=model_dir,
                post_process=post_process,
                parameters=parameters,
                run_info=run_info)
model.instantiate()
model.submit_model() # should submit post-process and model.
pth = pathlib.Path('~')/model.suite_dir.relative_to(UM_rose.puma_dir.parent)
print(f'Submited {pth} on puma')
print(model.print_output())

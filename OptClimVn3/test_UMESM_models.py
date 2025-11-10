## test can run and that post=process works.
import json
import pathlib
import genericLib
from UM_rose import UKESM1_1_c8 # cylc8 version
import shutil
import tempfile
import socket

genericLib.setup_env() # set up default env.
root_name='case'
hostname = socket.gethostname()
my_logger = genericLib.setup_logging(level='INFO')
config_amip = 'u-dr496'
config_amip_p4 = 'u-dr783'
archer= False
if hostname.startswith('ln0'): # on archer

    archer = True
    base_dir = genericLib.expand(f'/home/n02/n02-puma/tetts/roses')
    references = [base_dir/config  for config in [config_amip,config_amip_p4]] # ROSE configs
    model_dir_base = pathlib.Path('/work/n02/n02/tetts/test')
    pp_dir = '/work/n02/shared/tetts/OptClim/st2024/post_process/'
    suite_base_dir= None
    script = pp_dir + 'comp_sim_obs_UKESM1_1.py',
    post_process = dict(
        script=script,
        output_file='observations.json',
        mask_file=pp_dir + 'landfrac_N96.nc',
        mask_file_comment="Path for landfrac file.",
        mask_fraction=0.5,
        mask_fraction_comment="Critical Fraction. Specify if mask  is a land/sea fraction. Values >= are land < sea. Set to null if mask is a t/f mask",
        start_time='2011-01',
        start_time_comment="Start time as ISO std string. ",
        end_time="2011-12",
        end_time_comment="End time as str of ISO std string",
        file_pattern='*a.pm*.pp',
        file_pattern_comment="File pattern to match for post-processing. Use * as wildcard. ",

    )
else:
    my_logger.warning('Running on a non-archer host. Using temporary directory for testing.')
    archer = False
    base_dir = genericLib.expand(f'/home/n02/n02-puma/tetts/roses')
    references = [base_dir / config for config in [config_amip, config_amip_p4]]  # ROSE configs

    model_base_dir = tempfile.TemporaryDirectory(prefix='OptClim_test_') # use temp dir for testing.
    model_base_dir = pathlib.Path(model_base_dir.name) # use temp dir for testing.
    suite_base_dir = model_base_dir/'suites'
    post_process = None # no post-processing for testing.


with genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/parameters_UKESM1_1.ijson').open('rt') as f:
    config  = json.load(f)
parameters = {k:v*1.05 for k,v in config['defaultParams'].items() if not k.endswith('comment')}
# try with all parameters with default values scaled by 5%
# set ensembleMember to 2. Make sure archiving off and set the run target and resubmission time.
parameters.update(ensembleMember=2,archive_pp=False,archive_netcdf=False,
                  START_TIME='2010-09-01',RUN_TARGET='2011-12-30',RESUB_TIME='P4M')
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

for name,ref in zip(['ctl','p4k'],references):
    model_dir = model_dir_base/name
    if suite_base_dir is None: # no suite_base_dir defined so suite_dir is None.
        suite_dir = None
    else:
        suite_dir = suite_base_dir/name
        try:
            shutil.rmtree(suite_dir)
        except FileNotFoundError:
            pass
    try:
        shutil.rmtree(model_dir)
    except FileNotFoundError:
        pass

    model = UKESM1_1_c8(name=name,
                    suite_dir=suite_dir,
                    reference=reference,
                    model_dir=model_dir,
                    post_process=post_process,
                    parameters=parameters,
                    run_info=run_info)
    model.instantiate()

    if archer:
        model.submit_model() # should submit post-process and model.
        pth = pathlib.Path('~')/model.suite_dir.relative_to(model.puma_dir.parent)
        print(f'Submitted {pth} on puma')
    print(model.print_output())

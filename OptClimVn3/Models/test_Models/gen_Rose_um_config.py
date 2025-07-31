# generate a ROSE_um config.
import json
import pathlib
import genericLib
from UM_rose import UKESM1_1_c8 # cylc8 version
import shutil
import tempfile
import socket

genericLib.setup_env() # set up default env.
name='002case'
hostname = socket.gethostname()
my_logger = genericLib.setup_logging(level='INFO')
config = 'u-dr157'
archer= False
if hostname.startswith('ln0'): # on archer
    archer = True
    reference = pathlib.Path(genericLib.expand(f'/home/n02/n02-puma/tetts/roses'))/config  # ROSE config
    model_dir = pathlib.Path('/work/n02/n02/tetts/test')/name
    pp_dir = '/work/n02/shared/tetts/OptClim/st2024/post_process/'
else:
    my_logger.warning('Running on a non-archer host. Using temporary directory for testing.')
    archer = False
    reference = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/'))/config # ROSE config
    model_dir = tempfile.TemporaryDirectory(prefix='OptClim_test_') # use temp dir for testing.
    model_dir = pathlib.Path(model_dir.name) # use temp dir for testing.
    pp_dir =  '/work/n02/shared/tetts/OptClim/st2024/post_process/'
try:
    shutil.rmtree(model_dir)
except FileNotFoundError:
    pass



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


with genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/parameters_UKESM1_1.ijson').open('rt') as f:
    config  = json.load(f)
parameters = {k:v*1.05 for k,v in config['defaultParams'].items() if not k.endswith('comment')}
# try with all parameters with default values scaled by 5%


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
        if isinstance(values[0], (list, tuple)):
            print(f'{fn.__qualname__} = {values} with  len {len(values[0])}')
        else:
            print(f'{fn.__qualname__} = {values}')


    except Exception as e:
        my_logger.warning(f'{fn.__qualname__} failed with {e}')

if archer:
    model.submit_model() # should submit post-process and model.
    pth = pathlib.Path('~')/model.suite_dir.relative_to(model.puma_dir.parent)
    print(f'Submitted {pth} on puma')
print(model.print_output())

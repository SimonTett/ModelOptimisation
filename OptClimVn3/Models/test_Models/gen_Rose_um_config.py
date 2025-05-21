# generate a ROSE_um config.
import pathlib
import genericLib
from UM_rose import UM_rose
import shutil
genericLib.setup_env() # set up default env.
name='case002'
model_dir = pathlib.Path('/work/n02/n02/tetts/test')/name
try:
    shutil.rmtree(model_dir)
except FileNotFoundError:
    pass

#reference = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898')) # ROSE config
reference = pathlib.Path(genericLib.expand('/home/n02/n02-puma/tetts/roses/u-db898')) # ROSE config
post_process = dict(script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', output_file='obs.json')

parameters = dict( DP_CORR_STRAT=500.0,TWO_D_FSD_FACTOR=2,
                   ENT_FAC_DP= 1.0, AI=3e-2,RUN_TARGET='P2M')

run_info=dict(
    prebuild='/home/n02/n02/tetts/cylc-run/u-db898/share/fcm_make_um', # use prebuild from ref model
    use_scratch=True # use scratch space. Means models get cleaned up after 28 days. 
)
model = UM_rose(name=name, reference=reference,
                            model_dir=model_dir, post_process=post_process,
                            parameters=parameters,
                run_info=run_info)
model.instantiate()
model.set_status('SUBMITTED') # set status to submit.  But go and submit by hand. Once generally happy. Do this automatically....
pth = pathlib.Path('~')/model.suite_dir.relative_to(UM_rose.puma_dir.parent)
print(f'Submit case on puma with rose suite-run  -v -v -C {pth}')

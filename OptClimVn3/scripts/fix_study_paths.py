#!/usr/bin/env python
# load Study and fix the paths.
# new study gets written out.
# this mostly  uses pure json.
# example use: OptClimVn3/scripts/fix_study_paths.py wenjun_data/dfols_random3/dfols_r.scfg wenjun_fix --clean --max_model_simulations 25 --write_config

import json
import argparse
import pathlib
import logging
import sys
import typing
import copy

import SubmitStudy
import generic_json
import genericLib



def init_log(
        log: logging.Logger,
        level: str,
        log_file: typing.Optional[typing.Union[pathlib.Path, str]] = None,
        datefmt: typing.Optional[str] = '%Y-%m-%d %H:%M:%S',
        mode: str = 'a'
):
    """
    Set up logging on a logger! Will clear any existing logging.
    TODO: roll into optclim general lib so available to everything.
    :param log: logger to be changed
    :param level: level to be set.
    :param log_file:  if provided pathlib.Path to log to file
    :param mode: mode to open log file with (a  -- append or w -- write)
    :param datefmt: date format for log.
    :return: nothing -- existing log is modified.
    """
    log.handlers.clear()
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s',
                                  datefmt=datefmt
                                  )
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    # add a file handler.
    if log_file:
        if isinstance(log_file, str):
            log_file = pathlib.Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(log_file, mode=mode + 't')  #
        fh.setLevel(level)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    log.propagate = False
    return log

parser=argparse.ArgumentParser(description='Fix paths in a study. \n Example: OptClimVn3/scripts/fix_study_paths.py wenjun_data/dfols_random3/dfols_r.scfg wenjun_fix --clean --max_model_simulations 25')
parser.add_argument('input',type=pathlib.Path,help='Input study configuration')
parser.add_argument('outdir',type=pathlib.Path,help='Output *directory*')
parser.add_argument('--write_config',action='store_true',help='Write config out (from study config)')
parser.add_argument('--clean',action='store_true',help='Remove all models that are not in state:PROCESSED')
parser.add_argument('--max_model_simulations',type=int,help='Modify max_model_simulations when writing out config') # TODO allow a json file to overwrite
parser.add_argument("-v", "--verbose", action='count', default=0,
                    help="level of logging info level= 1 = info, level = 2 = debug ")
#parser.add_argument('basedir',)
args=parser.parse_args()

if args.verbose >= 2:
    level='DEBUG'
elif args.verbose >= 1:
    level = 'INFO'
else:
    level='WARNING'

my_logger = genericLib.setup_logging(
    level=level)


input_dir = args.input.parent
encoder = generic_json.JSON_Encoder()
decode = generic_json.obj_to_from_dict(error='warn')
clean=args.clean
with args.input.open() as fp:
    config = json.load(fp)

config_path = pathlib.PurePosixPath(decode.decode(config['object']['config_path']))
rewrite_paths = {config_path.parent:args.outdir}

args.outdir.mkdir(parents=True,exist_ok=True) # make output dir.
config_change = decode.rename_paths(config,rewrite_paths)
refdir = config['object']['refDir']['object']
config_change['object']['refDir']=encoder.default(pathlib.PurePosixPath(refdir)) # make sure refDir points to initial value


outpath  =  decode.dct_lst_to_obj(config_change['object']['config_path'])

# iterate over the original config, load models, rewrite paths and write out.
models_to_delete=[] # list of models to delete from *modified* study config
for key,model_path in config['object']['model_index'].items():
    path = pathlib.PurePath(model_path['object'])
    in_pth = input_dir/path.relative_to(config_path.parent)
    if not in_pth.exists():
        my_logger.warning(f'Failed to find {in_pth} skipping ')
        models_to_delete += [key]
        continue
    new_path=None
    for k, v in rewrite_paths.items():
        try: # replacing, in place, the model path info.
            new_path = v / path.relative_to(k)
            model_path['object'] = encoder.default(new_path)
            my_logger.debug(f'Rewrote {path} to {new_path}')
        except ValueError:
            pass
    if new_path is not None:
        with in_pth.open('tr') as fp:
            model_config = json.load(fp)
            model = decode.decode(model_config)
            if model.status != 'PROCESSED':
                my_logger.warning(f'{model.name } Status is {model.status} not PROCESSED ')
                if clean:
                    my_logger.warning(f'Removing {model.name}')
                    if new_path.exists():
                        new_path.unlink()
                        my_logger.warning(f'Removed {new_path} for model {model.name}')


                    models_to_delete  += [key]
                    continue
        new_model_config = decode.rename_paths(model_config,rewrite_paths)
        # overwrite the path with new_path
        new_model_config['object']['config_path'] = encoder.default(new_path)

        # List of keys that should be pure paths. That way we can't use it.
        pure_path_vars = ['reference','submit_script','continue_script','set_status_script','post_process_cmd_script']
        for key in pure_path_vars:
            if isinstance(model_config['object'][key],list):
                # TODO -- have recursive encoder.
                result = []
                for c in model_config['object'][key]:
                    if isinstance(c,dict) and '__cls__name__' in c:
                        de = pathlib.PurePosixPath(decode.decode(c))
                        result.append(encoder.default(de))
                    else:
                        result.append(c)

                new_model_config['object'][key] = result
            else:
                cname = decode.decode(model_config['object'][key])
                cname = pathlib.PurePosixPath(cname)
                new_model_config['object'][key]=encoder.default(cname)



        model_config_path = decode.decode(new_model_config['object']['config_path'])
        model_config_path.parent.mkdir(parents=True,exist_ok=True)

        with model_config_path.open('tw') as fp:
            json.dump(new_model_config,fp)
            my_logger.debug(f'Dumped model to {model_config_path}')

# Remove from config_change all the models in the delete list.
for key in models_to_delete:
    mpath =  decode.decode(config_change['object']['model_index'].pop(key))
    my_logger.debug(f'Removing model at {mpath} from config')

with outpath.open('tw') as fp:
    json.dump(config_change,fp) # write it out.
#change_config.dump_config()
mod_config = SubmitStudy.SubmitStudy.load(outpath,error='warn')
if args.max_model_simulations:
    mod_config.config.max_model_simulations(args.max_model_simulations)
    my_logger.info(f'Set max_model_simulations to {args.max_model_simulations}')
    mod_config.dump_config()
## write out config if write_config provided.
if args.write_config:
    config_file = mod_config.config_path.parent/mod_config.config._filename.name
    mod_config.config._filename = config_file
    my_logger.info(f'Writing data to {config_file}')
    mod_config.config.save(filename=config_file,verbose=True)
    mod_config.dump_config()
    print(f'To rerun study use: OptClimVn3/scripts/runAlgorithm.py {config_file} --readonly --update --dir {args.outdir}')
    my_logger.info(f'Wrote out modified config to {outpath}')


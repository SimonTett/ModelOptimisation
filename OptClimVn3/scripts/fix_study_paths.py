#!/usr/bin/env python
# load Study and fix the paths.
# new study gets written out.
# this just uses pure json..

import json
import argparse
import pathlib
import logging
import sys
import typing
import copy

import SubmitStudy
import generic_json
import model_base

my_logger = logging.getLogger(__name__)
def init_log(
        log: logging.Logger,
        level: str,
        log_file: typing.Optional[typing.Union[pathlib.Path, str]] = None,
        datefmt: typing.Optional[str] = '%Y-%m-%d %H:%M:%S',
        mode: str = 'a'
):
    """
    Set up logging on a logger! Will clear any existing logging.
    TODO: roll into optclim general lib sp available to everything.
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
# def rewrite(dct:typing.Union[dict,list],rewrite_paths:dict):
#     # each entry consits of group of three:
#     # __cls__name__
#     #__object__
#     #__module__
#     if isinstance(dct,dict):
#         cls_name = dct.get('__cls__name__')
#         if cls_name is None:
#             for k,value in dct.items():
#                 if isinstance(value,(dict,list)):
#                     rewrite(value,rewrite_paths)
#         elif cls_name in ['PosixPath','WindowsPath','PureWindowsPath','PurePosixPath']:
#             path = pathlib.PurePath(dct['object'])
#             parent = path.parent
#             for k,v in rewrite_paths.items():
#                 if parent == k:
#                     dct['object'] = str(v/path.name)
#                     dct['__cls__name__'] = 'PurePosixPath'
#                     my_logger.debug(f'Rewrote {parent} to {v}')
#         else:
#             if isinstance(dct['object'],dict):
#                 rewrite(dct['object'],rewrite_paths)
#     elif isinstance(dct,list):
#         for value in dct:
#             if isinstance(value, (dict, list)):
#                 rewrite(value, rewrite_paths)
#     else:
#         pass
#
#
#
#
#     elif isinstance(dct_lst, dict): # need to recurse but no rewrite at this level.
#         result=dict()
#         for k,value in dct_lst.items():
#             if isinstance(value,(dict,list)) :
#                 result[k] = rename_dct(value,rewrite_paths)
#             else:
#                 result[k] = value
#
#     elif isinstance(dct_lst, list):
#         result = list()
#         for value in dct_lst:
#             if isinstance(value,(list,dict)):
#                 result.append(rename_dct(value,rewrite_paths))
#             else:
#                 result.append(value)
#     else:
#         raise ValueError(f'Not list or dict... but {type(dct_lst)}')
#
#     return result









parser=argparse.ArgumentParser(description='Fix paths in a study.')
parser.add_argument('input',type=pathlib.Path,help='Input configuration')
parser.add_argument('outdir',type=pathlib.Path,help='Output *directory*')
#parser.add_argument('basedir',)
init_log(my_logger,'DEBUG')
args=parser.parse_args()
input_dir = args.input.parent
encoder = generic_json.JSON_Encoder()
decode = generic_json.obj_to_from_dict(error='warn')
with args.input.open() as fp:
    config = json.load(fp)

config_path = pathlib.PurePosixPath(decode.decode(config['object']['config_path']))
rewrite_paths = {config_path.parent:args.outdir}

args.outdir.mkdir(parents=True,exist_ok=True) # make output dir.
config_change = decode.rename_paths(config,rewrite_paths)
refdir = config['object']['refDir']['object']
config_change['object']['refDir']=encoder.default(pathlib.PurePosixPath(refdir)) # make sure refDir points to initial value


outpath  =  decode.dct_lst_to_obj(config_change['object']['config_path'])
with outpath.open('tw') as fp:
    json.dump(config_change,fp)
# iterate over the original config, load models, rewrite paths and write out.
for key,model_path in config['object']['model_index'].items():
    path = pathlib.PurePath(model_path['object'])
    in_pth = input_dir/path.relative_to(config_path.parent)
    if not in_pth.exists():
        my_logger.warning(f'Failed to find {in_pth} skipping ')
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



#change_config.dump_config()
mod_config = SubmitStudy.SubmitStudy.load(outpath,error='warn')

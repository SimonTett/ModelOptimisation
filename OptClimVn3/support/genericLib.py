"""
 Library of functions that can be called from any module in OptClim
   - get_default -- get default value. like get method to dict.
   - copyDir: Copy directory recursively in a sensible way to allow testing

"""
from __future__ import annotations

import errno
import os
import shutil
import stat
import pathlib
import typing

import numpy as np
import pandas as pd
import logging
import logging.config
import copy

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

def setup_logging(level:typing.Optional[int] = None,
                  rootname:typing.Optional[str] = None,
                  log_config:typing.Optional[dict]=None):
    """
    Setup logging. 
    :param: level: level of logging . If None logging.WARNIG will be used
    :param: rootname: rootname for logging. if None OPTCLIM will be used. 
    :param: log_config config dict for logging.config --
          see https://docs.python.org/3/library/logging.config.html
          If not None will only be used if level is not None and the actual 
          value of level will be ignored. 
    """
    
    if rootname is None:
        rootname  = 'OPTCLIM'
    
    optclim_logger = logging.getLogger(rootname) # get OPTCLIM root logger

    # need both debugging turned on and a logging config
    # to use the logging_cong
    if level is not None and log_config is not None:
        logging.debug("Using log_config to set up logging")
        logging.config.dictConfig(log_config) # assume this is sensible
        return optclim_logger

    if level is None:
        level = logging.WARNING
        
    # set up a sensible default logging behaviour. 

    optclim_logger.handlers.clear() #  clear any existing handles there are
    optclim_logger.setLevel(level) # set the level
    
    console_handler = logging.StreamHandler()
    fmt = '%(levelname)s:%(name)s:%(funcName)s: %(message)s'
    formatter = logging.Formatter(fmt)
    console_handler.setFormatter(formatter)

    optclim_logger.addHandler(console_handler) # turning this on gives duplicate messages. FIXME.
    optclim_logger.propagate = False # stop propogation to root level.
# see https://jdhao.github.io/2020/06/20/python_duplicate_logging_messages/
    return optclim_logger
        
def fake_fn(config: "OptClimConfigVn3", params: dict) -> pd.Series:
    """
    Wee test fn for trying out things.
    :param config -- configuration. Provides, parameter min, max & ranges and targets.
    :param params -- dict of parameter values
    returns  "fake" data as a pandas Series
    """
    params = copy.deepcopy(params)
    my_logger.debug("faking with params: " + str(params))
    # remove ensembleMember param.
    params.pop('ensembleMember', None)  # remove ensembleMember as a key.
    pranges = config.paramRanges()
    tgt = config.targets()
    min_p = pranges.loc['minParam', :]
    max_p = pranges.loc['maxParam', :]
    scale_params = max_p - min_p
    keys = list(params.keys())
    for k in keys: # remove parameters that do not have a range.
        if k not in pranges.columns:
            params.pop(k)
    param_series = pd.Series(params).combine_first(config.standardParam())  # merge in the std params
    pscale = (param_series - min_p) / scale_params
    pscale -= 0.5  # tgt is at params = 0.5
    result = 100 * (pscale + pscale ** 2)
    if np.any(result.isnull()):
        raise ValueError("Got null in result")
    # this fn has one minimum and  no maxima between the boundaries and the minima. So should be easy to optimise.
    result = result.to_numpy()
    while (len(tgt) > result.shape[-1]):
        result = np.append(result, result, axis=-1)
    result = result[0:len(tgt)]  # truncate it to len of tgt.
    result = pd.Series(result, index=tgt.index)  # brutal conversion to obs space.
    var_scales = 10.0 ** np.round(np.log10(config.scales()))
    result /= var_scales  # make sure changes are roughly right scales.

    result += tgt
    return result

def parse_isoduration( s: str | typing.List) -> typing.List|str:
    """ Parse a str ISO-8601 Duration: https://en.wikipedia.org/wiki/ISO_8601#Durations
      OR convert a 6 element list (y m, d, h m s) into a ISO duration.
    Originally copied from:
    https://stackoverflow.com/questions/36976138/is-there-an-easy-way-to-convert-iso-8601-duration-to-timedelta
    Though could use isodate library but trying to avoid dependencies and isodate does not look maintained.
    :param s: str to be parsed. If not a string starting with "P" then ValueError will be raised.
    :return: 6 element list [YYYY,MM,DD,HH,mm,SS.ss] which is suitable for the UM namelists
    """

    def get_isosplit(s, split):
        if split in s:
            n, s = s.split(split, 1)
        else:
            n = '0'
        return n.replace(',', '.'), s  # to handle like "P0,5Y"

    if isinstance(s, str):
        my_logger.debug(f"Parsing {str}")
        if s[0] != 'P':
            raise ValueError("ISO 8061 demands durations start with P")
        s = s.split('P', 1)[-1]  # Remove prefix

        split = s.split('T')
        if len(split) == 1:
            sYMD, sHMS = split[0], ''
        else:
            sYMD, sHMS = split  # pull them out

        durn = []
        for split_let in ['Y', 'M', 'D']:  # Step through letter dividers
            d, sYMD = get_isosplit(sYMD, split_let)
            durn.append(float(d))

        for split_let in ['H', 'M', 'S']:  # Step through letter dividers
            d, sHMS = get_isosplit(sHMS, split_let)
            durn.append(float(d))
    elif isinstance(s, list) and len(s) == 6:  # invert list
        durn = 'P'
        my_logger.debug("Converting {s} to string")
        for element, chars in zip(s, ['Y', 'M', 'D', 'H', 'M', 'S']):
            if element != 0:
                if isinstance(element, float) and element.is_integer():
                    element = int(element)
                durn += f"{element}{chars}"
            if chars == 'D':  # days want to add T as into the H, M, S cpt.
                if np.any(np.array(s[3:]) != 0):
                    durn += 'T'
        if durn == 'P':  # everything = 0
            durn += '0S'
    else:
        raise ValueError(f"Do not know what to do with {s} of type {type(s)}")

    return durn

def expand(filestr: str) -> pathlib.Path:
    """

    Expand any env vars, convert to path and then expand any user constructs.
    :param filestr: path like string
    :return:expanded path
    """
    path = os.path.expandvars(filestr)
    path = pathlib.Path(path).expanduser()
    return path


def errorRemoveReadonly(func, path, exc):
    """
    Function to run when error found in rmtree.
    :param func: function being called
    :param path: path to file being removed
    :param exc: failure status
    :return: None
    """

    excvalue = exc[1]
    # if func in (os.rmdir, os.remove,builtins.rmdir) and excvalue.errno == errno.EACCES:
    if excvalue.errno == errno.EACCES:
        # change the file to be readable,writable,executable: 0777
        try:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except WindowsError:  # dam windows.
            os.chmod(path, stat.S_IWRITE)
        func(path)


def delDirContents(dir):
    """
    Recursively Delete the contents of a directory
    :param dir: path to directory to have all contents removed.
    :return: Nada
    """
    # from stack exchange
    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python

    if not os.path.exists(dir):  # doesn't exist so return
        return

    with os.scandir(dir) as entries:
        for entry in entries:
            if entry.is_file() or entry.is_symlink():
                try:
                    os.chmod(entry, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                except WindowsError:  # dam windows.
                    os.chmod(entry, stat.S_IWRITE)
                os.remove(entry.path)  # and remove it
            elif entry.is_dir():  # directory -- remove everything in it.
                shutil.rmtree(entry.path, onerror=errorRemoveReadonly)  # remove all directories


def get_default(dct, key, default):
    """
    :param dct: dictionary to read value from
    :param key: key to use
    :param default: default value to use if not set or None
    :return: value from dct if provided, default if not provided or None
    """
    value = dct.get(key, default)
    if value is None:  # value is none then use default
        value = default
    return value  # and return the value


def copyTestDir(inDir, outDir, setDir='', trace=False):
    """
    copy a directory for test purposes. Need to do this so created directory is
    read/write so can be deleted.
    :paam inDir: Inout directory to be copied
    :param outDir: Target for copy. If it exists it will be deleted.
    :keyword trace: Defualt value False -- if True print out more info
    :keyword setDir: Default value '' -- if set copy this directory. If not set then only start will be copied
    :return: status == 0 if successful. Something else if not!
    """

    status = 0
    if not os.path.exists(inDir):
        raise ValueError(inDir + " does not exist")

    if os.path.exists(outDir):
        shutil.rmtree(outDir)  # remove the tgt directory
    # now create the directory and copy files and, if they exist, the  start and outDir dirs
    os.mkdir(outDir)
    for root, dirs, files in os.walk(inDir, topdown=True):  # iterate
        for name in dirs:  # iterate over  the directories first creating new ones
            newroot = root.replace(inDir, outDir, 1)  # root in new dir tree
            newdir = os.path.join(newroot, name)  # dir to be created in new tree
            if root == inDir and name in ['start', os.path.basename(setDir)]:
                # at toplevel only want to copy start and setDir across
                os.mkdir(newdir, 0o777)  # make the directory and make it world read/write/exec
                if trace:
                    print("created dir: ", newdir)
            elif root != inDir and os.path.isdir(newroot):  # make dir only if its root exists
                os.mkdir(newdir, 0o777)  # make the directory
                if trace:
                    print("created dir as root exists: ", newdir)
            else:
                pass

        for name in files:  # iterate over files in directory
            oldpath = os.path.join(root, name)
            newdir = root.replace(inDir, outDir, 1)
            newfile = os.path.join(newdir, name)
            if os.path.isdir(newdir):  # directory for file to go into exists
                shutil.copy(oldpath, newdir)  # copy file.
                os.chmod(newfile, stat.S_IWRITE)
                if trace:
                    print("copied ", oldpath, ' to: ', newdir)
            else:
                pass  # nothing to do.
    return status


# done with copyTestDir

def genSeed(param: pd.Series) -> int:
    """
    Initialise RNG based on parameter as pandas series. So is deterministic.
    :param param: pandas series of values
    :return: seed for RNG
    """

    paramValues = pd.to_numeric(param, errors='coerce').values
    L = np.isnan(paramValues)
    paramValues[L] = 1
    maxSeed = 2 ** (31) - 1  #
    seed = 0
    seed += int(np.product(paramValues))  # .view(np.uint64))
    seed += int(np.sum(paramValues))  # .view(np.uint64))
    while (seed > maxSeed):
        seed = seed // 2

    return seed


import argparse
import json


def std_post_process_setup(parser: argparse.ArgumentParser) -> typing.Tuple[argparse.Namespace,  dict]:
    """
    Adds standard post-processing arguments to a parser and then parse the parser.
     Then read in the post_processing information
    :param parser: parser to have arguments added to.
      arguments added are:
         CONFIG -- path to config file.
         -d/-dir -- name of input directory.
         OUTPUT -- path to output file
         -v/--verbose -- increase verbosity.  -v turns logging.info on while -v -v turns logging.debug on.

    config will be loaded from CONFIG and postProcess key extracted.
    output file  taken from post_process info (if present) otherwise
    sets appropriate  logging level using basicConfig and force=True (will overwrite any existing logging)

    :return: args (after parsing) post_process dict
      args contains whatever in parser and:
        CONFIG -- path to config file
        OUTPUT -- path to OUTPUT file
        dir -- path to directory where data to be read from.
        verbose -- level of verbosity.
    """
    parser.add_argument("CONFIG", type=str,help="The Name of the Config file. Should be a json file with a postProcess entry.")
    parser.add_argument("-d", "--dir", type=str, help="The path to the input directory", default=os.getcwd())
    parser.add_argument("OUTPUT", nargs='?', default=None,
                        help="The name of the output file. Will override what is in the config file")
    parser.add_argument("-v", "--verbose", help="Increase logging level. -v = info, -v -v = debug", action="count",
                        default=0)

    args = parser.parse_args()  # and parse the arguments
    # Get stuff in.
    args.CONFIG = expand(args.CONFIG)
    with open(args.CONFIG, 'rt') as fp:
        config = json.load(fp)
    post_process = config.get('postProcess', {})
    if args.OUTPUT is None:
        output_file = post_process['outputPath']  # better be defined so throw error if not
    else:
        output_file = args.OUTPUT

    args.OUTPUT = expand(output_file)  # expand users and env vars.
    args.dir = expand(args.dir) # expand users and env vars

    if args.verbose == 1:
        logging.basicConfig(force=True, level=logging.INFO)
    elif args.verbose > 1:
        logging.basicConfig(force=True, level=logging.DEBUG)
    else:  # nothing to do
        pass

    my_logger.debug("Post Process data")
    for key, value in post_process.items():
        my_logger.debug(f"{key}:{value}")

    return args,  post_process

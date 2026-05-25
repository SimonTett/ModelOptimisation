"""
 Library of functions that can be called from any module in OptClim
   - get_default -- get default value. like get method to dict.
   - copyDir: Copy directory recursively in a sensible way to allow testing

"""
from __future__ import annotations

import copy
import errno
import importlib
import logging
import logging.config
import os
import pathlib
import shutil
import stat
import sys
import typing
import tempfile
import collections
import argparse
import json

import numpy as np
import pandas as pd
import filelock

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")


def _reset_logger(logger_name):
    """
    AI generated function to reset a logger to default state. This is useful for testing purposes to avoid duplicate log messages.
    No test cases for this function as it is only used in testing and is not critical to the functioning of the code. It is also not expected to be used by users of the code.
    :param logger_name: name of logger to be reset
    :return: Nada.
    """
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def _reset_all_loggers():
    """
    Reset all loggers to default state, closing all handlers and deleting temp files if possible.
    """

    root_logger = logging.getLogger()
    for handle in root_logger.handlers[:]:
        handle.close()
        stream = getattr(handle, 'stream', None)
        if stream and hasattr(stream, 'name'):
            # Check if it's a tempfile
            if isinstance(stream, tempfile._TemporaryFileWrapper):
                try:
                    stream.close()
                    os.unlink(stream.name)
                except Exception:
                    pass
            elif isinstance(stream.name, str) and os.path.exists(stream.name):
                try:
                    os.unlink(stream.name)
                except Exception:
                    pass
        root_logger.removeHandler(handle)
    root_logger.setLevel(logging.NOTSET)
    # Reset all named loggers
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            for handle in logger.handlers[:]:
                handle.close()
                stream = getattr(handle, 'stream', None)
                if stream and hasattr(stream, 'name'):
                    if isinstance(stream, tempfile._TemporaryFileWrapper):
                        try:
                            stream.close()
                            os.unlink(stream.name)
                        except Exception:
                            pass
                    elif isinstance(stream.name, str) and os.path.exists(stream.name):
                        try:
                            os.unlink(stream.name)
                        except Exception:
                            pass
                logger.removeHandler(handle)
            logger.setLevel(logging.NOTSET)
            logger.propagate = True


error_handle_types = typing.Literal['fail', 'warn', 'ignore']


# will use error handle in various different places so define it here.
# It will be used to control whether to raise an error, log a warning or ignore an error when something goes wrong.
def error_handle(message: str, error: error_handle_types = 'fail'):
    """
    Handle an error according to the error handling strategy.
    :param message: message to be used in error or warning
    :param error: if 'fail' then raise an error with the message. If 'warn' then log a warning with the message. If 'ignore' then do nothing.
    :return: None
    """
    if error == 'fail':
        raise ValueError(message)
    elif error == 'warn':
        my_logger.warning(message)
    elif error == 'ignore':
        pass
    else:
        raise ValueError(f"Unknown error handling option {error}")


def expand_filelike_keys(dct: dict, mkdir: bool = False) -> dict:
    """
    Expand any keys in the dict that are filepath
    :param dct: dict to be processed
    :param mkdir -- If True create needed directories.
    :return: new dict with expanded keys
    """
    filepath_strings = ['filename']
    new_dct = {}
    for key, value in dct.items():
        if isinstance(value, (str, pathlib.PurePath)) and key in filepath_strings:
            # value is string or path and key is in filepath_strings so expand it.
            new_dct[key] = expand(value)
            if mkdir:
                new_dct[key].parent.mkdir(exist_ok=True, parents=True)
                logging.debug(f"Created directory {new_dct[key].parent} for {key}")
            logging.debug(f"Expanded {value} to {new_dct[key]}")
        elif isinstance(value, dict):
            # value is a dict so process it recursively
            new_dct[key] = expand_filelike_keys(value,mkdir=mkdir)
        else:
            new_dct[key] = value
    return new_dct


## Code to support logging

def setup_logging(level: typing.Optional[typing.Union[int, str]] = None,
                  rootname: typing.Optional[str] = None,
                  log_config: typing.Optional[dict] = None):
    """
    Setup logging. 
    :param: level: level of logging. If None logging.WARNING will be used
    :param: rootname: rootname for logging. if None OPTCLIM will be used. 
    :param: log_config config dict for logging.config --
          see https://docs.python.org/3/library/logging.config.html
          If not None will only be used if level is not None and the actual 
          value of level will be ignored. 
    """

    if rootname is None:
        rootname = 'OPTCLIM'

    optclim_logger = logging.getLogger(rootname)  # get OPTCLIM root logger

    # need both debugging turned on and a logging config
    # to use the logging_cong
    if level is not None and log_config is not None:
        logging.debug("Using log_config to set up logging")
        mkdir = log_config.pop('mkdir', False)
        log_config_expanded = expand_filelike_keys(log_config,
                                                   mkdir=mkdir)  # expand any file like keys in the log config
        logging.config.dictConfig(log_config_expanded)  # assume this is sensible
        return optclim_logger

    if level is None:
        level = logging.WARNING

    # set up a sensible default logging behaviour. 

    optclim_logger.handlers.clear()  #  clear any existing handles there are
    optclim_logger.setLevel(level)  # set the level

    console_handler = logging.StreamHandler()
    fmt = '%(levelname)s:%(name)s:%(funcName)s: %(message)s'
    formatter = logging.Formatter(fmt)
    console_handler.setFormatter(formatter)

    optclim_logger.addHandler(console_handler)  # turning this on gives duplicate messages.
    optclim_logger.propagate = False  # stop propogation to root level which suppresses duplicate messages.
    # see https://jdhao.github.io/2020/06/20/python_duplicate_logging_messages/
    return optclim_logger


def init_log(
        log: logging.Logger,
        level: str,
        log_file: typing.Optional[typing.Union[pathlib.Path, str]] = None,
        datefmt: typing.Optional[str] = '%Y-%m-%d %H:%M:%S',
        mode: str = 'a'
) -> logging.Logger:
    """
    Set up logging on a logger! Will clear any existing logging.
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





class DuplicateFilter(logging.Filter):
    """
    Code to support filtering out duplicate log messages. AI generated
    """

    def __init__(self, max_duplicates: int = 1000,
                 level: int|str = logging.WARNING):
        """
        :param max_duplicates: maximum number of duplicates
        """
        super().__init__()
        self.seen = collections.deque(maxlen=max_duplicates) # where we store what we have seen.
        if isinstance(level, str): # convert level to string
            level = logging.getLevelName(level.upper())
            if not isinstance(level, int):
                raise TypeError(f"level {level} is not an integer. Check your value of level for typos")
        self.level = level

    def filter(self, record) -> bool:
        """

        :param record: logging record
        :return: True if keep, False if not to keep
        """
        # Only apply filtering to  level and above
        if record.levelno < self.level:
            return True  # Let it through without checking duplicates

        msg = record.getMessage()

        if msg in self.seen:
            return False

        self.seen.append(msg)

        return True


def get_fn(mod_fn_str: str) -> typing.Callable:
    """
    Load a function from a module.
    :param mod_fn_str:
    :return: a callable
    """
    mod, fn_name = mod_fn_str.rsplit('.', maxsplit=1)
    module = importlib.import_module(mod)  # import the module
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise AttributeError(f"{fn_name} is not a callable in {mod}")

    return fn


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
    for k in keys:  # remove parameters that do not have a range.
        if k not in pranges.columns:
            params.pop(k)
    param_series = pd.Series(params).combine_first(config.standardParam())  # merge in the std params
    #TODO fix FutureWarning: The behavior of array concatenation with empty entries is deprecated.
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


def seconds_to_isoduration(seconds: int | float) -> str:
    """
    Convert seconds to ISO-8601 duration.
    :param seconds: Seconds (int or float).
    :return: Iso-duration string (and zeros will be ignored)
    If seconds is -ve then a value error is raised. This may not be necessary but -ve seconds needs a different
    processing and - duration.
    Uses https://github.com/pydantic/pydantic/blob/3704eccce4661455acdda1cdcf716bd4b3382e08/pydantic/deprecated/json.py#L135-L140
    """

    if seconds < 0:
        raise ValueError("Seconds must be positive")
    minutes, seconds = divmod(seconds, 60)
    minutes = int(minutes)  # convert minutes to an int
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    result = 'P'
    for value, prd, format in zip([days, hours, minutes, seconds], ['D', 'H', 'M', 'S'],
                                  ['d', 'd', f'd', f'2.3f']):
        if prd == 'H':
            result += 'T'
        if value > 0:
            result += f'{value:{format}}{prd}'
        elif prd == 'S' and result == 'PT':  # if no time then add 0S
            result += '0S'
        else:
            pass

    # remove trailing T (when Hr, min & secs are all zero)
    if result[-1] == 'T':
        result = result[:-1]

    return result


def parse_isoduration(s: str | typing.List) -> typing.List | str:
    """ Parse a str ISO-8601 Duration: https://en.wikipedia.org/wiki/ISO_8601#Durations
      OR convert a 6 element list (y m, d, h m s) into a ISO duration.
    Originally copied from:
    https://stackoverflow.com/questions/36976138/is-there-an-easy-way-to-convert-iso-8601-duration-to-timedelta
    Though could use isodate library but trying to avoid dependencies and isodate does not look maintained.
    :param s: str to be parsed. If not a string starting with "P" then ValueError will be raised.
    :return: 6 element list [YYYY,MM,DD,HH,mm,SS.ss] which is suitable for the UM namelists
    """

    def get_isosplit(s: str, split):
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
        if (len(split) == 1 and 'Y' in split[0]) or 'T' not in s:
            sYMD, sHMS = split[0], ''
        elif len(split) == 1:
            sYMD, sHMS = '', split[0]
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
        my_logger.debug(f"Converting {s} to string")
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


def expand(filestr: str | pathlib.PurePath,
           error: error_handle_types = 'fail') -> pathlib.Path:
    """

    Expand any env vars, convert to path and then expand any user constructs.
    :param filestr: path like string
    :param error: if 'fail' then raise an error if expanded path contains a $ or % (showing an env var was not expanded).
      If 'warn' then log a warning but return the path. If 'ignore' then just return the path.
    :return:expanded path
    """
    if isinstance(filestr, pathlib.PurePath):
        filestr = filestr.as_posix()
    if '%' in filestr:
        message = 'Expanding path with % in it. This may not work on all platforms. Use $ instead of % for env vars.'
        error_handle(message, error)
    path = os.path.expandvars(filestr)
    path = pathlib.Path(path).expanduser()
    if '$' in str(
            path):  # if there is still a $ or % in the path then an env var was not expanded. So raise an error or log a warning.
        message = f"Path {path} contains unexpanded env vars. Original string was {filestr}"
        error_handle(message, error)
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


def delete_dir_contents(direct: pathlib.Path):
    """
    Recursively Delete the contents of a directory
    :param direct: path to directory to have all contents removed.
    :return: Nada
    """
    # from stack exchange
    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python

    if not direct.exists():  # doesn't exist so return
        return
    if not direct.is_dir():
        raise ValueError(f"{direct} is not a directory")

    for entry in direct.iterdir():
        if entry.is_file() or entry.is_symlink():
            try:
                entry.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except WindowsError:  # dam windows.
                entry.chmod(stat.S_IWRITE)
            entry.unlink()  # and remove it
        elif entry.is_dir():  # directory -- remove everything in it.
            shutil.rmtree(entry, onerror=errorRemoveReadonly)  # remove all directories


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
    my_logger.warning(f"Calling copyTestDir -- probably should not be! Created {outDir}")
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


def std_post_process_setup(parser: argparse.ArgumentParser) -> typing.Tuple[argparse.Namespace, dict]:
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
    parser.add_argument("CONFIG", type=str,
                        help="The Name of the Config file. Should be a json file with a postProcess entry.")
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
    args.dir = expand(args.dir)  # expand users and env vars

    if args.verbose == 1:
        logging.basicConfig(force=True, level=logging.INFO)
    elif args.verbose > 1:
        logging.basicConfig(force=True, level=logging.DEBUG)
    else:  # nothing to do
        pass

    my_logger.debug("Post Process data")
    for key, value in post_process.items():
        my_logger.debug(f"{key}:{value}")

    return args, post_process


def setup_env():
    """
    Setup a default environment for OptClim. Sets the following variables iff they are undefined:
      OPTCLIMTOP
    :return: Nada
    """

    if 'OPTCLIMTOP' not in os.environ:
        optclimtop = pathlib.Path(__file__).resolve().parents[2]
        os.environ['OPTCLIMTOP'] = str(optclimtop)
        my_logger.debug(f"Setting OPTCLIMTOP to {os.environ['OPTCLIMTOP']}")
    else:
        my_logger.debug(f"OPTCLIMTOP already set to {os.environ['OPTCLIMTOP']}")
    return


def backup_file(path: pathlib.Path,
                ext: str = '.bak',
                create: typing.Optional[typing.Literal['copy', 'move']] = None, ) -> typing.Optional[pathlib.Path]:
    """
    Create a backup file with the given extension.
    :param path: Path to the original file.
    :param ext: Extension for the backup file. Default '.bak'
    :param create: If  copy -- create backup by copying the file, if move -- create backup by moving the file.
    If backup file exists then no backup file is created.
    If None then no backup file is created.
    :return: Path to the backup file or None if no backup is created.
    """
    # check input file exits. If not raise an error
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    backup_path = path.with_suffix(path.suffix + ext)
    if create is not None:  # create backup
        if backup_path.exists():
            my_logger.warning(f'Backup file {backup_path} already exists. Not creating')
            return None
        # create the backup file
        try:
            if create == 'move':
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                path.rename(backup_path)  # move the file to backup path
            elif create == 'copy':
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, backup_path)
            else:
                raise ValueError(f'Create {create} not known.')
            my_logger.debug(f'Backup file created: {backup_path}')
        except OSError as e:
            my_logger.error(f'Error creating backup file: {e}')

    return backup_path


def likely_text_file(file_path: pathlib.Path,
                     sample_size: int = 1024,
                     encoding: str = 'ascii') -> bool:
    """
    Guess if a file is likely a text file by checking if its sample_size bytes can be
      decoded to only printable or space chars.
    co-pilot generated.
    :param file_path: Path to the file.
    :param sample_size: Number of bytes to read for analysis.
    :param encoding: Encoding to use for decoding the file.
    :return: True if the file is likely a text file, False otherwise.
    """
    try:
        with open(file_path, 'rb') as file:
            sample = file.read(sample_size)
        # Try decoding the sample
        decoded_sample = sample.decode(encoding)
        # Check if all characters in the decoded sample are printable or space
        return all(char.isprintable() or char.isspace() for char in decoded_sample)
    except (UnicodeDecodeError, Exception):
        # If decoding fails or the file cannot be read, assume it's not a text file
        return False


try_symlinks = True  # try to use symlinks


def copy_files(in_direct: pathlib.Path,
               out_direct: pathlib.Path,
               files: list[pathlib.Path],
               symlinks: bool = False
               ) -> list[pathlib.Path]:
    """

    Only the files specified in files will be copied. Will ignore directories.
    :param in_direct: directory where study is currently located
    :param out_direct: directory where study is to be copied. Will be created if it does not exist and emptied if it does.
    :param files: list of  files (paths provided relative to in_direct) to be copied to new directory.
    :param symlinks: if True then rather than copying files, symlinks will be created.
      This should be faster but is less reliable

    :return: Copied object and list of files copied. (which should be inheriting from model_base)
    """
    global try_symlinks  # have to be global vars

    out_direct.mkdir(parents=True, exist_ok=True)  # make it so we check it is not same as config dir
    if in_direct.samefile(out_direct):
        raise FileExistsError(f"Cannot copy to same directory {out_direct}")

    # remove all existing files in out_direct
    delete_dir_contents(out_direct)  # remove all existing files in direct
    my_logger.debug(f"Created and cleaned {out_direct}")

    files_to_copy = set(files)  # just the unique files.
    files_copied = []
    for file in files_to_copy:
        in_file = in_direct / file
        if not in_file.exists():
            my_logger.warning(f"File {in_file} does not exist")
            continue

        tgt_path = out_direct / file
        tgt_path.parent.mkdir(parents=True, exist_ok=True)  # make parent dirs
        if in_file.is_dir():
            # work recursively to copy directory
            copy_files(in_file, tgt_path, symlinks=symlinks,
                       files=list(p.relative_to(in_file) for p in in_file.glob('*')))
            files_copied.append(file)
        else:  # its a file
            if tgt_path.exists():  # should not happen as we have deleted everything in direct
                raise FileExistsError(f"File {tgt_path} already exists")
            # rather complex logic to try and create symlink first, then use path.copy if available
            copy_file = True
            if symlinks and try_symlinks:  # want symlinks and no exception from trying symlinks yet
                try:
                    tgt_path.symlink_to(in_file)  # need elev priv on windows so may fail
                    copy_file = False  # have worked so no need to copy
                except OSError as e:
                    my_logger.warning(
                        f"Could not create symlink from {in_file} to {tgt_path}. Error: {e}. Copying instead.")
                    try_symlinks = False  # no more trying symlinks.
            if copy_file:  # need to copy the file
                if hasattr(in_file, 'copy'):
                    in_file.copy(tgt_path)  # works on py 3.14+
                else:
                    shutil.copy2(in_file, tgt_path)  #TODO remove this when we move to python 3.14+ (py pi :-)

            files_copied += [file]  # record relative path copied
            my_logger.debug(f"Copied  {in_file} to {tgt_path} ")

    return files_copied


# AI generated code for locking and then modified.
class ContextFileLock:
    """
    Context manager that acquires an exclusive file lock for the given path.

    Args:
      target_path: path to the resource file you want to protect (the lock file will be target_path + ".lock")
      timeout: number of seconds to wait for the lock. If 0 fail immediately if unable to get lock.
      poll_interval: how frequently to poll internally (forwarded to FileLock's acquire)
    Usage:
      with ContextFileLock("/path/to/config.json", timeout=30):
          # protected region
    """

    def __init__(self, target_path: pathlib.Path, timeout: float = 0.0,
                 poll_interval: typing.Optional[float] = None):
        """

        :param target_path: path to be locked
        :param timeout: timeout interval in seconds. Must be >= 0.0
        :param poll_interval:  polling interval in seconds
        """
        if timeout < 0:
            raise ValueError(f"timeout={timeout} must be non-negative")
        if poll_interval is None:
            poll_interval = max(timeout / 10, 0.01)
        if poll_interval <= 0.0:
            raise ValueError(f"poll_interval={poll_interval} must be positive")
        poll_interval: float  # poll_interval is a float now as None been dealt with.
        lock_path = target_path.with_suffix(target_path.suffix + ".lock")  # lock file is target file with .lock suffix
        self._lock = filelock.FileLock(lock_path, timeout=timeout, poll_interval=poll_interval)
        self._acquired = False

    def __enter__(self):
        self._lock.acquire()
        self._acquired = True
        my_logger.debug(f"Acquired lock for {self._lock.lock_file}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            try:
                self._lock.release()
            finally:
                self._acquired = False

    @property
    def lockfile_path(self) -> pathlib.Path:
        return pathlib.Path(self._lock.lock_file)

    @property
    def is_locked(self) -> bool:
        # FileLock keeps internal state; this mirrors whether this object thinks it's locked.
        return self._acquired

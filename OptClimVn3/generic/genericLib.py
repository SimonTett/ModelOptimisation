"""
 Library of functions that can be called from any module in OptClim
   - get_default -- get default value. like get method to dict.
   - copyDir: Copy directory recursively in a sensible way to allow testing

"""
import errno
import os
import shutil
import stat
import pathlib

import numpy as np
import pandas as pd
from StudyConfig import OptClimConfigVn3
import logging
import copy
def fake_fn(config:OptClimConfigVn3, params: dict) -> pd.Series:
    """
    Wee test fn for trying out things.
    :param config -- configuration. Provides, parameter min, max & ranges and targets.
    :param params -- dict of parameter values
    returns  "fake" data as a pandas Series
    """
    params = copy.deepcopy(params)
    logging.debug("faking with params: " + str(params))
    # remove ensembleMember param.
    params.pop('ensembleMember', None) # remove ensembleMember as a key.

    pranges = config.paramRanges()
    tgt = config.targets()
    min_p = pranges.loc['minParam', :]
    max_p = pranges.loc['maxParam', :]
    scale_params = max_p - min_p
    param_series = pd.Series(params).combine_first(config.standardParam()) # merge in the std params
    pscale = (param_series - min_p) / scale_params
    pscale -= 0.5  # tgt is at params = 0.5
    result = 100 * (pscale + pscale ** 2)
    # this fn has one minima and  no maxima between the boundaries and the minima. So should be easy to optimise.
    result = result.to_numpy()
    delta_len = len(tgt) - result.shape[-1]
    if delta_len > 0:
        result = np.append(result, result[0:delta_len], axis=-1)  # increase result
    result = pd.Series(result, index=tgt.index)  # brutal conversion to obs space.
    var_scales = 10.0 ** np.round(np.log10(config.scales()))
    result /= var_scales  # make sure changes are roughly right scales.

    result += tgt
    return result

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

def genSeed(param):
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

'''
 Library of functions that can be called from any module in OptClim
    gatherNetCDF -- read data from postprocessed netcdf file.
   - get_default -- get default value. like get method to dict.
   - copyDir: Copy directory recursively in a sensible way to allow testing

'''
import errno
import os
import shutil
import stat

import netCDF4
import numpy as np
import pandas as pd

__version__ = '0.2.0'
# subversion properties
subversionProperties = \
    {
        'Date': '$Date$',
        'Revision': '$Revision$',
        'Author': '$Author$',
        'HeadURL': '$HeadURL$'
    }

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
    import os
    import shutil
    import stat
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


def gatherNetCDF(runFile, obsNames, constraintName=None, trace=False, allowNan=False):
    """ Read data from netcdf runFile and
        return simulated observations and constraint as pandas series
     :param runFile - name of netcdf file containing simulated observables
     :param obsNames  list of observations wanted
     :param constraintName (optional) name of constraint wanted
     :param trace  (optional with default  False). If true produce more output
     :param allowNan (optional with defaiut False). If true return nan where no data for variable (or file missing everything is nan)
    """
    if (trace): print("reading from ", runFile)
    use_constraint = constraintName is not None
    ofile = netCDF4.Dataset(runFile,"r")  # open it up for reading using python exception handling if want to trap file not existing
    simobs = []
    for iv in obsNames:  # loop over variables
        if iv in ofile.variables:
            simobs.append(ofile.variables[iv][0])
        else:
            simobs.append(np.nan)

    simobs = np.array(simobs)
    if use_constraint:
        constraint = ofile.variables.get(constraintName, [np.nan])[0]
    ofile.close()  # close netcdf file.

    if not allowNan:  # check for nan in arrays and fail..
        if np.any(np.isnan(simobs)): raise ValueError("Nan in simobs for vars ")
        if use_constraint and np.isnan(constraint): raise ValueError("Nan in constraint")

    # wrap results up as pandas series.
    # constraint only done if constraintName passed.
    simobs = pd.Series(simobs, index=obsNames)
    if use_constraint:
        constraint = pd.Series(constraint, index=[constraintName])
    else:
        constraint = None

    return simobs, constraint


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

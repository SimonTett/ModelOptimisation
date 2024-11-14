# provides classes to handle namelist related configurations and abstract way of storing namelist information.
"""

namelists are indexed by file, namelist_name, var_name

Models have lots of different ways of handling configurations. This module provides abstractions to handle that.
broadly if your model has its own way of doing configurations you will need to add new classes to this  module.
See base_config for base (recommend you inherit from this)
 Module provides json, fortran namelist & rose methods:
 json config_json
 fortran_namelist config_fortran_namelist
 rose_namelist config_rose_namelist

This module assumes something else (Model) is handling the control so getting the right configurations.
And its own read_param() handles caching -- checks if the required file already loaded and if not loads it.
Similarly, handles updating. That may cause a potential problem in that parameter change modifies something in the model
which read then uses. So non-deterministic in that order might matter...
For now this does not matter.
 """
import copy
import dataclasses
import json
import logging
import pathlib
import shutil
import tempfile
import f90nml
import numpy as np
import typing

from model_base import model_base

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

type_allowed_fortran = typing.Union[int, float, bool, str]  # types allowed in fortran


class BaseConfig:
    """
    base class for configurations. Should not be instantiated. Here to provide std things which are overwritten.
    If need to add methods which could be done by a dict put them here!
    """

    def __init__(self, rel_filepath: pathlib.Path,
                 root_dir: pathlib.Path = pathlib.Path.cwd(),
                 allow_missing:bool = False):
        """
        :param rel_filepath: relative path to file containing namelists.
        :param root_dir: root_dir.

        Keep these two separate as rel_filepath comes from namelist & root_dir from model.
        """

        self.rel_filepath = rel_filepath
        self.root_dir = root_dir
        self.modified = False
        self.config = self.read(allow_missing=allow_missing)
        self.modified_values = dict()
        # track what was modified. If something updates or writes a modified value then trigger an error

    # utility methods.
    def filepath(self) -> pathlib.Path:
        """
        Work out the full path to the on disk configuration.
        :return: full path
        """
        return self.root_dir / self.rel_filepath

    def check_right_nl(self, namelist: 'namelist_var'):
        """
        Check the namelist and config match! Will fail with ValueError
        :param namelist: namelist to check
        :return: Nada
        """
        if self.rel_filepath != namelist.filepath:
            raise ValueError(f'Looking in the wrong config. f{self.rel_filepath} != {namelist.filepath} for {namelist}')

    def check_ok(self,namelist):
        """
        Check that OK -- right filepath and not modified the variable.
        :param namelist: namelist we are checking for
        :return: nade -- raises errors
        """
        if self.modified_values.get(namelist,False): # already modified this value trigger an error
            ValueError(f'Attempting to use modified value of {namelist}. You will need to modify code to allow this.')
        # check that namelist.filepath is the same as self.relpath
        self.check_right_nl(namelist)

    def backup(self,
               path: typing.Optional[pathlib.Path],
               backup: bool = True) -> pathlib.Path:
        """
        Work out path to write config too. If it exists backup file first(if backup True).
        :param path: path -- if None will be self.filename.
        :param backup: If True and file exists. Backup the file by renaming it to path.bak.
        :return: path to write data to
        """

        if path is None:
            path = self.filepath()  # will be overwriting the original
        if backup and path.exists():
            backup_name = path.parent / (path.name + '.bak')
            path.rename(backup_name)
            my_logger.debug(f'Renamed {path} to {backup_name}')

        return path


    # end of utility functions.

    # template classes for inherited classes. All raise NotImplementedError if called as need to
    # be written for each type of config.

    def read(self,allow_missing:bool = False):
        """
        Should be overwritten by classes that inherit
        :return:
        """
        raise NotImplementedError('Implement read for your class -- do not call the abstract class.')

    def write(self, path: typing.Optional[pathlib.Path],
              backup: bool = True):
        """
        Should be overwitten by classes that inherit
        :param path:
        :param backup:
        :return:
        """
        raise NotImplementedError('Implement write for your class -- do not call the abstract class.')

    def read_value(self, namelist,
                   raise_error: bool = True) -> type_allowed_fortran:
        """
        Read the  value from the config
        :param namelist: namelist to use
        :param raise_error: if True and namelist not found raise an error.
        :return: value
        """

        raise NotImplementedError('Implement read_value for your class -- do not call the abstract class')

    def update_value(self, namelist,
                     value,
                     create: bool = False):
        """
        Update the config with the values in namelist

        :param namelist: namelist
        :param value: value to set namelist in the config too.
        :param create: If True create values. if False fail if values do not exist.
        """
        raise NotImplementedError('Implement update_value for your class -- do not call the abstract class')

    # methods to return some information about the config.
    def namelist_names(self) -> list[str]:
        """
        Get the namelist names
        :return: a list of the namelist  names
        """
        namelists = [k for k in self.config.keys()]
        return namelists

    def var_names(self, namelist: str) -> list[str]:
        """
        Get the variable names in a namelist
        :param namelist: namelist to query. if does not exist will raise KeyError
        """
        var_names = [k for k in self.config[namelist].keys()]
        return var_names


class JsonNamelistConfig(BaseConfig):
    """
    Class to handle json namelist files.
    Base class provides much of what is needed including the __init__
    """

    def read(self,allow_missing:bool = False) -> dict:
        """
        Read in json config -- which will be a dict.
        :param allow_missing: Allow filepath to not exist. Create empty dict if it does not exist.
        :return: json config
        """
        filepath = self.filepath()
        try:
            with filepath.open('r+t') as fp:
                config = json.load(fp)
        except FileNotFoundError:
            if allow_missing:
                my_logger.warning(f'File {filepath} does not exist. Making empty config')
                config = {} # empty dict
            else:
                my_logger.warning(f'File {filepath} does not exist. Try setting allow_missing=True')
                raise  # raise the error

        return config

    def write(self, path: typing.Optional[pathlib.Path],
              backup: bool = True):
        """
        Write out the config (which is a dict) to a file.
        :param path: path  to write namelist info as a json file. if not specified will be
          constructed using self.backup()
        :param backup: passed to self.backup
        """
        path = self.backup(path,backup=backup)
        with path.open('w+t') as fp:
            json.dump(self.config,fp,indent=2)


    def read_value(self, namelist,
                   raise_error: bool = True) -> type_allowed_fortran:
        """
        Read value from config.
        :param namelist:
        :param raise_error: If True raise an error.
        :return: value.
        """
        self.check_ok(namelist)
        value = self.config.get(namelist.namelist, {}).get(namelist.nl_var, namelist.default)
        if value is None:
            if raise_error:
                raise KeyError(f'Failed to find {namelist} in {self.filepath()}')
            my_logger.debug(f'Failed to find {namelist} in {self.filepath()}')
        return value

    def update_value(self, namelist,
                     value:type_allowed_fortran,
                     create: bool = False): # update value might call the super class which takes an optional init type??
        """
        Update values in config
        :param namelist: namelist
        :param value: value to be updated
        :param create: Create the key if needed.
        :return:
        """

        self.check_ok(namelist)
        if create:
            if self.config.get(namelist.namelist) is None:
                self.config[namelist.namelist] = dict()  # set up an empty dict
        else:
            # try and read it. If var or namelist don't exist  then KeyError will be raised
            try:
                value = self.config[namelist.namelist][namelist.nl_var]
            except KeyError:
                my_logger.warning(f'Failed to find {namelist} in config read from {self.filepath}')

        self.config[namelist.namelist][namelist.nl_var] = value  # set the value
        self.modified_values[namelist] = True  # modified this namelist!
        my_logger.debug(f"Setting {namelist_var} to {value}")


class FortranNamelistConfig(BaseConfig):
    """
    Class to handle fortran namelist files.
    Inherits from base_config_namelist. No __init__ is provided as base class is fine!

    """

    def read(self,allow_missing:bool = False) -> f90nml.namelist.Namelist:
        """
        Read in namelist config. Really here as will have different types of configs each with their own way of reading.
        """
        filepath = self.filepath()
        try:
            config = f90nml.read(filepath)
        except FileNotFoundError:
            if allow_missing:
                my_logger.warning(f'File {filepath} does not exist. Making empty config')
                config =f90nml.namelist.Namelist()  # empty dict
            else:
                my_logger.warning(f'File {filepath} does not exist. Try setting allow_missing=True')
                raise  # raise the error

        return config

    def write(self,
              path: typing.Optional[pathlib.Path],
              backup: bool = True):
        """
        Write out the namelists to the file.
        :param path -- path to write namelist to.
          If not specified will be constructed using self.backup().
        :param backup  passed to self.backup().
        """
        path = self.backup(path, backup=backup)
        f90nml.write(self.config, path, force=True)
        my_logger.info(f'Wrote config to {path}')

    def read_value(self, namelist: 'namelist_var',
                   raise_error: bool = True) -> type_allowed_fortran:
        """
        Read the namelist value from the config.
        :param namelist: namelist to use.
        :param raise_error: if True and namelist not found raise an error.
        :return: value
        """

        self.check_ok(namelist)
        value = self.config.get(namelist.namelist, f90nml.namelist.Namelist()).get(namelist.nl_var, namelist.default)
        if value is None:
            if raise_error:
                raise KeyError(f'Failed to find {namelist} in {self.filepath()}')
            my_logger.debug(f'Failed to find {namelist} in {self.filepath()}')
        return value

    def update_value(self, namelist: 'namelist_var',
                     value: type_allowed_fortran,
                     create: bool = False):
        """
        Update the config with the values in namelist

        :param namelist: namelist
        :param value: value to set namelist in the config too.
        :param create: If True create values. if False fail if values do not exist.
        """
        self.check_ok(namelist)
        if create:
            if self.config.get(namelist.namelist) is None:
                self.config[namelist.namelist] = f90nml.namelist.Namelist()  # set up an empty namelist
        else:
            # try and read it. If var or namelist don't exist  then KeyError will be raised
            try:
                value = self.config[namelist.namelist][namelist.nl_var]
            except KeyError:
                my_logger.warning(f'Failed to find {namelist} in config read from {self.filepath}')
                raise  # raise the error now.

        self.config[namelist.namelist][namelist.nl_var] = value  # set the value
        self.modified_values[namelist] = True # modified this namelist!
        my_logger.debug(f"Setting {namelist_var} to {value}")

@dataclasses.dataclass(frozen=True)
class BaseNamelist:
    """
    Class to handle namelist variables.
    """
    filepath: pathlib.Path
    namelist: str
    nl_var: str
    name: str = None
    default: any = None

    def __repr__(self):
        """ Representation -- the name and the cpts"""
        r = self.__class__.__name__+' '
        r += f"{self.filepath}&{self.namelist} {self.nl_var}"
        if self.default is not None:
            r += f" default:{self.default}"
        if self.name is not None:
            r = f"{self.name}: " + r

        return r

    @classmethod
    def gen_config(cls,root_dir: pathlib.Path,
                   rel_path: pathlib.Path) -> BaseConfig:
        """
        Generate a config of appropriate type by reading one in.
        :param rel_path: relative path to where the config file is
        :param root_dir: root directory for relative paths.
        :return: config
        """
        raise NotImplementedError(f'{cls.__name__}.{__name__} should nto be directly called ')
        config = BaseConfig(rel_path, root_dir=root_dir)
        return config

class JsonNamelistVar(BaseNamelist):
    """
    JSON version of namelists. Same as the base config except gen_config is different
    """
    def gen_config(cls,root_dir: pathlib.Path,
                   rel_path: pathlib.Path) -> JsonNamelistConfig:
        """
        See BaseNamelist.gen_config for argument details. This version retunrs a JsonNamelistConfig

        """
        config = JsonNamelistConfig(root_dir,rel_path)
        return config

@dataclasses.dataclass(frozen=True)
class namelist_var(model_base): # not sure I need to inherit from model_base as no need to write instances out.
    """
    Class to handle namelist variables. Provides several **class** methods.
    """
    filepath: pathlib.Path
    namelist: str
    nl_var: str
    name: str = None
    default: any = None


    def __repr__(self):
        """ Representation -- the name and the cpts"""
        r = f"{self.filepath}&{self.namelist} {self.nl_var}"
        if self.default is not None:
            r += f" default:{self.default}"
        if self.name is not None:
            r = f"{self.name}: " + r

        return r

    _file_cache = dict()  # where we cache files TODO remove.

    @staticmethod
    def gen_config(root_dir: pathlib.Path,
                   rel_path: pathlib.Path) -> FortranNamelistConfig:
        """
        Generate a config of appropriate type by reading one in.
        :param rel_path: rel_path to where the config file is
        :param root_dir: root directory for relative paths.
        :return: config
        """
        config = FortranNamelistConfig(rel_path, root_dir=root_dir)
        return config

    def read_value(self,
                   dirpath: pathlib.Path = pathlib.Path.cwd(),
                   clean: bool = False): # REMOVE
        """ Read  value from disk file containing namelists. Files will be cached to speed up subsequent reads.
        :param dir: Directory where namelist is.
        :param clean -- If True clean the cache before reading.
        """
        namelists = self.file_cache(dirpath / self.filepath, clean=clean)
        try:
            value = namelists[self.namelist].get(self.nl_var.lower(), self.default)
        except KeyError:  # namelist does not exist. Use default value
            value = self.default
            my_logger.info(f"Failed to read {self.namelist} returning {self.default} for {self.nl_var}")
        if value is None:
            raise KeyError(f"{self} not found")
        return value

    def to_dict(self):# REMOVE
        """ Return a dictionary representation, suitable to conversion to JSON,  of namelist-var.  """
        d = dataclasses.asdict(self)
        return d

    # class methods now!
    @classmethod
    def from_dict(cls, dct):# REMOVE
        """
        Generate namelist_var from dictionary
        :param dct: dct
        :return: return  namelist initialised from dict.
        """
        return cls(**dct)

    @classmethod
    def clean_cache(cls):# REMOVE
        """
    Clean out the file cache
        :return:
        """
        cls._file_cache = dict()

    @classmethod
    def file_cache(cls, filepath: pathlib.Path,
                   clean: bool = False, make_copy: bool = True):# REMOVE/move to Model
        """
        Read/cache file
        :param filepath: path to file containing namelist
        :param clean: If True,clean the cache
        :param make_copy: If True, make a deep copy of the cached data
        :return: namelists in the file. Will read in data if needed.
        """
        if clean:
            cls._file_cache = dict()  # reset the cache.
        if filepath in cls._file_cache.keys():
            namelists = cls._file_cache[filepath]
        else:
            namelists = f90nml.read(filepath)
            cls._file_cache[filepath] = namelists
            my_logger.info(f"Read in data from {filepath}")
        if make_copy:
            namelists = copy.deepcopy(namelists)
        return namelists

    @classmethod
    def modify_namelists(cls, nl_info: iter,
                         dirpath: pathlib.Path = pathlib.Path.cwd(),
                         update: bool = False,
                         clean: bool = False) -> dict:# REMOVE
        """
        Update dict indexed by files. Each containing a f90nml namelist.
        Really for internal use by this class.
        :param nl_info -- iterable of namelist, values.
        :param dirpath: path to root directory for namelists
        :param clean: clean cache if set to True
        :param update If true will update namelist from (cached) file system. If not will return nl info for changes.
        :return: dict indexed by the namelist filepath

        Example usage files=files_to_change([(VF1_nl,3.0), (RHCRIT_nl,[0.8,0.8,0.8,...0.9,0.95])],dirpath=pathlib.Path('test_dir'))
        """
        file_dict = {}
        for (nl, value) in nl_info:
            if not isinstance(nl, namelist_var):
                raise ValueError(f"{nl} is {type(nl)} expecting type: namelist_var")
            path = dirpath / nl.filepath
            if path not in file_dict.keys():  # got this file? if not add it in.
                if update:
                    file_dict[path] = cls.file_cache(path, clean=clean)  # not got it so use cache
                    my_logger.debug(f"Updating namelists in {path}")
                else:
                    file_dict[path] = f90nml.namelist.Namelist()  # initialise to empty namelist.
                    my_logger.debug(f"Setting {path} empty")

            if nl.namelist.lower() not in file_dict[path].keys():
                my_logger.debug(f"Setting {nl.namelist.lower()} to empty")
                file_dict[path][nl.namelist.lower()] = f90nml.namelist.Namelist()

            file_dict[path][nl.namelist.lower()][nl.nl_var.lower()] = value
            if isinstance(value, np.ndarray):  # convert numpy arrays.
                file_dict[path][nl.namelist.lower()][nl.nl_var.lower()] = value.tolist()
            my_logger.debug(f"Setting {nl}  to {value}")
        return file_dict

    @classmethod
    def nl_modify(cls, nl_info: iter, dirpath=pathlib.Path.cwd()):# REMOVE
        #TDOO move to model as how it is done is rather model specific.
        """
        Modifiy namelist files. Sadly f90nml.patch() is a bit flaky.
          So, for each file we read in the entire contents. 
         Update using the changes, write to a temp file, remove the input file
            and move the temp file to the original location.
        :param nl_info: iterable of namelist_var, value pairs,
          will also clear cache after all modification done.
        :return: nada though all files used will be modified.

        Example usage namelist_var.nl_modify({VF1_nl:0.5,ENTCOEF_nl:[0.8,0.8,0.85,...0.9,0.95])
        """

        namelists = cls.group_namelists(nl_info)
        for fpath, nl_patch in namelists.items():
            filepath = dirpath / fpath
            bak_file = filepath.with_name(filepath.name + ".bak")
            shutil.copy2(filepath, bak_file, follow_symlinks=False)  # keep symlinks as symlinks.
            my_logger.debug(f" {filepath} copied to {bak_file}")
            with tempfile.NamedTemporaryFile(dir=dirpath, delete=False, mode='w') as tmpNL:
                # control how namelist is output.
                nl_patch.end_comma = True
                nl_patch.uppercase = True
                nl_patch.logical_repr = ('.FALSE.', '.TRUE.')  # how to represent false and true
                full_nl = f90nml.read(filepath)
                full_nl.update(**nl_patch)
                f90nml.write(full_nl, tmpNL)
                tmpNL.close()
            filepath.unlink()  # remove the input file
            pathlib.Path(tmpNL.name).rename(filepath)  # move temp file to original location.
            my_logger.info(f"Modified {filepath}")
        cls.clean_cache()  # cache now "dirty" (been modified) and so needs to  be cleaned.
        return True  # modification succeeded

    grouped_nl = dict[str, dict[str, f90nml.namelist.Namelist]]  # type hint for grouped namelists

    @staticmethod
    def group_namelists(
            nl_info: list[tuple['namelist_var', typing.Union[type_allowed_fortran, list[type_allowed_fortran]]]],
            input_file_dict: typing.Optional[grouped_nl] = None) -> grouped_nl:
        """
        Group together namelist info by filepath and namelist name.
        :param nl_info -- iterable of namelist, values.
        :param input_file_dict -- if provided updates values (modifying input_file_dict as a side effect).
        :return: dict indexed by the namelist filepath with
        values being a f90nml Namelist (which contains all the updated namelist info).
        Example usage grouped_nl =self.group_namelist([(VF1_nl,3.0), (RHCRIT_nl,[0.8,0.8,0.8,...0.9,0.95])])

        """
        if input_file_dict is None:
            file_dict = {}
            my_logger.debug('Initialising file_dict to empty')
        else:
            file_dict = input_file_dict
            my_logger.debug('Using input_file_dict')
        for (nl, value) in nl_info:
            path = nl.filepath
            if path not in file_dict.keys():  # Initialise file_dict[path]
                file_dict[path] = f90nml.namelist.Namelist()
                my_logger.debug(f"Initialised {path}")

            if nl.namelist not in file_dict[path].keys():  # not got this namelist name so initialise it
                file_dict[path][nl.namelist] = f90nml.namelist.Namelist()
                my_logger.debug(f'Initialised {path}{nl.namelist}')

            file_dict[path][nl.namelist][nl.nl_var] = value
            if isinstance(value, np.ndarray):  # convert numpy arrays.
                file_dict[path][nl.namelist][nl.nl_var] = value.tolist()
            my_logger.debug(f"Setting {nl} to {value}")
        return file_dict

    @staticmethod
    def to_fortran(value: type_allowed_fortran) -> str: #TODO  Move to rose_config
        """
        Convert scaler value to a fortran string. Has to handle int, float, bool & str
        :param value: value to convert
        :return: string
        """
        if isinstance(value, (int, float)):
            v = str(value)
        elif isinstance(value, bool):
            v = '.true.' if value else '.false.'
        elif isinstance(value, str):
            v = "'" + value + "'"  # quotes around it.
        else:
            raise ValueError(f"Unsupported type {type(value)}")
        return v

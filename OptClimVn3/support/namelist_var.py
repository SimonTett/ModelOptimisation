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

from numpy.distutils.misc_util import rel_path

from model_base import model_base

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

type_allowed_fortran = typing.Union[int, float, bool, str,list[int,float,bool,str]]  # types allowed in fortran


def register_class_info(name: typing.Union[str,list[str]], init_func: typing.Callable):
    """
    Register a namelist class via name and also provide the init_function to create the relevant config class.
    :param name: Name of the class
    :param init_func: Init function for the related config class.
    :return: A decorated class with config_init and class_init updated.
    """
    if isinstance(name, str): # convert to a list if need be.
        name = [name]
    def decorator(cls):
        # Ensure the class has a superclass with the required attribute
        if not (hasattr(cls, 'config_init') and hasattr(cls, 'class_init')):
            raise AttributeError(f"Superclass of {cls.__name__} must have config_init  & class_init attributes")

        # Add the information to the superclass's class_info attribute
        cls.config_init[cls] = init_func # the init function for the config class.

        for n in name: # add all the known names to the class_init
            cls.class_init[n] = cls # this is init function for the generated class.

        return cls
    return decorator

## Stuff for namelist vars.
# These are data type with named fields. To extend you need to inherit from Namelistvar and add new fields.
# You  also need to do the following:
#  1) functions you write produce the correct types
#  2) modify param_info.read_from_file to generate the right tu
#

@dataclasses.dataclass(frozen=True)
class NamelistVar:
    """
    Class to handle namelist variables.
    """
    type_name:str # name of the namelist_var which tells us what config to use.
    filepath: pathlib.Path # relative path to the file being use
    namelist: str # name of the namelist
    nl_var: str # name of the namelist variable
    name: str = None # parameter name
    default: any = None # default value

    # ** class variables **
    config_init=dict() # Keys are namelist class and values are __init__ fn to create appropriate config.
    class_init=dict() # Keys are names and values are __init__ fn to create appropriate namelist_var.
    # Keys are the cls name  and values are the name and classes which do the reading/writing.
    def __repr__(self):
        """ Representation -- the name and the cpts"""
        r = self.type_name
        r += f"-{self.filepath}: {self.namelist}&{self.nl_var}"
        if self.default is not None:
            r += f" default:{self.default}"
        if self.name is not None:
            r = f"{self.name}: " + r

        return r


    def init_config(self,root_dir: pathlib.Path
                   ,**kwargs) -> "BaseConfig":
        """
        Generate a config of appropriate type by reading one in.
        Looks up init function in class_init and uses that to generate the config.
        :param root_dir: root directory for relative paths.
        All other kwargs are passed to the init function.
        :return: A config of the appropriate type.
        """
        try:
            init_fn = BaseConfig.config_init[self.type_name]

        except KeyError:
            keys = BaseConfig.config_init.keys()
            my_logger.warning(f'Failed to find {self.type_name} in class_info. Allowed names are {", " .join(keys)}. Did you register it?')
            raise
        config = init_fn(root_dir,self.filepath,type_name=self.type_name,**kwargs) # call the init function
        return config



"""
Stuff for handling configurations. These go with coupled objects for namelist variables (or equivalent).
To add a new type of configuration you need to add a new class which inherits from BaseConfig.
You will need to implement read, write, read_value and update_value methods.
Use the decorator register_class_info to register the class with the name of the namelist_var.
You need to define the mapping from parameters to configuration changes. 
Two ways -- define functions (and register them with register_param) or read them in from a csv file.
See HadCM3 for examples which use namelist_var. 
"""
def register_class_info(name: typing.Union[str,list[str]]):
    """
    Register a config  class via name
    :param name: Name of the related namelist_var
    :return: A decorated class with  class_init updated.
    class_init is a dictionary with keys being the name and values being the __init__ function to create the configuration.
    """
    if isinstance(name, str): # convert to a list if need be.
        name = [name]
    def decorator(cls):
        # Ensure the class has a superclass with the required attribute
        if not (hasattr(cls, 'config_init')):
            raise AttributeError(f"Superclass of {cls.__name__} must have config_init attributes")

        # Add the information to the superclass's class_info attribute


        for n in name: # add all the known names to the class_init
            cls.config_init[n] = cls
            # this is init function for the generated config. Here for namelists to generate
            # a config based on their type_name  .

        return cls
    return decorator
class BaseConfig:
    """
    base class for configurations. Should not be instantiated. Here to provide std things which are overwritten.
    If need to add methods which could be done by a dict put them here!

    """
    config_init=dict() # Keys are namelist class and values are __init__ fn to create appropriate config.
    def __init__(self,   root_dir: pathlib.Path = pathlib.Path.cwd(),
                 rel_filepath: pathlib.Path = None,
                 allow_missing:bool = False,
                 type_name: typing.Optional[str] = None):
        """
        :param rel_filepath: relative path to file containing namelists.
        :param root_dir: root_dir.

        Keeping these two separate variables as rel_filepath comes from namelist & root_dir from model.
        """

        self.rel_filepath = rel_filepath
        self.root_dir = root_dir
        self.config = self.read(allow_missing=allow_missing)
        self.modified_values = dict()
        # Track what was modified. If something updates or writes a modified value then trigger an error.
        self.type_name = type_name # expected type_name. Complain if not right.


    # utility methods.
    def filepath(self) -> pathlib.Path:
        """
        Work out the full path to the on disk configuration.
        :return: full path
        """
        return self.root_dir / self.rel_filepath

    def check_right_nl(self, namelist: NamelistVar) -> bool:
        """
        Check the namelist and config match! Will fail with ValueError if Not and return True if OK.
        :param namelist: namelist to check
        :return: Nada
        """
        if (self.type_name is not None) and (self.type_name != namelist.type_name):
            raise ValueError(f'Expected {self.type_name}  for {namelist}.')
        if self.rel_filepath != namelist.filepath:
            raise ValueError(f'Looking in the wrong config. f{self.rel_filepath} != {namelist.filepath} for {namelist}')
        return True

    def check_ok(self,namelist) -> bool:
        """
        Check that config  OK -- namelist compatible and not modified the variable.
        :param namelist: namelist we are checking for
        :return: True if OK
        """

        # check if modified the variable already and throw an error if so.
        if self.modified_values.get(namelist,False): # already modified this value trigger an error
            raise ValueError(f'Attempting to use modified value of {namelist}. You will need to modify code to allow this.')
        # check that namelist.filepath is the same as self.relpath
        ok = self.check_right_nl(namelist)
        return ok

    def backup(self,
               path: typing.Optional[pathlib.Path]=None,
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
            backup_path = path.parent / (path.name + '.bak')
            path.rename(backup_path)
            my_logger.debug(f'Renamed {path} to {backup_path}')

        return backup_path


    # end of utility functions.

    # methods to return some information about the config.
    def namelist_names(self) -> list[str]:
        """
        Get the namelist names
        :return: a list of the known namelist names
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


    # template classes for inherited classes. All raise NotImplementedError if called as need to
    # be written for each type of config.
    def read(self,allow_missing:bool = False):
        """
        Should be overwritten by classes that inherit
        :return:
        """
        raise NotImplementedError('Implement read for your class -- do not call the abstract class.')

    def write(self,
              path: typing.Optional[pathlib.Path]=None,
              backup: bool = True):
        """
        Should be overwitten by classes that inherit
        :param path: path to write out to.
        :param backup:If True backup the file if it exists
        :return:the path written to
        """
        raise NotImplementedError('Implement write for your class -- do not call the abstract class.')

    def read_value(self, namelist:NamelistVar,
                   raise_error: bool = True) -> type_allowed_fortran:
        """
        Read the  value from the config
        :param namelist: namelist to use
        :param raise_error: if True and namelist not found raise an error.
        :return: value
        """

        raise NotImplementedError('Implement read_value for your class -- do not call the abstract class')

    def update_value(self, namelist:NamelistVar,
                     value,
                     create: bool = False):
        """
        Update the config with the values in namelist

        :param namelist: namelist
        :param value: value to set namelist in the config too.
        :param create: If True create values. if False fail if values do not exist.
        """
        raise NotImplementedError('Implement update_value for your class -- do not call the abstract class')



@register_class_info('json_nl')
class JSON_Config(BaseConfig):
    """
    Class to handle json namelist files.
    Base class provides much of what is needed, including  __init__ .
    This provides read, write, read_value and update_value methods.
    """
    # class variables
    indent = 2  # how much to indent json files by. Change if you want different indention.
    default = None # What gets called to serialize objects. None is std.


    def read(self,allow_missing:bool = False) -> dict:
        """
        Read in JSON config -- which will be a dict.
        :param allow_missing: Allow filepath to not exist. Create empty dict if it does not exist.
        :return: JSON config as dict
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

    def write(self, path: typing.Optional[pathlib.Path]=None,
              backup: bool = True):
        """
        Write out the config (which is a dict) to a file.
        :param path: where to write namelist info as a JSON file. if not specified will be
          constructed using self.backup() which will backup the file. .
        :param backup: passed to self.backup.
        """
        path = self.backup(path,backup=backup)
        with path.open('w+t') as fp:
            json.dump(self.config,fp,indent=self.indent,default=self.default)


    def read_value(self, namelist: 'NamelistVar',
                   raise_error: bool = True) -> type_allowed_fortran:
        """
        Read value from config.
        :param namelist:
        :param raise_error: If True raise an error if Value not found.  (othewise return None)
        :return: value.
        """
        self.check_ok(namelist)
        value = self.config.get(namelist.namelist, {}).get(namelist.nl_var, namelist.default)
        if value is None:
            if raise_error:
                raise KeyError(f'Failed to find {namelist} in {self.filepath()}')
            my_logger.warning(f'Failed to find {namelist} in {self.filepath()}')

        return value

    def update_value(self, namelist: 'NamelistVar',
                     value:type_allowed_fortran,
                     create: bool = False):
        """
        Update values in config
        :param namelist: namelist
        :param value: value to be updated
        :param create: Create the key if needed. Generally should not be set.
        :return:
        """

        self.check_ok(namelist)
        if create and self.config.get(namelist.namelist) is None:
            self.config[namelist.namelist] = dict()  # set up an empty dict

        try:
            self.config[namelist.namelist][namelist.nl_var] = value  # set the value
        except KeyError:
            my_logger.warning(f'Failed to find {namelist} in config read from {self.filepath}')
            raise

        self.modified_values[namelist] = True #  namelist has been modified.
        my_logger.debug(f"Setting {namelist_var} to {value}")

@register_class_info(['namelist_var','fortran_nl']) # have namelist_var for compatibility with earlier code.
class FortranNamelistConfig(BaseConfig):
    """
    Class to handle fortran namelist files.
    Inherits from base_config_namelist. No __init__ is provided as base class is fine!

    """
    # Control how namelist is output. Default values as call variables. Modify them if you want to change how
    # writing is done.
    end_comma = True
    uppercase = True
    def read(self,allow_missing:bool = False) -> f90nml.namelist.Namelist:
        """
        Read in namelist config. Really here as will have different types of configs each with their own way of reading.
        """
        filepath = self.filepath()
        try:
            parser = f90nml.parser.Parser()
            # Here one could modify the parser to change how I/O done. But that is not needed yet!
            config = parser.read(filepath)
        except FileNotFoundError:
            if allow_missing:
                my_logger.warning(f'File {filepath} does not exist. Making empty config')
                config =f90nml.namelist.Namelist()  # empty dict
            else:
                my_logger.warning(f'File {filepath} does not exist. Try setting allow_missing=True')
                raise  # raise the error

        return config

    def write(self,
              path: typing.Optional[pathlib.Path] = None,
              backup: bool = True):
        """
        Write out the namelists to the file.
        :param path -- path to write namelist to.
          If not specified will be constructed using self.backup().
        :param backup  passed to self.backup().
        """
        path = self.backup(path, backup=backup)
        config_to_write = copy.copy(self.config)  # make a copy so we can modify it.
        # Modify parameters of the config to control how it is written.
        config_to_write.end_comma = self.end_comma
        config_to_write.uppercase = self.uppercase
        config_to_write.write(path, force=True)  # force overwriting of file.
        my_logger.info(f'Wrote config to {path}')

    def read_value(self,
                   namelist: NamelistVar,
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

    def update_value(self,
                     namelist:  NamelistVar,
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


class GroupConfig(model_base):
    # class to handle a group of configs. Really just a dict of configs + root_dir

    def __init__(self, root_dir: pathlib.Path):
        """

        :param root_dir: The root directory for relative paths.
        """
        self.configs: dict[pathlib.Path, BaseConfig] = dict()
        self.root_dir = root_dir

    def load_config(self, namelist: NamelistVar):
        """
        Load in the config for the namelist if not present.
        :param namelist: namelist to load in.
        """
        if namelist.filepath not in self.configs.keys():
            config = namelist.init_config(self.root_dir)
            self.configs[config.rel_filepath] = config

    def read_value(self, namelist: NamelistVar, raise_error: bool = True) -> type_allowed_fortran:
        """
        Read value from  config .
        :param namelist: Namelist to be read in.
        :param raise_error -- If True raise an error if value not found.
        :return: Value read in.
        Side effect -- will load in data to config if not already loaded.
        """
        self.load_config(namelist)
        value = self.configs[namelist.filepath].read_value(namelist, raise_error=raise_error)
        return value

    def update_value(self, namelist: NamelistVar, value: type_allowed_fortran):
        """
        Update the value in the config.
        :param namelist: namelist being updated
        :param value: value to set
        :return: Name

        Side effect -- will load in data to config if not already loaded.
        """
        self.load_config(namelist)
        self.configs[namelist.filepath].update_value(namelist, value)

    def write_values(self, nl_values: dict[NamelistVar, type_allowed_fortran]):
        """
        Set & write the values in the configs.
        :param nl_values: dictionary indexed by NamelistVar with values to set.

        After writing the configs are reset.
        """
        for namelist, value in nl_values.items():  # update values
            self.update_value(namelist, value)

        for config in self.configs.values():  # Iterate over configs and write them out.
            config.write()
        self.configs = dict()  # reset the configs.

    def to_dict(self) -> dict:
        """
        Return a dictionary representation of the GroupConfig
        Configs are not cached.
        :return: dictionary
        """
        dct = vars(self)
        dct['configs'] = dict()  # no need to serialise  the cached configs.
        return dct



## When done all below can be removed.

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

# provides classes to handle namelist related configurations and abstract way of storing namelist information.
"""

namelists are indexed by file, namelist_name, var_name

Models have lots of different ways of handling configurations. This module provides abstractions to handle that.
broadly if your model has its own way of doing configurations you will need to add new classes to this  module.
See base_config for base (recommend you inherit from this)
 Module provides json, fortran namelist & rose methods:

 type_name        config class
 -------------------------------
 json_nl          JSON_Config
 fortran_nl       FortranNamelistConfig
 um_rose         config_rose_namelist

Namelist variables are encapsulated in NamelistVar. See that for description.
Key is the type_name which links to a registered class which handles I/O of the namelist variables.

If you want a new way of dealing with I.O for your namelist variables then you should subclass BaseConfig and
  use the register_class_info decorator. See BaseConfig for what is needed when subclassing.


tge GroupConfig class handles multiple files each, in principle, with their own way of doing I/O.
In practice, it is likely that all files use the same I/O approach.

 """
import copy
import dataclasses
import json
import logging
import pathlib
import f90nml
import typing
from model_base import model_base
import metomi.rose.config

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

type_allowed_fortran = typing.Union[int, float, bool, str,list[int,float,bool,str]]  # types allowed in fortran



## Stuff for namelist vars.


@dataclasses.dataclass(frozen=True)
class NamelistVar:
    """
    Class to handle namelist variables.
    A namelist is defined from relative filepath, namelist name and variable name. Also includes name,type_name and default.
You might be able to extend this if you need something more complex.
    """
    type_name:str # name of the namelist_var which tells us what kind of namelist we are .
    filepath: pathlib.Path # relative path to the file where namelist lives
    namelist: str # name of the namelist
    nl_var: str # name of the namelist variable
    name: str = None # parameter name
    default: type_allowed_fortran = None # default value

    # ** class variables **
    config_init=dict() # Keys are namelist class and values are __init__ fn to create appropriate config.
    class_init=dict() # Keys are names and values are __init__ fn to create appropriate namelist_var.
    # Keys are the cls name  and values are the name and classes which do the reading/writing.
    def __repr__(self):
        """ Representation -- the name and the cpts"""
        r = self.type_name
        r += f":{self.filepath}: {self.namelist}&{self.nl_var}"
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
        config = init_fn(root_dir,self.filepath,**kwargs) # call the init function
        return config

    def to_dict(self) -> dict:
        """ Return a dictionary representation, suitable to conversion to JSON,  of namelist-var.  """
        d = dataclasses.asdict(self)
        return d

"""
Stuff for handling configurations. These go with NameListVar. They are all registered with a name
that matches the type_name in the namelist.
To add a new type of configuration you need to add a new class which inherits from BaseConfig.
You will need to implement read, write, read_value and update_value methods.
Use the decorator register_class_info to register the class with the name of the namelist_var.
Do write some test cases for your new/modified methods.  
You need to define the mapping from parameters to configuration changes. See Model description for guidance. 

"""
def register_class_info(name: typing.Union[str,list[str]]):
    """
    Register a config  class via name.
    :param name: Name or list of Names of the related configurations.
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

        cls.type_name = name
        # this is init function for the generated config. Here for namelists to generate
        # a config based on their type_name  .

        return cls
    return decorator

@register_class_info('base_nl') # needed for testing...
class BaseConfig:
    """
    base class for configurations. Should not be instantiated. Here to provide std things which are overwritten.
    If need to add methods which could be done by a dict put them here!

    """
    config_init=dict() # Keys are namelist class and values are __init__ fn to create appropriate config.
    def __init__(self,   root_dir: pathlib.Path = pathlib.Path.cwd(),
                 rel_filepath: pathlib.Path = None,
                 allow_missing:bool = False):
        """
        :param rel_filepath: relative path to file containing namelists.
        :param root_dir: root_dir.

        Keeping these two separate variables as rel_filepath comes from namelist & root_dir from model.
        """

        self.rel_filepath = rel_filepath
        self.root_dir = root_dir
        self.config=None # config is read in from file.
        self.modified_values = None # track what has been modified.
        self.reset(allow_missing=allow_missing)


    def reset(self,allow_missing:bool = False):
        """
        Reset the config by reading in from file and setting modified_values to empty.
        """
        self.config = self.read()
        self.modified_values = dict()
        # Track what was modified. If something updates or writes a modified value then trigger an error.

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
        if namelist.type_name not in self.type_name:
            raise ValueError(f'For {namelist} expected {namelist.type_name} to be one off {self.type_name}')
        if self.rel_filepath != namelist.filepath:
            raise ValueError(f'Looking in the wrong config. f{self.rel_filepath} != {namelist.filepath} for {namelist}')
        return True

    def check_ok(self,namelist,
                 check_modify:bool = True) -> bool:
        """
        Check that config  OK -- namelist compatible and not modified the variable.
        :param namelist: namelist we are checking for
        :param check_modify: If True check that the variable has not been modified already.
        :return: True if OK
        """

        # check if modified the variable already and throw an error if so.
        if check_modify and self.modified_values.get(namelist,False): # already modified this value trigger an error
            raise ValueError(f'Attempting to use modified value of {namelist}. You will need to modify code to allow this.')
        # check that namelist.filepath is the same as self.relpath
        ok = self.check_right_nl(namelist)
        return ok

    def backup(self,
               path: typing.Optional[pathlib.Path]=None) -> pathlib.Path:
        """
        Work out path to write config too. If it exists backup file first(if backup True).
        :param path: path to write data.-- if None will be self.filename.
        If path exists then it will be backed up to path.bak. If the backup path exists will raise an error.
        :return: path to write data to
        """

        if path is None:
            path = self.filepath()  # will be overwriting the original
        if path.exists():
            backup_path = path.parent / (path.name + '.bak')
            if backup_path.exists():
                raise ValueError(f'Backup file {backup_path} already exists. ')

            path.rename(backup_path)
            my_logger.debug(f'Renamed {path} to {backup_path}')

        return path


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
    def read(self,allow_missing:bool = False) :
        """
        Should be used  by classes that inherit
        :return:
        """
        raise NotImplementedError('Implement read for your class -- do not call the abstract class.')

    def write(self,
              path: typing.Optional[pathlib.Path]=None,
              backup: bool = True) -> pathlib.Path:
        """

         This superclass does backing up!
        :param path: path to write out to.
        :param backup:If True backup the file if it exists
        :return:the path (to be) written to
        """
        if path is None:
            path = self.filepath()
        if backup:
            path = self.backup(path)

        return path

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
    Class to handle json namelist files -- where variables are stored in dicts in a json file.
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
              backup: bool = True) -> pathlib.Path:
        """
        Write out the config (which is a dict) to a file.
        :param path: where to write namelist info as a JSON file. if not specified will be
          constructed using self.backup() which will backup the file. .
        :param backup: passed to self.backup.
        """
        path = super().write(path, backup=backup)

        with path.open('w+t') as fp:
            json.dump(self.config,fp,indent=self.indent,default=self.default)
        my_logger.info(f'Wrote json config to {path}')
        return path

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
        my_logger.debug(f"Setting {namelist} to {value}")

@register_class_info(['namelist_var','fortran_nl']) # have namelist_var for compatibility with earlier code.
class FortranNamelistConfig(BaseConfig):
    """
    Class to handle fortran namelist files.
    Inherits from base_config_namelist. No __init__ is provided as base class is fine!

    """
    # Control how namelist is output. Modify them if you want to change how
    # writing is done. see f90nml namelist.Namelist and f90nml.parser.Parser for allowed properties.
    write_properties = dict(
        end_comma = True,
        uppercase = True
    ) # properties to set when writing out the config.
    read_properties = dict(

    ) # properties to set when reading in the config.


    def __init__(self,   root_dir: pathlib.Path = pathlib.Path.cwd(),
                 rel_filepath: pathlib.Path = None,
                 allow_missing:bool = False):
        """
        Fortran namelist specific init. Needed as want to dsetup a parser depending on read properties.
        :param root_dir:
        :param rel_filepath:
        :param allow_missing:
        """
        parser = f90nml.parser.Parser()
        for attr, value in self.read_properties.items():
            if isinstance(getattr(parser.__class__, attr), property):
                setattr(parser, attr, value)
        self.parser = parser
        super().__init__(root_dir,rel_filepath,allow_missing=allow_missing)


    def read(self,allow_missing:bool = False) -> f90nml.namelist.Namelist:
        """
        Read in namelist config.
        """
        filepath = self.filepath()
        ## code to allow property setting

        try:
            config = self.parser.read(filepath)
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
              backup: bool = True) -> pathlib.Path:
        """
        Write out the namelists to the file.
        :param path -- path to write namelist to.
          If not specified will be constructed using self.backup().
        :param backup  passed to self.backup().
        """
        path = super().write(path, backup=backup) # call the superclass write method.
        config_to_write = copy.copy(self.config)  # make a copy so we can modify it.
        # Modify parameters of the config to control how it is written.
        # different fortran configs may need different parameters.
        # for example GAMIL needs column_width to be 5000. I suspect the UM likes 80!  

        for attr,value in self.write_properties.items():
            if isinstance(getattr(config_to_write.__class__,attr),property):
                setattr(config_to_write,attr,value)

        config_to_write.write(path, force=True)  # force overwriting of file.
        my_logger.info(f'Wrote fortran namelist config to {path}')

        return path

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
                self.config[namelist.namelist][namelist.nl_var] # don't want the val
            except KeyError:
                my_logger.warning(f'Failed to find {namelist} in config read from {self.filepath}')
                raise  # raise the error now.

        self.config[namelist.namelist][namelist.nl_var] = value  # set the value
        self.modified_values[namelist] = True # modified this namelist!
        my_logger.debug(f"Setting {namelist} to {value}")

@register_class_info('gamil3_nl')
class GAMIL3Config(FortranNamelistConfig):
    """
    Class to handle GAMIL namelist files. Just changes write_properties.
    Inherits from FortranNamelistConfig. No __init__ is provided as base class is fine!
    """
    # Control how namelist is output. Modify them if you want to change how
    # writing is done. see f90nml namelist.Namelist (write_properties) and f90nml.parser.Parser (read_properties) for allowed properties.
    write_properties = dict(
        end_comma = True,
        uppercase = True,
        column_width = 5000
    )
    read_properties = dict() # use default f90nml read properties.

@register_class_info('um_rose')
class UMroseNamelistConfig(BaseConfig):
    """
    Class to handle I/O of namelists for UM rose configurations.
    """
    # utility function to convert data to something Fortran namelists expect.
    @staticmethod
    def to_fortran(value: type_allowed_fortran) -> str:
        """
        Convert scaler or list or value to a fortran string. Has to handle int, float, bool & str
        :param value: value to convert
        :return: string
        """
        if isinstance(value,list):
            v = [UMroseNamelistConfig.to_fortran(val) for val in value]
            v = ','.join(v) # and stick them together with ','.
        elif isinstance(value, (int, float)):
            v = str(value)
        elif isinstance(value, bool):
            v = '.true.' if value else '.false.'
        elif isinstance(value, str):
            v = "'" + value + "'"  # quotes around it.
        else:
            raise ValueError(f"Unsupported type {type(value)}")
        return v
    # utility function to convert strings back to values
    @staticmethod
    def parse_value(value:str) -> type_allowed_fortran:
        """
        Parse strings from um_rose config. They are **sort of** fortran namelist text so will
          use f90nml to parse them.
        :param value: str to be parsed,
        :return: type_allowed_fortran (int,float,bool, str or array of such)
        """

        s = f"&temp_nl nl_var = {value} /"
        result = f90nml.reads(s)["temp_nl"]["nl_var"]
        return result

    def read(self,allow_missing:bool = True) -> metomi.rose.config.ConfigNode:
        """
        Read in namelist config. Normally called from __init__
        Really here as will have different types of configs each with their own way of reading.
        """
        filepath = self.filepath()
        try:
            with filepath.open('rt') as fp:
                config = metomi.rose.config.load(fp)
        except FileNotFoundError:
            if allow_missing:
                my_logger.warning(f'File {filepath} does not exist. Making empty config')
                config = metomi.rose.config.ConfigNode()  # empty config
            else:
                my_logger.warning(f'File {filepath} does not exist. Try setting allow_missing=True')
                raise  # raise the error

        return config

    def write(self,
              path: typing.Optional[pathlib.Path]=None,
              backup: bool = True) -> pathlib.Path:
        """
        Write out the UM rose namelists to the file.
        :param path -- path to write namelist to.
          If not specified will be constructed using self.backup().
        :param backup  passed to self.backup().
        """
        path = super().write(path, backup=backup)  # call the superclass write method.
        config_to_write = copy.copy(self.config)  # make a copy so we can modify it.
        path.unlink(missing_ok=True) # need to remove the file as dump errors if the file exists...
        try:
            metomi.rose.config.dump(config_to_write, str(path))  # dump the modified config
        except FileExistsError as err : # windows does not like to rename over an existing path. Sigh. And dump creates it...
            # dump really should not create it. It only does so it can set file permissions...
            # but hacking the code is probably not sensible.... But I might raise an issue.
            # need to extract the filename for the handle
            if err.errno != 17: # we only need to deal with this
                raise  # raise the error
            my_logger.warning('Hacking (probably only on windows) problems with dump which creates a file and then renames tmp file to it')
            # remove filename2 and rename filename1 to filename2
            file= pathlib.Path(err.filename)
            file2= pathlib.Path(err.filename2)
            file2.unlink() #remove the file we want to rename to
            file.rename(file2) # and actually do the rename

        self.reset()
        my_logger.info(f'Wrote UM rose namelist config to {path}')
        return path



    def read_value(self, namelist:NamelistVar,
                   raise_error: bool = True,
                   check_modify:bool = True) -> type_allowed_fortran:
        """
        Read the namelist value from the config.
        :param namelist: namelist to use.
        :param raise_error: if True and namelist not found raise an error.
        :param check_modify: If True check that the variable has not been modified already.
        :return: value
        """
        self.check_ok(namelist,check_modify=check_modify)
        value = self.config.get([namelist.namelist, namelist.nl_var])
        if value is None:
            if raise_error:
                raise KeyError(f'Failed to find {namelist} in {self.filepath()}')
            my_logger.debug(f'Failed to find {namelist} in {self.filepath()}')
            value = namelist.default # return the default value
        elif value.state != self.config.STATE_NORMAL:  # if it is a comment then raise error.
            if raise_error:
                raise ValueError(f'Found comment {value} for {namelist} in {self.filepath()}')
            my_logger.debug(f'Found comment {value} for {namelist} in {self.filepath()}')
            value = namelist.default
        else:
            # need to parse this string to convert to value. This is a bit of a pain.
            value = self.parse_value(value.value) # convert it to numeric value.
        return value


    def update_value(self, namelist:NamelistVar,
                     value,
                     create: bool = False):
        """
        Update the config with the values in namelist

        :param namelist: namelist
        :param value: value to set namelist in the config too.
        :param create: If True create values. if False fail if values do not exist.
        """
        self.check_ok(namelist)
        nl=(namelist.namelist,namelist.nl_var)
        if create:
            if self.config.get([namelist.namelist]) is None:
                self.config[namelist.namelist] = metomi.rose.config.ConfigNode()  # set up an empty namelist
        else:
            # try and read it. If var or namelist don't exist  then None will be returned
            v = self.config.get(nl)
            if v  is None:
                err_msg = f'Failed to find {namelist} in config read from {self.filepath()}'
                my_logger.warning(err_msg)
                raise KeyError # raise the error
            elif v.state != self.config.STATE_NORMAL : # not normal status. Log a warning message.
                my_logger.warning(f'{nl} status={v.status} not Normal. Will be set to normal.')

        self.config.set(keys=nl, value=self.to_fortran(value),state=self.config.STATE_NORMAL)  # set the value -- convert to string. State set to normal.
        self.modified_values[namelist] = True  # modified this namelist!
        my_logger.debug(f"Setting {namelist} to {value}")

class GroupConfig(model_base):
    # class to handle a group of configs. Really just a dict of configs + root_dir

    def __init__(self, root_dir: typing.Optional[pathlib.Path]= None):
        """

        :param root_dir: The root directory for relative paths.
        """
        self.configs: dict[pathlib.Path, BaseConfig] = dict()
        self.root_dir = root_dir

    def load_config(self, namelist: NamelistVar,reload:bool = False):
        """
        Load in the config for the namelist if not present.

        :param namelist: namelist to load in.
        :param reload:  If True then reload the config.
        """
        if namelist.filepath not in self.configs.keys() or reload:
            config = namelist.init_config(self.root_dir)
            self.configs[config.rel_filepath] = config
            my_logger.debug(f'Read in config for {namelist} from {config.rel_filepath}')

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

    def update_value(self, namelist: NamelistVar,
                     value: type_allowed_fortran,
                     create: bool = False):
        """
        Update the value in the config.
        :param namelist: namelist being updated
        :param value: value to set
        :param create: If True create the 'namelist' if it does not exist.
        :return: Name

        Side effect -- will load in data to config if not already loaded.
        """
        self.load_config(namelist)
        self.configs[namelist.filepath].update_value(namelist, value,create=create)

    def write_values(self,
                     nl_values: dict[NamelistVar, type_allowed_fortran],
                     backup: bool = True,
                     create: bool = False):
        """
        Set & write the values in the configs.
        :param nl_values: dictionary indexed by NamelistVar with values to set.
        :param backup: If True backup the file if it exists.
        :param create: If True create the 'namelist' if it does not exist.
        After writing, the configs are reset.
        """
        for namelist, value in nl_values.items():  # update values
            self.update_value(namelist, value,create=create)

        for config in self.configs.values():  # Iterate over configs and write them out.
            config.write(backup=backup)
            my_logger.debug(f'Wrote out {config} to {config.filepath()}')
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



import logging
import os
import pathlib

import generic_json


class model_base:
    """
    Generic base class for models. Provides default methods for reading/writing data.
    Note that it is not expected to actually be instantiated. See sub-classes (Model for exmaple) for larger functionality.

        Methods:
    --------
    from_dict(cls, dct: dict) -> Any:
        Return class initialised from dct by copying keys to attributes in obj but only that exist after initialisation.
        This is really a factory method.

    to_dict(self) -> dict:
        Convert an object to a dict.

    load(cls, class_json: pathlib.Path) -> Any:
        Load an object configuration from specified file. The correct type of object will be returned.

    dump(self, config_path: pathlib.Path) -> Any:
        Write object (as json) to config_path.
    """

    def __init_subclass__(cls, *args, **kwargs):
        # obscure python. See https://peps.python.org/pep-0487/#new-ways-of-using-classes
        """
        initialise the subclass
        First call superclass __init__subclass__ method
        Then store default methods in registries
        :poram args: args to be passed to supper class __init__subclass__
        :param kwargs:keywords to be passed to super class __init__subclass__
        :return: Nada
        """
        super().__init_subclass__(*args, **kwargs)
        # if subclassing you can change the methods but better to define from_dict and to_dict for your own purposes.
        generic_json.obj_to_from_dict.register_FROM_VALUE(cls, cls.from_dict)
        generic_json.obj_to_from_dict.register_TO_VALUE(cls, cls.to_dict)


    # class methods


    @classmethod
    def from_dict(cls, dct: dict):
        """
        Return  class initialised from dct by copying keys to attributes in obj
        but only those that  exist after initialisation.
        Make sure your dct is sensible...This is very generic.
        This is really a factory method
        :param dct: dict containing information needed by class_name.from_dict()
        :return: initialised object
        """
        obj = cls()  # create an default instance
        for name, value in dct.items():
            if hasattr(obj, name):
                setattr(obj, name, value)
            else:
                logging.warning(f"Did not setattr for {name} as not in obj")
        return obj

    def to_dict(self):
        """
        Convert an object to a dict.
        :return: dct
        """
        dct = dict()
        for key, value in vars(self).items():
            dct[key] = value

        return dct

    @classmethod
    def load(cls, file: pathlib.Path):
        """
        Load an object configuration from specified file.
        The correct type of object will be returned.
        :param file path to file to be read in.
        :return: model configuration of appropriate type.
        """
        # read it in.

        with open(file, 'r') as fp:
            cfg = generic_json.load(fp)
            # this runs all the magic needed to create objects that we know about
        for k, v in vars(cfg).items():  # debug info.
            logging.debug(f"{k}: {v} ")
        logging.info(f"Read configuration from {file}")
        return cfg

    def class_name(self):
        """
        Return the class name.
        """
        return self.__class__.__name__

    def dump(self, config_path: pathlib.Path):
        """
        Write object (as json) to config_path
        :param config_path: path to write data to. Directory where this goes will be created if necessary.
        :return:whatever result of generic_json.dump is
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as fp:
            result = generic_json.dump(self, fp, indent=2)  # JSON encoder does the magic needed
        logging.info(f"Wrote to {config_path}")
        return result

    @staticmethod
    def expand(filestr: str) -> pathlib.Path:
        """
        Expand any env vars, convert to path and then expand any user constructs.
        :param filestr: path like string
        :return:expanded path
        """
        path = os.path.expandvars(filestr)
        path = pathlib.Path(path).expanduser()
        return path


from __future__ import annotations



from model_base import model_base
from  param_info import ParamInfo
import logging
import copy
import typing
import pathlib
import genericLib
genericLib.setup_env()

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

# code from David de Klerk 2023-04-13

def register_param(name: str) -> typing.Callable:
    """
    Decorator/function to register a parameter.
    :param name: name of the parameter
    :return:
    """

    # The decorator attaches values to the definition of a function.
    # In this example, we are attaching a boolean tag and a key/value pair.
    def decorator(func):
        setattr(func, '_is_param', True)
        setattr(func, '_name', name)
        # setattr(func, '_value', value)
        return func

    return decorator



type_param_fn = typing.Union[list[tuple['NamelistVar', typing.Union[list[float],float]]], typing.Union[float,list[float]]]

class ModelBaseClass(model_base):
    # T
    """
    A base class for all models. uses __init_subclass__ to setup param_info from superclasses and registered methods.
    See register_param function above which tags functions while the init_subclass uses those tags to put that
    function in the param_info attribute. This class inherits from model base which through its __init_subclass__
    which sets up things for dumping and loading instances to disk as a json file.
    Provides *class* methods to register classes in ModelBaseClass -- so 'global' state.
    This allows right initialisation to be done.
    """
    class_registry = dict()  # where class information for model_init is used. This should be at ModelBaseClass.
    param_info = ParamInfo()
    # Where parameter information is stored. Note is a *class* attribute as instances should have the same parameters
    @classmethod
    def register_functions(cls) -> ParamInfo:
        """
        Register all functions in the class
        :return: param_info to be merged into other information,
        """
        my_param_info = ParamInfo()
        for name, member in cls.__dict__.items():
            # Loop through all members of the subclass and populate param_info
            # accordingly. These should all be functions.
            my_logger.debug(f"Processing member {name}")
            if getattr(member, '_is_param', False):
                param_name = getattr(member, '_name')
                my_param_info.register(param_name, member)
                my_logger.info(f"Registered param {param_name} for {cls.__name__}")
        return my_param_info  # should be merged into rest of param_info.

    def __init_subclass__(cls, **kwargs):
        # The __init_subclass__ uses the values attached by the decorator
        # to update populate param_info.
        # Note, this could go in __init_subclass__ of Model, but then
        # Model won't be able to use the @register_param decorator.
        super().__init_subclass__(**kwargs)

        # Depending on how you want param_info to be inherited there
        # are two options here. With this definition, subclasses won't
        # inherit param_info from their super classes:
        # cls.param_info = {} # Define param_info

        # With this definition, all params in parent classes are duplicated
        # in subclasses.

        my_param_info = ParamInfo()
        for bcls in reversed(cls.__bases__):  # iterate over base classes updating parameters from them.
            parent_param_info = getattr(bcls, 'param_info', ParamInfo())
            my_param_info.update(parent_param_info)  # update overwrites existing info for named parameters.
            my_logger.info(f"Updated param_info from {bcls}")
        if hasattr(cls, 'param_info'):  # Already got param_info. Update from it
            my_param_info.update(cls.param_info)
            my_logger.info(f"Updated param_info from {cls.param_info}")
        my_param_info.update(cls.register_functions())

        cls.param_info = copy.deepcopy(my_param_info)
        ModelBaseClass.register_class(newcls=cls)
        # register the class for subsequent creation. This allows model_init to work.
        my_logger.info(f"Registered {cls.__name__}")

    @classmethod
    def register_class(cls,
                       newcls: typing.Optional[typing.Callable]=None,
                       name: typing.Optional[str] = None) -> typing.Callable:
        """
        Register class
        :param newcls: class to register. If None then class_name will be loaded.
        :param name: name to register it under. If None use the class name.
          If name is XXXX.YYYY then genericLib.get_fn(name) will be called.
              This will load module XXXX and extract fn from it.  The class name will be removed
              from the registry to stop duplication.
        """
        if newcls is None and name is not None:
            newcls = genericLib.get_fn(name)
            # this might register the Model. So remove it.
            c=ModelBaseClass.class_registry.pop(newcls.__name__,None)
            if c is not None:
                my_logger.info(f'Removed class {newcls.__name__}')
        elif name is None:
            name = newcls.__name__
        elif (name is not None) and (cls is not None):
            pass
        else:
            raise ValueError("You must specify either newcls or name")
        my_logger.info(f"Registering class with name {name}")
        ModelBaseClass.class_registry[name] = newcls
        return newcls

    @classmethod
    def remove_class(cls,
                     name: typing.Optional[str]  = None,
                     all_classes:bool = False) -> typing.Any:
        """
        Remove class from registry. By default class.
        :param name. If not None the name of a class to be removed
        :param all. If True all classes will be removed.
        :return: Class removed from register or None
        """

        if all_classes:
            ModelBaseClass.class_registry = dict()
            return None

        if name is None:
            name = cls.__name__

        my_logger.info(f"Removing {name} from registry")
        r = ModelBaseClass.class_registry.pop(name, None)
        if r is None:
            my_logger.warning(f"{name} not in registry")
        return r

    @classmethod
    def known_models(cls):
        """
        Return list of known models.
        :return: list of known models.
        """
        return list(cls.class_registry.keys())

    @classmethod
    def model_init(cls, class_name: str, *args, **kwargs) -> typing.Any:
        """
        Create a model
        :param class_name: name of class to make
        :param args: positional arguments to pass to initiation
        :param kwargs: kwargs to pass to init.
        :return: New model object.
        """
        try:
            newcls = ModelBaseClass.class_registry[class_name] # do we already have it?
            my_logger.debug(f"Loaded {class_name} from registry")
        except KeyError:
            if '.' in class_name:  # Specifying via module.class_name. Will use the module to import the class.
                newcls = ModelBaseClass.register_class(name=class_name) # register the class.
                my_logger.info(f'Loaded {newcls.__name__} from {class_name}')

            else: # not specified by module.
                raise ValueError(f"Failed to find {class_name}. Allowed classes are " + " ".join(cls.class_registry.keys()))

        result = newcls(*args, **kwargs)
        my_logger.debug(f"Created {class_name} with args {args} and kwargs {kwargs}")
        return result



    @classmethod
    def add_param_info(cls, param_info: dict, duplicate=True):
        """
        Add information on parameters and functions.

        :param param_info: A dict with keys variable names and values either a namelist_var or callable.
          You probably should not use a callable here as better to register it when declared.
        :param duplicate If True allow duplicates which will add new namelist/callable info to existing.
        :return: Nothing
        """
        for varname, value in param_info.items():
            cls.param_info.register(varname, value, duplicate=duplicate)
            my_logger.debug(f"Registered {varname} with {value}")

    @classmethod
    def update_from_file(cls, filepath: pathlib.Path, duplicate=True):
        """
        Update **class info** on known parameters from CSV file
         Calls param_info.update_from_file(filepath) to actually do it!
         See documentation for that
        :param filepath: path to csv file
        :param duplicate -- allow duplicates.
        :return:
        """
        cls.param_info.update_from_file(filepath, duplicate=duplicate)




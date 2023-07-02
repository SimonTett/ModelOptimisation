from __future__ import annotations

from model_base import model_base
from  param_info import param_info
import logging
import copy
import typing
import pathlib

# code from David de Klerk 2023-04-13

def register_param(name: str) -> typing.Callable:
    """
    Decorator to register a parameter.
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



class ModelBaseClass(model_base):
    """
    A base class for all models. uses __init_subclass__ to setup param_info from superclasses and registered methods.
    See register_param function above which tags functions while the init_subclass uses those tags to put that
    function in the param_info attribute. This class inherits from model base which through its __init_subclass__
    which sets up things for dumping and loading instances to disk as a json file. TODO -- merge into model_base.
    """
    class_registry = dict()  # where class information for model_init is used.
    #param_info = param_info()

    @classmethod
    def register_functions(cls) -> param_info:
        """
        Register all functions in the class
        :return: param_info to be merged into other information,
        """
        my_param_info = param_info()
        for name, member in cls.__dict__.items():
            # Loop through all members of the subclass and populate param_info
            # accordingly. These should all be functions.
            logging.debug(f"Processing member {name}")
            if getattr(member, '_is_param', False):
                param_name = getattr(member, '_name')
                my_param_info.register(param_name, member)
                logging.info(f"Registered param {param_name}")
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

        my_param_info = param_info()
        for bcls in reversed(cls.__bases__):  # iterate over base classes updating parameters from them.
            parent_param_info = getattr(bcls, 'param_info', param_info())
            my_param_info.update(parent_param_info)  # update overwrites existing info for named parameters.
            logging.info(f"Updated param_info from {bcls}")
        if hasattr(cls, 'param_info'):  # Already got param_info. Update from it
            my_param_info.update(cls.param_info)
            logging.info(f"Updated param_info from {cls.param_info}")
        my_param_info.update(cls.register_functions())

        cls.param_info = copy.deepcopy(my_param_info)
        ModelBaseClass.register_class(cls)
        # register the class for subsequent creation. This allows model_init to work.
        logging.info(f"Registered {cls.__name__}")

    @classmethod
    def register_class(cls, newcls: typing.Any):
        """
        Register class
        """
        logging.info(f"Registering class {newcls.__name__}")
        ModelBaseClass.class_registry[newcls.__name__] = newcls

    @classmethod
    def remove_class(cls, name: str | None = None):
        """
        Remove class from registry. By default self.
        :param name . If not None the name of a class to be removed
        :return: Class removed from registry
        """
        if name is None:
            name = cls.__name__

        logging.info(f"Removing {name} from registry")
        r = ModelBaseClass.class_registry.pop(name, None)
        if r is None:
            logging.warning(f"{name} not in registry")
        return r

    @classmethod
    def known_models(cls):
        """
        Return list of known models
        :return: list of know models
        """
        return list(cls.class_registry.keys())

    @classmethod
    def model_init(cls, class_name: str, *args, **kwargs) -> typing.Any:
        """
        Create a model
        :param class_name: name of class to make
        :param args: positional arguments to pass to initiation
        :param kwargs: kwargs to pass to init
        :return: new model object
        """
        try:
            newcls = ModelBaseClass.class_registry[class_name]
        except KeyError:
            raise ValueError(f"Failed to find {class_name}. Allowed classes are " + " ".join(cls.class_registry.keys()))
        result = newcls(*args, **kwargs)
        logging.debug(f"Created {class_name}")
        return result

    @classmethod
    def load_model(cls, model_path: pathlib.Path):
        """
        Load a configuration
        :param model_path:  where the configuration  is stored
        :return: loaded model
        """
        model = super().load(model_path)  # using json "magic". See generic_json for what actually happens.
        model.config_path = model_path  # replace config_path with where we actually loaded it from.
        return model

    @classmethod
    def add_param_info(cls, param_info: dict, duplicate=True):
        """
        Add information on parameters and functions.

        :param param_info: a dict with keys variable names and values either a namelist_var or callable.
          You probably should not use a callable here as better to register it when declared.
        :param duplicate If True allow duplicates which will add new namelist/callable info to existing.
        :return: Nothing
        """
        for varname, value in param_info.items():
            cls.param_info.register(varname, value, duplicate=duplicate)
            logging.debug(f"Registered {varname} with {value}")

    @classmethod
    def update_from_file(cls, filepath: pathlib.Path, duplicate=True):
        """
        Update class info on known parameters from CSV file
         Calls param_info.update_from_file(filepath) to actually do it!
         See documentation for that
        :param filepath: path to csv file
        :param duplicate -- allow duplicates.
        :return:
        """
        cls.param_info.update_from_file(filepath, duplicate=duplicate)

    @classmethod
    def remove_param(cls):
        """
        Remove param info
        :return: Nada
        """
        cls.param_info = param_info()

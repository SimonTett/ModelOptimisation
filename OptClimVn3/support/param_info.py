from __future__ import annotations

import logging
import pathlib
import typing

import numpy as np
import pandas as pd

from model_base import model_base
from namelist_var import namelist_var


class param_info(model_base):
    """
    Provides support for parameters that map to namelists and functions that return namelists (or nothing).
    This is used in Model.

    Attributes:
        param_constructors (dict): A dictionary of parameter constructors.

    Methods:
        __init__(): Initializes an instance of the class by setting `param_constructors` to an empty dictionary.
        register(parameter, var_to_set): Registers a list of namelists or a callable to a parameter.
        param(model, parameter, value): Returns parameter information for a specific value.
        to_dict(): Generates a dictionary representation using `param_constructors`.
        from_dict(dct, allowed_keys=None): Initializes an instance of the class from a dictionary representation.

    Example:
    """

    def __init__(self):
        """
        Initialise the params instance by setting param_constructors to an empty dict
        """
        self.param_constructors = dict()  # information on parameters -- currently namelist_var or functions
        self.got_vars = set()  # set of the known variables (namelist or functions)
        self.known_functions = dict()  # known functions indexed by __qualname__

    def update(self, other_param_info):
        """
        Update param info by merging in other_param_info.
        Does by iterating over other_param_info.param_constructors.items() and
           then calling register() on each element in the list. This then updates everything needed!
        :param other_param_info: another param_info
        :return: nada. Modifies param info in place
        """

        for key, lst in other_param_info.param_constructors.items():
            duplicate = False  # start of not allowing duplicates
            for v in lst:
                self.register(key, v, duplicate=duplicate)
                duplicate = True  # anything else adds to existing list

    def register(self, parameter: str, var_to_set: [typing.Callable | typing.Any], duplicate=False):
        """
        register a namelist or callable to the parameter. An error will be raised if var_to_set has already been registered unless
           parameter is being replaced.
        :param parameter: name of parameter
        :param var_to_set: namelist_vars (though could be anything that can form key to hash) or method to run
        :param duplicate: If True allow duplicate keys -- new values will be appended rather than replacing existing values
        :return: Nada
        :Examples: param_construct.register('RHCRIT',rhcrit),
                   nl= namelist_var(filepath='fred',namelist='atmos',nl_var='VF1')
                   param_construct.register(VF1=nl)
        """
        if not duplicate:
            try:
                var = self.param_constructors.pop(parameter)
                logging.info(f"Overwriting {parameter} and removing {var}")
                for v in var:
                    self.got_vars.remove(v)  # remove v from set of things we already have.
                    if callable(v):  # remove the function info.
                        self.known_functions.pop(v.__qualname__, None)
            except KeyError:
                pass

        if var_to_set in self.got_vars:
            raise ValueError(f"Already got var {var_to_set}. No duplicates allowed")
        self.got_vars.add(var_to_set)
        existing = self.param_constructors.pop(parameter, [])
        existing.append(var_to_set)
        self.param_constructors[parameter] = existing  # append to anything  that already exists.

        if callable(var_to_set):
            fname = var_to_set.__qualname__
            self.known_functions[fname] = var_to_set
            logging.debug(f"Parameter {parameter} uses method {fname} ")
        else:
            logging.debug(f"Set {parameter} to {var_to_set}")

    def read_param(self, model, parameter: str):
        """
        Read parameter value from model instance.
        :param model: model instance. Only used if method first element in parameter defn,
        :param parameter: parameter wanted
        :return: value. Depends on what is in the model...
        """
        try:
            stuff = self.param_constructors[parameter][0]  # just want first element of list.
        except KeyError:
            raise KeyError(f"Parameter {parameter} not found.\n Allowed parameters are: " +
                           " ".join(list(self.param_constructors.keys())))
        if callable(stuff):  # is it a callable
            result = stuff(model, None)  # callable. Run it in inverse mode.
            logging.debug(f"Called {stuff.__qualname__} with inverse and got {result} ")

        elif isinstance(stuff, namelist_var):
            result = stuff.read_value(dirpath=model.model_dir)
            logging.debug(f"Read data from {stuff}")
        else:
            raise NotImplementedError(f"Do not know how to deal with {stuff} of type {type(stuff)}")
        return result

    def param(self, model, parameter: str, value) -> list:
        """
        Return parameter information for a specific value as namelist/value tuple. Later functions will actually set them
        :param model: the model instance -- only used if method used
        :param parameter: parameter name
        :param value: value to be set and passed to method

        :return:
        """
        stuff = self.param_constructors[parameter]  # will fail if parameter does not exist.
        if not isinstance(stuff, list):
            raise ValueError(f"Parameter {parameter} did not return list but returned {stuff}")

        result = []
        for s in stuff:
            if callable(s):  # function.
                err_msg = f"Parameter {parameter} with {value} and method {s}  returned odd output. Should  either be: " \
                          f"None, a tuple (nl,value) or list of such tuples "
                r = s(model, value)  # run the function
                logging.debug(f"Parameter {parameter} called {s.__qualname__} with {value} and returned {r}")
                # check output.
                if r is None:  # function did something but returned nothing.
                    continue
                elif isinstance(r, tuple) and (len(r) == 2):  # returned a 2-element tuple
                    result.append(r)
                elif isinstance(r, list):  # list -- check each element.
                    for el in r:
                        if not (isinstance(el, tuple) and (len(el) == 2)):
                            raise ValueError(err_msg)
                        result.append(el)
                else:  # something else. Error!
                    raise ValueError(err_msg)

            else:  # singleton so extend result with tuple (s, value)
                result.append((s, value))
                logging.debug(f"Parameter {parameter} set {s} to {value}")

        return result

    def gen_parameters(self, model, **kwargs):
        """
        Generate parameter settings.
        :param model: model (needed for call to param)
        :param kwargs: parameter/values
        :return:list of things to be actually set. That actually should be done by the model
        """

        stuff_to_set = []
        for parameter, value in kwargs.items():
            stuff_to_set.extend(self.param(model, parameter, value))

        return stuff_to_set  # this is a list of (variable_set_info, value)

    def to_DataFrame(self):
        """
        Convert parameter info to a pandas dataframe. Any functions will be ignored as they get defined at model instance time.
        Currently, only deals with namelist_var. Extend if you need different or additional types.
        :return: df
        """
        series = []
        for param, lst in self.param_constructors.items():
            for value in lst:
                if callable(value):
                    d = dict(parameter=param, type='function')
                    d.update(function_name=value.__qualname__)
                    series.append(pd.Series(d))
                elif isinstance(value, namelist_var):
                    d = dict(parameter=param, type='namelist_var')
                    d.update(value.to_dict())
                    d['filepath'] = str(d['filepath'])
                    series.append(pd.Series(d))
                else:
                    logging.warning(f"Do not how to deal with type {type(value)}")
        df = pd.DataFrame(series)
        # want fixed ordering of columns in df.
        ordering = ['parameter', 'type']
        ordering.extend(
            namelist_var.__dataclass_fields__.keys())  # namelist_var is a dataclass so that is how we get the keys
        ordering.append('function_name')
        df = df.reindex(columns=pd.Index(ordering))
        return df

    def update_from_file(self, filepath: pathlib.Path|str, duplicate: bool = False, **kwargs):
        """
        Add parameters from file.
        :param filepath: path to csv file (can be anything accepted by pandas.read_csv).
          parameter type  filepath namelist nl_var name -- extend for different types.
          For this implementation type **must** be namelist_var. Extend for different ways of specifing parameters.
          All four parameters needed to create a namelist_var must be present
        :param duplicate: If True allow duplicates otherwise final parameter found is used.
        :param kwargs -- all remaining arguments are passed to read_csv.
        :return: Nothing.Modifies self in place
        """

        parameter_df = pd.read_csv(self.expand(filepath), **kwargs)
        parameter_df = parameter_df.replace({np.nan:None}) # replace any Nan with None. (on write out None get written as nan)
        for indx, row in parameter_df.iterrows():
            param, typ = row.loc[['parameter', 'type']]
            if typ == 'namelist_var':
                name = row.loc['name']
                if pd.isnull(name):  # no name defined set it to param.
                    name = param
                nl = namelist_var(filepath=pathlib.Path(row.loc['filepath']), namelist=row.loc['namelist'],
                                  nl_var=row.loc['nl_var'], default=row.loc['default'],name=name)
                self.register(param, nl, duplicate=duplicate)
                logging.debug(f"Registered {nl} for parameter {param}")
            elif typ == 'function':
                # check we have it and warn if not.
                fname = row.loc["function_name"]
                if fname not in self.known_functions.keys():
                    logging.warning(f"Function {fname} not found. Likely some discrepancy")
                logging.debug(f"Got function {fname} for parameter {param}")
                continue
            else:
                raise NotImplementedError(f"No implementation for type {type}")

        logging.info(f"Registered {len(parameter_df.index)} parameters")

    def print_parameters(self):
        """
        Print out the parameters we have registered
        :return: Nothing
        """

        for param, lst in self.param_constructors.items():
            print(param, end=' [')
            for value in lst:
                if callable(value):
                    print(f"function: {value.__qualname__}", end=' ')
                else:
                    print(value, end=" ")
            print("]")

    def to_dict(self):
        """
        Generate a dictionary rep using param_constructors. If callable the function name will be returned.
        Otherwise will be left alone.
        :return: dict
        """
        dct = {}
        for key, lst in self.param_constructors.items():
            dct[key] = []
            for value in lst:
                if callable(value):
                    dct[key].append(['function', value.__qualname__])
                else:
                    dct[key].append(value)
        return dct

    @classmethod
    def from_dict(cls, dct):
        """
        Tricky part is need to ignore the functions.
        :return: object
        """
        obj = cls()  # follow std initialisation.
        for key, lst in dct.items():
            values = obj.param_constructors.get(key, [])  # empty list of not already defined
            for value in lst:
                if (isinstance(value, list)) and (value[0] == 'function'):
                    # if decoding function then remember to modify self.known_functions.
                    logging.debug(f"{key} is function = {value[1]}")
                    continue
                values.append(value)
                obj.got_vars.add(value)  # list of things we have.
                logging.debug(f"{key} set to {value}")
            if len(values) > 0:  # got something?
                obj.param_constructors[key] = values
        return obj

    def known_parameters(self):
        """
        Return a list of all parameters that have been registered.
        :return:
        """
        return list(self.param_constructors.keys())

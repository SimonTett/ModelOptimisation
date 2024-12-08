# temporary model for testing.
# defines methods to work with new approach to generalised namelists.
# when confident code working this will move into Model.Model replacing some definitions there.
import copy

from Model import Model
from namelist_var import GroupConfig, type_allowed_fortran, NamelistVar
import typing
import pathlib
import logging
my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

class tempModel(Model):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.configs = GroupConfig(root_dir=self.model_dir)

    def to_dict(self) -> dict:
        """
        Convert a tempModel to a dict dropping configs from the result.
        Most of the work is done by calling the super class to_dict method.
        :return:
        """
        dct = super().to_dict()
        # and drop the configs as only relevant when reading/writing configs and is dynamically generated
        dct.pop('configs')
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> 'tempModel':
        """
        Convert a dict representing the Model to a Model object.
        :param dct: dict to be convered to a model
        :return:a tempModel instance.
        """
        dct.pop('configs',None) # if got configs drop it.
        obj = super().from_dict(dct) # use the super class method to actually do the conversion.
        return obj


    def read_nl_value(self,nl_var:NamelistVar) -> type_allowed_fortran:
        """
        Read value from namelist.
        :param nl_var: namlist variable to read
        :return: value obtained by reading the namelist.
        """
        value = self.configs.read_value(nl_var)
        return value


    def gen_params(self,
                   parameters: typing.Optional[dict] = None) -> dict[NamelistVar:type_allowed_fortran]:
        """
        Compute dict of namelists/values that will be used to set the parameters.
        :param parameters: If None use self.parameters augmented by self.parameters_no_key.
        :return: An iterable of  namelist, value pairs.
        Example: nl_values= model.gen_params()
        """
        if parameters is None:
            parameters = copy.deepcopy(self.parameters)
            parameters.update(self.parameters_no_key)  # augment/update from parameters_no_key
        else:
            self.update_history(f"Setting parameters using parameters {parameters} rather than self.parameters")
        param_set_info = []
        for parameter, value in parameters.items():
            param_set_info.extend(self.param(parameter, value))
        result = dict()
        for (nl, value) in param_set_info:
            if nl in result:
                raise ValueError(f"Duplicate namelist {nl} in param_set_info")
            result[nl] = value

        return result

    def set_params(self,
                   parameters: typing.Optional[dict] = None,
                   backup:bool = True):

        """
        Set parameters
         If self.fake is True then no parameters are set.
        :param self: Model instance
        :param parameters -- dict(or None) of parameters to use.
        :param backup -- if True backup the current parameters.
        :return: Nothing
        """
        if self.fake:
            return  # nothing to be done if faking.
        nl = self.gen_params(parameters=parameters)  # get the namelist/value stuff
        self.configs.write_values(nl,backup=backup)  # and write them all out.

    def read_param(self, parameter: str) -> type_allowed_fortran:
        """
        Read parameter value from model instance.
        :param parameter: parameter wanted
        :return: value. Depends on what is in the model...
        """
        try:
            stuff = self.param_info.param_constructors[parameter][0]  # just want the first element of the list.
        except KeyError:
            raise KeyError(f"Parameter {parameter} not found.\n Allowed parameters are: " +
                           " ".join(list(self.param_info.param_constructors.keys())))

        if callable(stuff):  # is it a callable? If so run it in inverse mode.
            result = stuff(self, None)
            my_logger.debug(f"Called {stuff.__qualname__} with inverse and got {result} ")
        else:
            #result = self.configs.read_value(stuff)
            result = self.read_nl_value(stuff)
            my_logger.debug(f"Read data from {stuff}")

        return result

    def param(self, parameter: str,
              value: type_allowed_fortran) -> list[tuple[NamelistVar, type_allowed_fortran]]:
        """
        Return parameter information for a specific value as namelist/value tuple. Later functions will actually set them
        :param parameter: parameter name
        :param value: value to be set and passed to method
        :return:
        """
        try:
            stuff = self.param_info.param_constructors[parameter]  # will fail if parameter does not exist.
        except KeyError:
            raise KeyError(f"Parameter {parameter} not found.\n Allowed parameters are: " +
                           " ".join(list(self.param_info.param_constructors.keys())))
        if not isinstance(stuff, list):
            raise ValueError(f"Parameter {parameter} did not return list but returned {stuff}")
        result = []
        for s in stuff:
            if callable(s):  # function.
                err_msg = f"Parameter {parameter} with {value} and method {s}  returned odd output. Should  either be: " \
                          f"None, a tuple (NamelistVar,value) or list of such tuples "
                r = s(self, value)  # run the function
                my_logger.debug(f"Parameter {parameter} called {s.__qualname__} with {value} and returned {r}")
                # check output.
                if r is None:  # function did something but returned nothing.
                    continue
                elif isinstance(r, tuple) and (len(r) == 2) and isinstance(r[0],
                                                                           NamelistVar):  # returned a 2-element tuple
                    result.append(r)
                elif isinstance(r, list):  # list -- check each element.
                    for el in r:
                        if not (isinstance(el, tuple) and (len(el) == 2) and isinstance(el[0], NamelistVar)):
                            raise ValueError(err_msg)
                        result.append(el)
                else:  # something else  so raise an error!
                    raise ValueError(err_msg)

            else:  # Singleton so append result with tuple (s, value).
                if not isinstance(s, NamelistVar):
                    raise ValueError(f"Parameter {parameter} returned {s} which is not a NamelistVar")
                result.append((s, value))
                my_logger.debug(f"Parameter {parameter} set {s} to {value}")

        return result

    def perturb(self, parameters: typing.Optional[dict] = None):
        """
        Set status to PERTURBED. Will need to be continued or submitted which requires submission information.
        :param parameters: dict of parameters & values to use to generate random perturbation.
        This will update parameters_no_key so key generation is unaffected and
          increase perturb_count by 1 so algorithm can adjust if multiple perturbations done,
        Note any namelist files modified will *not* be backed up.
        :return:None

        This likely needs overwriting for specific models as parameters will be set.
        For implementation in actual model class have something like:
        def perturb(self):

            parameters=dict(rand_init=random())
            super().perturb(parameters)
        """
        if parameters is None:
            parameters = {}
            my_logger.debug("Setting perturb parameters to empty dict")

        self.parameters_no_key = copy.deepcopy(parameters)  # set parameters_no_key to the perturbed parameters
        self.set_params(backup=False)  # set parameter values but with no backup done.
        self.update_history(f'Perturbed using {parameters}')  # so at least we can find out what was done
        self.perturb_count += 1
        self.set_status('PERTURBED')
        my_logger.debug(f" parameters_no_key is now {self.parameters_no_key}")

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
        cls.param_info.update_from_file_new(filepath, duplicate=duplicate)


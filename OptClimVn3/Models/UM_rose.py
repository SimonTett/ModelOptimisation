# Class to support Unified Model running in Rose.
# Is going to drive a fairly extensive refactoring of param_info.
import logging
import typing
import tempfile
import f90nml
import numpy as np
import pandas as pd

from ModelBaseClass import register_param # might be used to allow functions.
from Model import Model # note this seems to be quite important. Import Model from Model means the registration does not happen..
import pathlib
from namelist_var import namelist_var
import copy
import metomi.rose.config # tools to read and write rose config files.

my_logger = logging.getLogger(f"OPTCLIM.{__name__}") # have this anywhere you want logging

class UM_rose(Model):
    """
    Class to support the Unified model running in ROSE.
    Complication is that this class will need to run a rose job on another super-computer
    Also need to deal with rose handing of namelists.

    """

    def __init__(self, *args, **kwargs):
        """
        Init the UM_rose instance. Calls the super-class. May need to set some variables.
        :param args: positional args. Passed through to super-class
        :param kwargs: kwargs -- passed through to super-class
        """
        super().__init__(*args, **kwargs)  #
        # switch off fixed scripts. Will change submit_cmd instead.
        self.submit_script = None
        self.continue_script = None # probably don't need this for now. There if have an error.
        # I think ROSE handles that kind of stuff so just need to resubmit the config on puma.
        # Alternatively (if easier) modify the super class submit method
        self.clean_cache()  # initialise the cache.



    # to_dict & clean_cache can probably go to Model.
    # but that would require a fairly large (and so risky) refactorisation
    # of Model and param_info.
    def to_dict(self):

        """
        Return a dictionary representation, suitable to conversion to JSON.
          Uses the super class to_dict method and then removes file caches.
          Don't want to serialise cached values"""
        d = super().to_dict()
        for k in ['_config_cache']: # list of cache keys
            d.pop(k)
        return d
    def clean_cache(self):
        # TODO Move into Model class so all sub-classes get this
        """
        Clean out the caches
        :return: Nothing -- just sets _config_cache to empty dictionary.
        """
        self._config_cache = dict()

    type_allowed_fortran = typing.Union[int, float, bool, str]  # types allowed in fortran
    import metomi.rose.config
    def file_cache(self, filepath: pathlib.Path,
                   clean: bool = False, make_copy: bool = True) -> metomi.rose.config.ConfigNode:
        """

        Read/cache file
        :param filepath: path to file containing namelist
        :param clean: If True,clean the cache
        :param make_copy: If True, make a deep copy of the cached data
        :return: config
        """

        if clean:
            self.clean_cache()
        if filepath in self._config_cache.keys():
            config = self._config_cache[filepath]
        else:
            config = metomi.rose.config.load(str(self.model_dir/filepath))
            self._config_cache[filepath] = config
        if make_copy:
            config = copy.deepcopy(config)
        return config



    def write_nml_values(self, nl_info: iter) -> bool:
        # TODO provide a write to json for the base class. Then HadCM3 can overwrite!
        # and see if can also fix gamil3
        """
        Update  rose config namelist info from changed namelist_var values.
         Modifies config,  write the modified config to a temp file, removes the input file
            and moves the temp file to the original location.
        :param nl_info: iterable of namelist_var, value pairs,
          will also clear cache after all modification done.
        :return: True if successful

        Example usage write_nml_values({VF1_nl:0.5,ENTCOEF_nl:[0.8,0.8,0.85,...0.9,0.95])
        """

        namelists = namelist_var.group_namelists(nl_info)
        # group name_namelists is not doing what it should!
        # fixit!
        for fp, nl_patch in namelists.items():
            # Iterate over the files and patches to namelist info.
            # work out full filepath
            filepath=self.model_dir/fp
            config = self.file_cache(filepath) # get the config for this file from the cache
            bak_file = filepath.with_name(filepath.name + ".bak")  # name of backup file
            my_logger.debug(f" {filepath} renamed to {bak_file}")
            # write to a temp file. If something goes wrong, input should still be OK,
            with tempfile.NamedTemporaryFile(dir=self.model_dir, delete=False, mode='w') as tmp:
                # iterate over the namelists and update with the new values.
                for nl_name, nl in nl_patch.items():
                    # Loop over 'namelists' in file. Namelists have values
                    # Worth adding a to_string method to var_namelist
                    # that returns a dict with keys the var names and values converted to strings.
                    # in fact have a look in f90nml as that might have one!
                    for var_name,value in nl.items():
                    # Convert value to a string
                        if isinstance(value, (list, tuple)):
                            v = ','.join([namelist_var.to_fortran(x) for x in value])  # only want one level of recursion
                            # if want more than make to_fortran recursive.
                        else:
                            v = namelist_var.to_fortran(value)  # this wil raise an error if value is not a supported type.
                        config.set(['namelist:' + nl_name, var_name], value=v)
                    my_logger.debug(f"Set {nl_name} {var_name} to {v}")
                metomi.rose.config.dump(config,tmp) # dump the modified config
                my_logger.debug(f"Dumped to {tmp.name}")
            # mv input config file to backup file
            filepath.rename(bak_file)
            pathlib.Path(tmp.name).rename(filepath)  # move temp file to original location.
            my_logger.info(f"Modified {filepath}") # end of iteration over files.
        self.clean_cache()  # cache now "dirty" (been modified) and so needs to  be cleaned.
        return True  # modification succeeded

    def set_params(self, parameters: typing.Optional[dict] = None):
        #TODO move to Model class so available to everyone. That would be part of a gulp refactor
        # as then need implementations of read_nml_var and write_nml_var in Model and inherited models.
        """
        Set parameters.

         If self.fake is True then no parameters are set.
        :param self: Model instance
        :param parameters -- dict(or None) of parameters to use.
        :return: Nothing
        """
        if self.fake:
            return # nothing to be done if faking.
        nl = self.gen_params(parameters=parameters)
        self.write_nml_values(nl)  # and modify the model

    def read_nml_var(self, namelist: namelist_var,
                     ) -> typing.Union[list[type_allowed_fortran], type_allowed_fortran]:
        """
        Extract namelist variable value from the cached config file.
        This is specific to the rose config used by the UM in rose
        :param namelist -- namelist_var for which we want the value.
        :return: value
        """
        config = self.file_cache(self.model_dir / namelist.filepath)
        value = config.get(['namelist:' + namelist.namelist, namelist.nl_var]).value
        if value is None:  # not found
            my_logger.debug(f"Failed to read {namelist} from {namelist.filepath} returning default")
            return namelist.default
        # need to parse this string to convert to value. This is a bit of a pain.
        # use f90nml to do that -- which does not seem to have parse from  fortan nml to value public fn.
        # Not worth caching as we probably will only
        # read 10-20 variables.  If we read 100s+ then worth rewritting to be more efficient
        # and read a whole namelist (rather than just individual vars in the namelist)
        str = f"&{namelist.namelist} {namelist.nl_var} = {value} /"
        result = f90nml.reads(str)[namelist.namelist][namelist.nl_var]
        return result

    # The two methods below should be moved back to Model as they
    # do not assume anything about reading and writing configs.
    def read_param(self, parameter:str) -> type_allowed_fortran:
        #TODO: Move this to Model
        #That will require Model having a default read_nml_value/write_nml_value.
        #And then overwriting that in other methods
        # That's a gulp refactor.
        """
        Read parameter values from model instance.
        :param parameter: parameter wanted
        :return: value. Depends on what is in the model...
        """
        param_info = self.param_info
        try:
            stuff = param_info.param_constructors[parameter][0]  # just want first element of list.
            #TODO -- understand why getting a list and remove it!
        except KeyError:
            raise KeyError(f"Parameter {parameter} not found.\n Allowed parameters are: " +
                           " ".join(list(param_info.param_constructors.keys())))
        if callable(stuff):  # is it a callable
            result= stuff(self, None)  # callable. Run it in inverse mode which should return the value.
            # methods might want to use read_param to do that!
            my_logger.debug(f"Called {stuff.__qualname__} with inverse and got {result} ")

        elif isinstance(stuff, namelist_var): # namelist var.
            result = self.read_nml_var(stuff) # actually read it.
            # The per model implimentation of read_nml_var should handle the complexity of actually doing that!
            my_logger.debug(f"Read data from {stuff}")
        else:
            raise NotImplementedError(f"Do not know how to deal with {stuff} of type {type(stuff)}")

        return result

    def read_params(self, parameters: typing.Optional[typing.Union[str, list[str]]], fail: bool = True) -> dict:
        #TODO move to Model class so available to everyone.
        """
        Read multiple parameters values from config files.
        :param parameters: parameter of list of parameters to read
        :param fail: If True raise an error if a parameter not found
        :return: dict indexed by parameter names.
        """

        result = dict()
        if isinstance(parameters, str):
            parameters = [parameters]  # make it a list.
        if parameters is None:
            parameters = self.param_info.known_parameters()  # get all parameters

        for parameter in set(parameters):  # set means we iterate over unique parameters
            try:
                result[parameter] = self.read_param(parameter)
            except (KeyError, FileNotFoundError):
                if fail:
                    raise
                my_logger.warning(f"Parameter {parameter} not found in {self.name}")
                result[parameter] = None

        return result



    def create_model(self):
        """
        Create the model. This is a rose specific version of create model.

        """
        super().create_model()

    def modify_model(self):
        """
        UM rose specific version of modify model,
        Needs to:
        1) Modify directories so that running in a  sensible place wih output going there.
           use self.model_dir to get sensible place
        1a) Modify the rose config so that it no longer copies to archer2 the rose info.
         (as we already have it)
        2) Put in the cylc config just before model starts running (which could be multiple times):
            self.set_status_script self.config_path RUNNING
        3) Add to the cylc config after the model has finished
           self.set_status_script self.config_path SUCCEEDED
        4) Optionally add to the cylc config where errors are detexted
               self.set_status_script self.config_path FAILED

        """
        super().modify_model()  # call the base class method.
        # now do the specific stuff...

    def submit_cmd(self) -> typing.List[str]:
        """"
        Generate the submission command. Over rides the super-class version.
        """
        # check status sensible for submitting run,
        if self.status not in ['INSTANTIATED', 'PERTURBED','CONTINUE']:
            raise ValueError(f"Status {self.status} not expected ")
        remote_machine = self.run_info.get('remote_machine','')
        # remote machine where rose runs.
        cmd = self.expand('$OPTCLIMTOP/OptClimVn3/scripts/UM_rose/SUBMIT_to_puma.sh')
        cmd = [cmd,self.model_dir,remote_machine]
        return cmd



pth = pathlib.Path(__file__).parent /'parameter_config/UM_rose_Parameters.csv'
UM_rose.update_from_file(pth, duplicate=True)

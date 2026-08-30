"""
Class for abstract Model
All different types of models should inherit from this. It provides the core methods
and information needed,

To add a new model you likely need to do, at least, the following:

1) Implement your own version of modify_model.
   It should first call the super class modify_model method to do the basic stuff.
   Your code should turn your reference configuration into a stand along configuration running
     in self.model_dir.
   You should also modify the configuration so that:
   a) just before model starts running insert:
       self.set_status_script self.config_path RUNNING
    This could be run every time a model starts running which could be multiple times. This case has not be tested.
   b) Insert  in the configuration after the model has finished all its simulations:
        self.set_status_script self.config_path SUCCEEDED
    c) Optionally insert in the config where errors are detected:
        self.set_status_script self.config_path FAILED

2) Implement your own version of submit_cmd so it returns the command to submit the model to your Q system,

3) Define your parameters.
 Hopefully they can be described by a 'namelist  -- file, group name & variable name.
 See support/namelist_var.py.
 See namelist_var.py for guidance on existing types of 'namelists' and guidance for adding new ones.

 Easy is parameters whose values directly set one or more 'namelist variables' to that value.
 For this the recommended approach is to create a csv file with the parameter names and namelist values.
 For examples See the csv files  in parameter_config.
 After the model class has been defined then do :
     ModelClass.update_from_file(path_to_config_file)
 where path_to_config_file is a pathlib.Path of the parameter configuration file.

 More tricky you might have a 'hyper' parameter which modifies multiple namelist variable,
 sets a namelist variable to an error, or modifies the model in some other way.
 To do this you need to write a function and register the function using ModelBaseClass.register_param
 Your methods should take a single value or None. If None the method should return a scaler which is the current value of
 the parameter (as set in the configuration files) this is 'inverse'. If a value is passed  the function should
 either return a list of namelist,value tuples OR silently set the values itself and return None. See HadCM3 for an extensive set of methods for possible examples.

4) Write a study configuration file. This is a json file that describes the study. You will need to define
  many elements in the run_info dict. See existing cases in configurations to see what is needed.
  Note you can include various things in this configuration file.

Do write tests for your new Model testing your new and modified methods.

"""



import copy
import functools
import logging
import os
import pathlib
import shutil
import tarfile
import typing
import tempfile

import numpy as np
import pandas as pd
import xarray

import json

import genericLib
import namelist_var
from model_base import journal
from ModelBaseClass import ModelBaseClass, register_param
from namelist_var import NamelistVar, GroupConfig, type_allowed_fortran
from engine import abstractEngine
import shlex


my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

type_status = typing.Literal['CREATED', 'INSTANTIATED', 'SUBMITTED',
'RUNNING', 'FAILED', 'PERTURBED', 'CONTINUE',
'SUCCEEDED', 'PROCESSED']  # allowed strings for status


class Model(ModelBaseClass, journal):
    # type definitions for attributes.
    name: str
    config_path: typing.Union[pathlib.Path, pathlib.PurePath]
    reference: typing.Union[pathlib.Path, pathlib.PurePath]
    reference_name: typing.Optional[str]
    model_dir: typing.Union[pathlib.Path, pathlib.PurePath]
    post_process: dict
    post_process_cmd_script: typing.Optional[list[str]]
    fake: bool
    perturb_count: int
    submission_count: int
    parameters: dict
    parameters_no_key: dict
    run_info: dict
    engine: abstractEngine
    pp_jid: typing.Optional[str]
    model_jids: list[str]
    submitted_jid: typing.Optional[str]
    submit_script: typing.Union[pathlib.Path, pathlib.PurePath]
    continue_script: typing.Union[pathlib.Path, pathlib.PurePath]
    set_status_script: typing.Union[pathlib.Path, pathlib.PurePath]
    status: type_status
    simulated_obs: typing.Optional[pd.Series]
    remote_directory: typing.Optional[pathlib.PurePath]
    _post_process_input: typing.Optional[str]
    _post_process_output: typing.Optional[str]
    configs: GroupConfig
    remote: dict[str, str | pathlib.PurePath]
    study_info: dict[str, str]
    StudyConfig_path:typing.Optional[pathlib.Path] # Try and remove this. Used in simple model.

    """
    Abstract model class. Any class that inherits from this will have name lookup.
    Also provides top-level methods and class methods.
    Model.load_model() will load a model from disk. An object of the appropriate class will be returned as long as that
    class inherits from Model.
    As it inherits from journal  it has methods to update history,output and run commands.
    public attributes: (Be careful if you  change them)
        model_dir -- directory where model information is stored
        reference --  where the reference configuration came from.
        reference_name -- name of the reference model config. If None then reference.name is used.
        config_path -- where the configuration is to be written to (or was read from).
        name -- name of the model
        status -- status of the model
        post_process -- post-processing information. See Model.set_post_process for details.
        fake -- If True model is faked.
        perturb_count -- no of times perturbation has been done.
        parameters -- dict of parameters/values. Used to generate key and set values.
        parameters_no_key -- dict of parameters/values. Overrides parameters to set values but  do not form part of the key
        set_status_script -- path to script that sets_status. Your model will need to call this.
        engine -- submission engine.
        pp_jid -- post-processing job id. This gets released when model status changes to SUCCEEDS
        model_jids -- list of model job ids.
        configs - cache of configuration files (namelists or equivalent)
        pertub_count -- no of times model has been perturbed.
        submission_count -- no of times model has been submitted.
        remote -- dict containing info on remote machine and directory if needed. Keys are machine and directory respectively.
        study_info -- dict containing information from the study (and up). Model should never use this. 
        StudyConfig_path -- path to StudyConfig.
        
        Private attributes:
          _post_process_input -- name of input file for post-procesing
          _post_process_output -- name of output file for post-processing
        Note that update_history and store_output (see Journal for doc for those) set up private attributes.
    """
    post_proccess_json = "post_process.json"  # where post-process info gets written
    status_info = dict(CREATED=None,
                       INSTANTIATED=["CREATED"],  # Instantiate a model requires it to have been created
                       SUBMITTED=['INSTANTIATED', 'PERTURBED', 'CONTINUE'],
                       # Submitting needs it to have been instantiated, perturbed or to be continued.
                       RUNNING=["SUBMITTED", "RUNNING"],
                       # running the model should have been submitted or already been running
                       FAILED=["RUNNING", "SUBMITTED"],  # Failed means it should have been running or SUBMITTED
                       PERTURBED=["FAILED"],  # Allowed to perturb a model after it failed.
                       CONTINUE=["FAILED", "PERTURBED"],
                       # failed can just be continued or can be perturbed. For example ran out of time or disk space
                       # full
                       SUCCEEDED=["RUNNING"],  # SUCCEEDED means it should have been running
                       PROCESSED=['SUCCEEDED'])  # Processed means it should have succeeded.
    # Q Perturbed comes in two flavours. Perturb and continue or perturb and restart. How to handle that?
    allowed_status = set(status_info.keys())
    _known_parameters_cache: typing.Optional[set[str]] = None  # cache for known parameters.

    @classmethod
    def load_model(cls, model_path: pathlib.Path):
        """
        Load a configuration and update paths if needed.
        :param model_path:  where the configuration is stored


        :return: loaded model
        """
        model = super().load(model_path)  # Using json "magic". See generic_json for what actually happens.

        if not (isinstance(model.config_path,pathlib.Path) and model.config_path.samefile(model_path)):
            my_logger.warning(f"Model {model} model_path changed to {model_path}")
            model.config_path = model_path  # Replace config_path with where we actually loaded it from.

        if not (isinstance(model.model_dir,pathlib.Path) and model.model_dir.samefile(model_path.parent)):
            my_logger.warning(f"Model {model} model_dir changed to {model_path.parent} ")
            model.model_dir = model_path.parent  # update directory with where we actually loaded it from.

        return model

    # methods now.
    def __init__(self,
                 name: str,
                 reference: pathlib.Path,
                 reference_name: typing.Optional[str] = None,
                 post_process: typing.Optional[dict] = None,
                 model_dir: pathlib.Path = pathlib.Path.cwd(),
                 config_path: typing.Optional[pathlib.Path] = None,
                 status: type_status = "CREATED",
                 parameters: typing.Optional[dict] = None,
                 engine: typing.Optional[abstractEngine] = None,
                 run_info: typing.Optional[dict] = None,
                 fake: bool = False,
                 config_dir: typing.Optional[pathlib.Path] = None,
                 study_config_path:typing.Optional[pathlib.Path] = None,):
        """
        Initialize the Model class.

        :param name -- name of model
        :param reference -- reference directory. Should be a pathlib.Path
                keyword arguments
        :param reference_name -- name of reference model. If None then reference.name is used.
        :param model_dir --- where model will be created and any files written.
             Should be a pathlib.Path. Will, if needed, be created. If none cwd will be used.
             Must be different from reference
        :param config_dir -- where configuration files are written. This should be relative to model_dir.
          If None then config files stored in model_dir
        :param config_path: Where configuration will be created.
               If not defined (or None) will be model_dir/(self.name+".mcfg")
        :param status -- model status. Default = "CREATED"
        :param parameters -- dict of parameters names and values.
        :param post_process -- dict for post-processing. If None no post-processing will be done.
         Dict will be made available to post-processing code.
          See set_post_process method for details.
           post_process must contain script
            The following keys may  be used:
                interp -- the interpreter used for the script.
                    Handy for windows but better to use #! in your script!
                input_file -- name of the input file for the post-processing.
                     This will contain the contents of post_process minus the script stuff.
                output_file -- name of the output file from the post-processing
                runTime -- time in seconds for the post-processing.
          This should be constructed in the  setup using the StudyConfig.
        :param run_info -- A dict containing information for submission of the model and post-processing.
           other keys are:
            runTime -- the time (seconds) for job
            runCode  -- the code to use to run the job.
            runUser -- the UserId to run the job with.

        :param fake -- if True then model is faked. No submission will be done.
        :param study_config_path -- path to StudyConfig. This is there in case model wants to interrogate it at init time.

        """
        # set up default values.

        # General attributes

        self.name = name

        if config_path is None:
            config_path = model_dir / (self.name + '.mcfg')

        self.config_path = config_path
        # TODO -- when bringing in from another system. reference may not exist
        # It probably should be a PurePath so there is a problem with convert_
        # Check model_dir is not a reference.
        if (
                isinstance(reference, pathlib.Path) and
                ((model_dir == reference) or
                 (model_dir.exists() and reference.samefile(model_dir)))
        ):
            raise ValueError(f"Model_dir {model_dir} is the same as reference {reference}")

        self.reference = reference
        if reference_name is None:
            reference_name = reference.name
        self.reference_name = reference_name
        self.model_dir = model_dir
        if config_dir is None:
            self.config_dir = model_dir
        elif model_dir is not None:
            self.config_dir = model_dir / config_dir  # config_dir **relative** to model_dir
        else:
            raise ValueError("config_dir must be specified if model_dir is None")
        if status not in self.allowed_status:
            raise ValueError(f"Status {status} not in " + " ".join(self.allowed_status))

        self.StudyConfig_path = None  # setup StudyConfig_path attribute.
        if study_config_path is not None:
            self.StudyConfig_path = study_config_path # store the path to the config.
            if not self.StudyConfig_path.is_absolute(): # not absolute so make it so
                self.StudyConfig_path = pathlib.Path.cwd()/self.StudyConfig_path

        # attributes to do with post-processing
        self.post_process = {}  # where all post-processing info stored.
        self.post_process_cmd_script = None  # cmd (list) to be run for post-processing.
        self._post_process_input = None  # filename where input for post-processing goes
        self._post_process_output = None  # filename where output for post-processing goes.
        # post processing info
        if post_process is not None:
            self.set_post_process(post_process)

        # attributes to do with model meta-information.
        self.fake = fake  # did we fake the model? Changed later.
        self.perturb_count = 0  # how many times have we perturbed the model?
        self.submission_count = 0  # how mamy times have we submitted the model?

        # history and output
        self.update_history(None)  # init history.
        self.store_output(None, None)  # init store output

        # parameters
        if parameters is None:
            parameters = {}
        else:
            parameters = copy.deepcopy(parameters)

        self.parameters = parameters
        self.parameters_no_key = {}  # parameters that do not generate key and augment/modify parameters.

        # "system" stuff. Things to do with actually submitting a model and the post-processing. Those are all in runInfo
        self.run_info = {}  # make runInfo an empty dict.
        if run_info is not None:
            self.run_info = copy.deepcopy(run_info)  # copy the run_info into Model.

        if engine is None:  # no submission engine provided. So create one. Default will be SLURM.
            engine = abstractEngine.create_engine(engine_name=self.run_info.get('submit_engine', 'SLURM'),
                                                  ssh_node=self.run_info.get('ssh_node'))  # default engine.
            my_logger.info(f'Set abstract Engine to {engine}')
        self.engine = engine  # submission engine.
        self.model_jids = []  # list of all model job ids running came across.
        self.pp_jid = None  # post-processing job id
        self.submitted_jid = None  # job id of last submitted model submitted.
        # setup submit and continue script. Create them as pure paths (here) as we do not expect
        # to actually create this class of Model for real use.
        self.submit_script = pathlib.PurePath("submit.sh")
        self.continue_script = pathlib.PurePath("continue.sh")
        # setup path to where script that sets status is.
        root = self.expand("$OPTCLIMTOP/OptClimVn3")
        script_pth = root / "scripts/set_model_status.py"
        self.set_status_script = script_pth

        # and simulated obs.
        self.simulated_obs = None
        self.configs = GroupConfig(root_dir=self.config_dir)  # grouped configs for writing out generic namelists
        # Set up remote stuff.
        remote_machine = self.run_info.get('remote_machine', None)
        remote_model_dir = self.expand(self.run_info.get('remote_model_dir', None), local=False)
        local_root_dir = self.expand(self.run_info.get('local_root_dir', None))
        if remote_model_dir is not None:
            remote_model_dir = self.new_path(self.model_dir, remote_model_dir, root_dir=local_root_dir)

        self.remote = dict(remote_machine=remote_machine, remote_model_dir=remote_model_dir)  # remote info
        self.study_properties=dict()
        # Set status
        self.status = status
        if self.status == 'CREATED':  # creating model for the first time
            self.update_history("CREATING model")

    @classmethod
    def get_param_info(cls, parameter: str) -> list[NamelistVar | typing.Callable]:
        """
        Get parameter info for a parameter. Searches mro wise through object inheritence.
        :param parameter: name of the parameter
        :return: list of NamelistVars or functions that can be used to set the parameter.
        """
        # OPTIMISATION: cache the result so we do not have to search through __mro__ every time.
        # Set this up when the class is created -- see ModelBaseClass.
        # search through __mro__ for the parameter.
        stuff = None
        for bcls in cls.__mro__:
            if hasattr(bcls, 'param_info') and parameter in bcls.param_info.param_constructors:
                stuff = bcls.param_info.param_constructors[parameter]
                my_logger.debug(f'Found parameter {parameter} in {bcls.__name__}')
                break  # exit the loop as we have found the parameter.
        if stuff is None:
            params_known = sorted(cls.known_parameters())
            raise KeyError(f"Parameter {parameter} not found.\n Allowed parameters are: " + "\n" +
                           " ".join(params_known))
        elif not isinstance(stuff, list):
            raise ValueError(f"Parameter {parameter} did not return list but returned {stuff}")
        return stuff

    @classmethod
    def known_parameters(cls) -> set[str]:
        """
        Get a list of known parameters for this model. Caches the result.
        :return: list of parameter names.
        """

        # search through __mro__ for the parameter.
        known_params = []
        for bcls in cls.__mro__:
            if hasattr(bcls, 'param_info'):
                known_params += list(bcls.param_info.known_parameters())

        return set(known_params)

    def set_post_process(self, post_process: typing.Optional[dict] = None):

        """
        Set up post_process info.
        :param post_process -- None or  dict containing information for post-processing.
            If None immediately returns without doing anything.
        If dict must include:
            script -- full path to the script to run. Will be passed through self.expand to expand vars and user id.
             Will be checked for existence, and if script_interp is not None,  for execute permission.
            If None/not present raises an error.
        post_process can include
            interp -- if not None the name of the interpreter.
            input_file -- name of input file for post-processing. If None will be input.json
                 post-processing info will be written to that file so post processor has access to it
                Will be stored in self._post_process_input
            output_file -- name of output file for post-processing. If None will be sim_obs.json
                This is where simulated observations go (which are then read in).
                Will be stored in self._post_process_output
            Can also include per reference.name post_processing info which overwrites anything in the main dict.
        post_process will be deep-copied to self.post_process with script, interp, input_file, output_file removed
        self.post_process_cmd_script will hold the command to run the post-processing.
        :return: None
        """
        if post_process is None:
            return

        pp = copy.deepcopy(post_process)

        # now deal with reference_name if needed.
        pp_for_ref = pp.pop('post_process_for_reference', None)  # individual post processing wanted?
        if pp_for_ref is not None:  # have per reference post-processing info
            if self.reference_name is None:
                raise ValueError('post_process_for_reference present but reference_name is None')
            my_logger.debug(f"Using post_process_per_reference for {self.reference_name}")
            dct: typing.Optional[dict] = pp_for_ref[
                self.reference_name]  #  Will trigger error if self.reference_name not in dict.
            if dct is not None:  # actually got something for this reference name.
                pp.update(pp_for_ref[self.reference_name])  # update with per reference info.

        # then set up script    etc.
        script = pp.pop('script', None)
        if script is None:
            raise ValueError("No script in post_process")

        interp = pp.pop('interp', None)
        input_file = pp.pop('input_file', 'input.json')
        output_file = pp.pop('output_file', 'sim_obs.json')

        self._post_process_input = input_file
        self._post_process_output = output_file
        script = self.expand(script)
        # check script is read/executable by us raising errors if not.
        ok = script.is_file()
        if not ok:
            raise ValueError(f"Script file {script} does not exist")
        if not os.access(script, os.R_OK):
            raise ValueError(f"Script file {script} does not have read permission")

        if (not interp) and (not os.access(script, os.X_OK)):  # if script only then test for it being executable.
            raise ValueError(f"Script file {script} does not have execute permission")

        pp_cmd = []
        if interp:
            pp_cmd += [interp]
        pp_cmd += [script]
        pp_cmd += [input_file, output_file]
        self.post_process_cmd_script = pp_cmd  # assumed to be running in model_dir
        self.post_process = pp  #

    def __eq__(self, other, vars_to_ignore: typing.Optional[typing.List[str]] = None):
        """
        Compare two objects and identify their differences by comparing their attributes.
        :param other: The other object to compare against.
        :param vars_to_ignore: List of variables to ignore in the comparison. Will always have configs appended
        :return: Set of differing attributes between the objects.
        """
        if vars_to_ignore is None:
            vars_to_ignore = []
        vars_to_ignore.append('configs')
        result = super().__eq__(other, vars_to_ignore=vars_to_ignore)
        return result

    def compare_objects(self, other):
        """
        Compare two Model objects and identify their differences by comparing their attributes.
        Based on __eq__ method. With rewriting by chatGPT
        :param other: The other object to compare against.
        :return: Set of differing attributes between the objects.
        """
        if self == other:
            return []  # Objects are identical

        if type(self) is not type(other):
            return set(['Different types:', type(self), type(other)])

        vself = vars(self)
        vother = vars(other)

        diff_attrs = set()
        for k in vself.keys():
            if (vself[k] is None) and (vother[k] is None):
                pass
            elif (vself[k] is None) or (vother[k] is None):
                diff_attrs.add(k)
            elif isinstance(vself[k], type(vother[k])):
                diff_attrs.add(k)
            elif isinstance(vself[k], pd.Series):
                if not np.allclose(vself[k], vother[k]):  # check for fp diffs in the series.
                    diff_attrs.add(k)  # pandas series differ
            elif vself[k] != vother[k]:
                diff_attrs.add(k)

        # Print the values of differing attributes
        for attr in diff_attrs:
            print(f"{attr}: self={vself[attr]}, other={vother[attr]}")

        return diff_attrs

    def __repr__(self):
        """
        String that represents model. Shows type,name,status, no of parameters and
          (if possible) when history last updated
        :return: str
        """
        last_hist_key = self.last_history_key()
        if last_hist_key is None:
            last_hist_key = "Never"
        s = f"Type: {self.class_name()} Config Name: {self.config_name()} Name: {self.name}" \
            f" Status: {self.status} " \
            f"Nparams: {len(self.parameters)} Last Modified:{last_hist_key}"
        return s

    def dump_model(self):
        """
        dump a model configuration to self.model_dir/model_config_name
        :return: whatever dump does
        """

        return self.dump(self.config_path)  # call the  *dump* method.

    # Code to deal with parameters.
    def read_params(self, parameters: str | typing.List[str] | None, fail: bool = True) -> dict:
        """
        Read parameter values from self.model_dir
        :param parameters: list of parameters OR parameter to read. If None all known parameters will be read.
        :param fail. If true fail if namelist file is not found.
        :return: dict of parameter/value tuples
        """
        result = dict()
        if isinstance(parameters, str):
            parameters = [parameters]  # make it a list.
        if parameters is None:
            # TODO make this work in __mro__ order so that we can read parameters from all classes.
            parameters = self.known_parameters()  # get all parameters

        for parameter in set(parameters):  # set means we iterate over unique parameters
            try:
                param = self.read_param(parameter)
                if isinstance(param, dict):  # got multiple values back
                    result.update(param)  # update the result
                elif parameter in result:
                    my_logger.warning(f"Already got {parameter}. Skipping")
                    pass
                else:
                    result[parameter] = param
            except (KeyError, FileNotFoundError):
                if fail:
                    raise
                my_logger.warning(f"Parameter {parameter} not found in {self.name}")
                result[parameter] = None

        return result

    def gen_parameters(self, **kwargs) -> list[tuple[NamelistVar, type_allowed_fortran]]:
        """
        Generate parameter settings.
        :param kwargs: parameter/values
        :return:list of things to be actually set. That actually should be done by the model
        """

        stuff_to_set = []
        for parameter, value in kwargs.items():
            stuff_to_set.extend(self.param(parameter, value))

        return stuff_to_set  # this is a list of (variable_set_info, value)

    ## end of parameter related methods
    def create_model(self,
                     direct: typing.Optional[pathlib.Path] = None,
                     copy_ref: bool = True):
        """
        Create a new model by copying reference. If self.fake is True then no copy is done.
         Overwrite (and call superclass) for your own model.
         :param direct: path to directory to create. If None then self.model_dir will be used.
         :param copy_ref: If True copy reference directory to dir
        For example if you want to modify your reference model.
        :return:nothing.
        """
        if direct is None:
            direct = self.config_dir
        if (not self.fake) and copy_ref:
            direct.mkdir(parents=True, exist_ok=True)  # create the directory if needed.
            my_logger.debug(f"Created {direct}")
            # empty the directory (if we are creating)
            for file in direct.iterdir():
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()
            # now copy the reference config in
            shutil.copytree(str(self.reference), str(direct), symlinks=True,
                            dirs_exist_ok=True)  # copy from reference.
            my_logger.debug(f"Copied {self.reference} to {direct}")

    def check_status(self,new_status: type_status,
                     check_existing: bool = True,
                     check_allowed: bool = True
                     ) -> bool:
        """
        check new_status is OK
        :param new_status: new status for model
        :param check_existing: Check current status is as expected.
        :param check_allowed: Check that new status is allowed.

        If both set to False then no checking will actually be done

        Will raise an error if there is a problem.

        """

        if check_allowed:
            if new_status not in self.allowed_status:
                raise ValueError(f"Status {new_status} should be one of " + " ".join(self.allowed_status))
            if new_status == 'CREATED':
                raise ValueError(f"Do not set status to CREATED")

        if check_existing:
            expected_status = self.status_info[new_status]
            if self.status not in expected_status:
                raise ValueError(
                    f"Expected current status {self.status}  to be one of " + " ".join(
                        expected_status) + f" as changing to {new_status}")
        return True

    def set_status(self, new_status: type_status,
                   check_existing: bool = True,
                   check_allowed: bool = True) -> None:
        """
        Set the status of Model.
        Checks that new status is allowed and consistent with current status
        Then sets the status. Writes out the model configuration.
        See self.status_info for allowed status names and what is expected.
        :param new_status: new status for model
        :param check_existing: Check current status is as expected.
        :param check_allowed: Check that new status is allowed.

        """
        ok = self.check_status(new_status, check_existing, check_allowed)

        my_logger.debug(f"Changing status from {self.status} to {new_status}")
        self.update_history(f"Status set to {new_status} in {self.model_dir}")
        self.status = new_status
        self.dump_model()  # write to disk

    def instantiate(self, fake: bool = False) -> None:
        """
        Run create_model and set_params, update status.
        If fake is True then do not actually create the model.
         Just create the directory. Actual behaviour depends on class
         implementations of create_model, modify_model and set_params.
        Will verify that (if defined) post-processing script exists failing if not
        Also, should make any changes needed to those files.
        Also runs install_remote using run_info['remote_machine'] and run_info['remote_dir'] if they are not None.
        :return:
        """
        # check we are OK to instantiate
        status:type_status = 'INSTANTIATED'
        self.check_status(status)
        self.fake = self.fake or fake
        # if fake is True then we are faking it unless we are already faking it!
        self.create_model()  # create model
        self.modify_model()  # do any modifications to model needed before setting params.
        self.set_params()  # set the params
        self.check()  # check the model is ok. Very model dependent.
        # set permissions to rxw,rx,rx for submit and continue script.
        if not fake:
            for file in [self.submit_script, self.continue_script]:
                if file is not None:
                    (self.model_dir / file).chmod(0o755)  # set permission
            # install remote if needed.

            cmds = self.install_remote_command(remote_machine=self.remote.get('remote_machine'),
                                               remote_model_dir=self.remote.get('remote_model_dir'))
            # expect list of cmds to run. make dir and then do rsync
            if cmds is not None:
                for cmd in cmds:
                    output = self.run_cmd(cmd, convert_to_posix=True, quote=False)
                my_logger.debug(f"Installed model remotely with {cmd} and got {output}")


        else:
            self.fake = True  # we are faking now!
        # possibly do remote_install

        self.set_status(status)

    def modify_model(self):
        """
        Modify model.This method is minimal; call from your own class
        Those should call this first as it  updates history

        :return: None
        """

        self.update_history(f"modifying model")

        return None

    def check(self) -> bool:
        """
        Check the model is ok. You should call this and then do your own stuff in your class method
        This method checks:
           set_status_script, submit_script and continue_script is a file

        Will raise ValueError if the model is not ok.
        :return: True if model is ok.
        """
        self.update_history("Checking model")
        if not self.fake:
            if not self.set_status_script.is_file():
                raise ValueError(f"Need {self.set_status_script} is not a file")

        return True

    def submit_post_process(self) -> str:
        """
        Submit post-processing job. Will be submitted held. Later processing will release the job. 
        """

        pp_cmd = [str(self.set_status_script), str(self.config_path), 'PROCESSED']
        # post-process cmd. Which gets submitted now and the job id recorded.
        job_params = self.engine.extract_job_submission_params(self.run_info, default_values=dict(
            runTime=1800))  # extract stuff needed to submit the job.
        outputDir = self.model_dir / 'PP_output'  # post-processing output goes in Model Dir
        outputDir.mkdir(exist_ok=True, parents=True)
        my_logger.debug(f"Created {outputDir}")
        pp_cmd = self.engine.submit_cmd(pp_cmd, f"PP_{self.name}",
                                        outdir=outputDir,
                                        hold=True,
                                        rundir=self.model_dir,
                                        **job_params)  # generate the submit cmd.
        # note the post-processing is submitted "held".It needs to be released once the model
        # has actually finished. That could require multiple simulations. So we don't hold it on the model
        # and instead will explicitly release it when status gets set to SUCCEEDED
        output = self.run_cmd(pp_cmd)  # submit the post-processing job.
        my_logger.debug(f"post-processing run {pp_cmd} and got {output}")
        pp_jid = self.engine.job_id(output)  # extract the job-ID.
        return pp_jid

    def submit_model(self,
                     fake_function: typing.Optional[typing.Callable[[dict], pd.Series]] = None,
                     ) -> typing.Optional[str]:
        """
        Submit a model and its post-processing.
        Post-processing gets submitted first but held.
        Model gets cmd to release_job post-processing included before it gets submitted.

        :param fake_function -- if provided, no submission will be done.
          Instead, this function will be used to generate fake obs.
          Designed for testing code that runs whole algorithms.
          Takes one argument -- dict of parameters. Returns pandas series.

        :return: The jobid of the post-process job submitted. (If a post-processing job submitted)
            Post processing runs after the model has completed.
             At the time of submission we don't know what the final model job id is -- because it might self continue.

        Example:
         model.submit_model()
        """
        status: type_status = 'SUBMITTED'
        self.check_status(status)  # check we are allowed to submit.
        pp_jid = None  # unless we do something will have no pp job.

        # deal with fake_function.
        if self.fake and fake_function is None:
            raise ValueError(f"Fake model {self.name} and fake_function not provided. Not allowed")
        if fake_function:  # handle fake function
            self.pp_jid = None  # no cmd to run as we just run it!
            self.simulated_obs = fake_function(self.parameters).rename(self.name)  # compute the simulated_obs
            if not isinstance(self.simulated_obs, pd.Series):
                raise ValueError(f"{fake_function} did not return pandas series. Returned {self.simulated_obs}")

            # check for nulls
            null = self.simulated_obs.isnull()
            if np.any(null):
                raise ValueError("Fake function produced null values at: " + ", ".join(self.simulated_obs.index[null]))
            self.fake = True  # we are faking it!
            my_logger.info(f"Using fake functions {fake_function.__name__}")
            my_logger.info(f"Faking {self.name}")
            # work through rest of order.
            self.set_status(status)
            self.running()  # running stuff
            self.succeeded()  # succeeded stuff
            self.process()  # and process.
            return pp_jid  # no post-processing submitted.

        # Actually running a model now
        # first sort out the post-processing.
        if self.is_continuable():  # Model would like to continue. So no pp submission.
            # But check have a pp_jid and fail if not
            if self.pp_jid is None:
                raise ValueError(f"self.pp_jid is None. Should be set to a job id of a post-processing job")
            pp_jid = None
        else:  # starting so generate and submit a post processing job.
            if self.pp_jid is not None:  # self.pp_jid should be None. Fail if not!
                raise ValueError(f"Have pp_jid {self.pp_jid} should be None")
            pp_jid = self.submit_post_process()
            self.pp_jid = pp_jid

        # Done submitting (if needed) a post-processing job. Now submit the model!
        # Model has been modified so that will run model.set_status("SUCCEEDED")
        # which will release the post-processing job.
        cmd = self.submit_cmd()  # cmd that submits the model.
        remote_machine = self.remote.get('remote_machine')
        remote_dir = self.remote.get('remote_model_dir')
        if remote_dir is not None:
            remote_dir = pathlib.PurePath(remote_dir)  # remote_dir should be a string
        cmd = self.ssh_command(cmd, remote_machine=remote_machine, remote_model_dir=remote_dir)
        output = self.run_cmd(cmd)  # and run the command
        ## There is a potential race condition here. In that model could start running
        ## before status gets updated (and model state saved).
        ## One option is to move the set_status before the run_cmd, but then what to do if the run_cmd fails?
        ## Catch and set status to FAILED? generating a warning message.
        ## ALt reset status to INSTANTIATED
        ## Then allow update status for running to have a failed status?
        ## which assumes some intervention outside system.
        jid = self.engine.job_id(output)  # and work out the job id.
        self.submitted_jid = jid  # model will
        my_logger.debug(f"Model submission: ran {cmd} and got {output}")
        self.submission_count += 1  # increase time.
        self.set_status(status)

        return pp_jid  # return the submission  post processing jid (Which will be None if continuing)

    def submit_cmd(self) -> typing.List[str]:
        """"
        Generate the submission command. Over-ride this for your own model.
        If status is INSTANTIATED or PERTURBED then this runs self.engine.submit_cmd on  [self.submit_script] and
        if CONTINUE runs on  [self.continue_script]
        output should go to model_dir/'model_output' which will be created if it does not exist.
        """
        # TODO -- As Model should never be directly instantiated then
        #  consider moving this into simple_model and just having very generic version for Model case.
        # Then individual model classes can run the generic code first and then do their own thing.
        if self.status in ['INSTANTIATED', 'PERTURBED']:
            script = self.submit_script
        elif self.status == 'CONTINUE':
            script = self.continue_script
        else:
            raise ValueError(f"Status {self.status} not expected ")
        runCode = self.run_info.get('runCode')
        runTime = self.run_info.get('runTime', 2000)  # 2000 seconds as default
        # need to (potentially) modify model script so runTime and runCode are set.
        # but in this case just use the submit.
        outdir = self.model_dir / 'model_output'
        outdir.mkdir(parents=True, exist_ok=True)
        my_logger.debug(f"Created {outdir}")

        cmd = self.engine.submit_cmd([pathlib.PurePath(self.model_dir / script)],
                                     f"{self.name}{len(self.model_jids):05d}", outdir,
                                     run_code=runCode, time=runTime, rundir=self.model_dir)
        return cmd

    def running(self, jid: typing.Optional[str] = None) -> typing.Optional[str]:
        """
        Set status to running, store current job id
        : jid -- if not None then use this as job id. Otherwise get job id from engine.
           This allows this method to be called from versions that do more.
        :return: current job id
        """
        status:type_status = 'RUNNING'
        self.check_status(status)  # check we are allowed to set to running.
        if not self.fake and (jid is None):  # not faking and don't have a jid
            my_jid = self.engine.my_job_id()
            my_logger.debug(f"My jobid is {my_jid}")

        else:
            my_jid = jid

        self.model_jids.append(my_jid)

        self.set_status(status)
        return my_jid

    def guess_failed(self) -> bool:
        """
        Guess of a model has failed.
          STATUS = RUNNING or SUBMITTED, and status of last model jobid is unKnown likely means model has failed.
        :return: True if guessed FAILED, False if not
        """

        if self.status in ["RUNNING", "SUBMITTED"]:
            if self.status == "RUNNING":
                model_jid = self.model_jids[-1]
            else:
                model_jid = self.submitted_jid

            stat = self.engine.job_status(model_jid)
            if stat == "notFound":  # no job found.
                my_logger.debug(f"Could not find status for jid:{model_jid} for model {self}. Setting status to FAILED")
                self.set_failed()  # we have failed.
                return True

        return False  # this model is not having its status changed.

    def set_failed(self):
        """
        Set status to failed.
        :return:
        """
        self.set_status('FAILED')

    def continue_simulation(self):
        """
        Mark simulation as continuing.
        :return: Nada
        """

        self.set_status("CONTINUE")

    def restart_simulation(self):
        """
        Mark simulation as restarting -- "instantiated"
        :return: nda
        """
        self.set_status("INSTANTIATED")

    def succeeded(self):
        """
        Run the post-processing job by releasing self.jid
        Then set status to SUCCEEDED
        :return: output from running self.post_process_cmd
        """

        status: type_status = 'SUCCEEDED'
        self.check_status(status)
        if self.pp_jid is not None:
            # release_job the post-processing job.
            cmd = self.engine.release_job(self.pp_jid)
            result = self.run_cmd(cmd)
            # will raise an error if it failed.
            my_logger.info(f"Ran post-processing cmd {cmd}")
            output = result
        else:
            my_logger.info("No post-processing processing jid")
            output = None  # no output.

        self.set_status(status)
        return output

    def process(self, update: bool = False,
                ) -> typing.Optional[str]:
        """
        Run the post-processing, store output and set status to PROCESSED.
        "Contract" for a post-processing script
        1) takes a json file as input (arg#1) and puts output in file (arg#2).
        2) It is being ran in the model_directory.
         arg#1 needs json.load to read the json file. Code should expect a dict and use the postProcess entry.
             This allows it to read in and act on a StudyConfig file.
         arg#2 can be .json or .csv or .nc which is where the output will be written.

        :param update If True update the post-processed data which will rereun the post-processing script. State must be Processed.
        :return: output from post-processing.
        """
        # various checks
        if self._post_process_input is None:
            raise FileNotFoundError("Set _post_process_input to something.")
        if self.post_process_cmd_script is None:
            raise FileNotFoundError("Set post_process_cmd_script to something.")
        status: type_status = 'PROCESSED'
        self.check_status(status)  # check we are allowed to set to processed.
        if update and self.status != 'PROCESSED':  # handle updating.
            raise ValueError(f"Updating and status is {self.status} != PROCESSED")

        if self.fake:  # faking?
            my_logger.debug("Faking")
            self.set_status(status)  # just update the status which saves the state
            return

        input_file = self.model_dir / self._post_process_input  # generate json file to hold post process info
        my_logger.debug(f"Dumping post_process to {input_file}")
        output = dict(postProcess=self.post_process)  # wrap post process in dict
        with open(input_file, 'w') as fp:
            json.dump(output, fp, indent=2)
        # dump the post-processing dict for the post-processing to  pick up.

        result = self.run_cmd(self.post_process_cmd_script, cwd=self.model_dir)  # run post-processing
        # get in the simulated obs which also sets them
        obs = self.compute_simulated_observations(use_cache=False)
        if update:  # Update the history for updating
            self.update_history("Reprocessed model")

        my_logger.debug(f"Sim obs are {obs}")
        self.set_status(status)  #  update the status
        return result  # Should this actually return the simulated observations??

    def compute_simulated_observations(self, use_cache:bool = True) -> typing.Optional[pd.Series]:
        """
        Read the post-processed data.
         This default implementation reads, if necessary,  simulated obs from netcdf, json or csv data and
         stores it in the Model as a pandas series. Tests that nothing is null.
         Will read from self.model_dir/self._post_process_output
         If use_cache is True AND self.simulated_obs are not None then just returns the simulated_obs.
        :param use_cache: whether to use cached data if available.
        :return: a pandas series of the simulated obs or None if not available
        """
        # see if we can use cached value to start with.


        if use_cache and (self.simulated_obs is not None):
            if self.status != 'PROCESSED':
                raise ValueError(f"Should not have simulated observations when status is {self.status}")

            return self.simulated_obs # do this first as test cases don't want to set _post_process_output and don't want to read in data.

        if self.status not in ['PROCESSED','SUCCEEDED']: # No obs for this model.
            return None
        if self._post_process_output is None:
            raise FileNotFoundError("self._post_process_output is None. Should be set")

        # Now readin the data.
        post_process_file = self.model_dir / self._post_process_output
        if not post_process_file.is_file():
            raise FileNotFoundError(f"Could not find post-processed file {post_process_file}")
        fileType = post_process_file.suffix  # type of file wanted

        # read in data. Details depend on type of file.
        if fileType == '.nc':  # netcdf
            ds = xarray.load_dataset(post_process_file)
            obs = {var: float(ds[var]) for var in ds.data_vars if ds[var].size == 1}
            my_logger.debug("netcdf file got " + " ".join(obs.keys()))

        elif fileType == '.json':  # json file
            with open(post_process_file, 'r') as fp:
                obs = json.load(fp)
            my_logger.debug("json file got " + " ".join(obs.keys()))

        elif fileType == '.csv':  # data is a csv file.
            obsdf = pd.read_csv(post_process_file, header=None, index_col=False)
            obs = obsdf.to_dict()
            my_logger.debug("csv file got " + " ".join(obs.keys()))

        else:  # don't know what to do. So raise an error
            raise NotImplementedError(f"Do not recognize {fileType}")

        my_logger.info(f"Read {fileType} data from {post_process_file}")
        obs = pd.Series(obs).rename(self.name)


        # check for nulls
        null = obs.isnull()
        if np.any(null):
            raise ValueError("Obs contains null values at: " + ", ".join(obs.index[null]))
        self.simulated_obs = obs # set obs.

        return obs  # return the obs.

    def is_instantiable(self) -> bool:
        """
        Return True if model is instantiable -- which means its status is CREATED
        :return:
        """
        return self.status in ["CREATED"]

    def is_submittable(self) -> bool:
        """
        Return True if model is submittable -- which means its status is CONTINUE or INSTANTIATED or PERTURBED
        :return:
        """
        return self.status in ["CONTINUE", "INSTANTIATED", "PERTURBED"]

    def is_failed(self) -> bool:
        """
        Return True if model has failed
        :return:
        """
        return self.status in ['FAILED']

    def is_continuable(self) -> bool:
        """
        Return True if model is continuable.
        :return: Return True if model is continuable.
        """

        return self.status in ['CONTINUE']

    def is_running(self) -> bool:
        """
        Return True if model is running.
        :return: True if model status is RUNNING
        """
        return self.status in ['RUNNING']

    def is_submitted(self) -> bool:
        """
        Return True if model is submitted.
        :return: True if model status is SUBMITTED
        """
        return self.status in ['SUBMITTED']

    def is_succeeded(self) -> bool:
        """
        Return True if model is succeeded.
        :return: True if model status is SUCCEEDED
        """
        return self.status in ['SUCCEEDED']

    def is_processed(self) -> bool:
        """
        Return True if model is processed.
        :return: True if model status is PROCESSED
        """
        return self.status in ['PROCESSED']

    def delete(self):
        """
        Delete all on disk stuff. Do by deleting all files in self.model_dir and self.config_path. 
        Also remove all associated jobs.
        
        :return: None
        """
        if not self.fake:  # not faking
            self.kill()

        shutil.rmtree(self.model_dir, ignore_errors=True)
        my_logger.info(f"Deleted everything in {self.model_dir}")
        self.config_path.unlink(missing_ok=True)
        return True

    def attrs_for_key(self) -> dict:
        """
        Return dict  that can be used to generate a key for this model.
        :return: a dict.
        """

        params = self.parameters.copy()
        params.update(reference=self.reference)  # add reference




        return params

    @register_param("ensembleMember")
    def ens_member(self, ens_member: typing.Optional[int]) -> None:
        """
        Do nothing as perturbing initial conditions is model-specific. But needed
         for test cases.
        :param ens_member: ensemble member. The ensemble member wanted.
        :return: None (for now) as nothing done.
        """

        inverse = (ens_member is None)
        if inverse:
            my_logger.warning("Can not invert ensMember")
            return None

        my_logger.warning(f"Nothing set for {ens_member}. Override in your own model")
        return None

    def copyConfig(self, direct: pathlib.Path,
                   extra_files: typing.Optional[list[pathlib.Path]] = None,
                   update_paths: bool = True) -> "Model":
        """
        Copy Model to a new directory. Different Model classes may well want to override this by adding their own extra_files
        This basic version will only copy the model config file and the post-processing output file.
           All files must in self.model_dir
        :param direct: directory where Model  is to be copied. Will be created if it does not exist
        :param extra_files: list of extra files (paths provided relative to self.model_dir) to be copied to new directory.

        :param update_paths -- If True then any path parameters will be updated to reflect new directory structure.
           If False and update_parameters is Truety then ValueError will be raised.
        :return: Copied Model. Will copy only Model config & post process unless extra_files provided.

        """

        # check tgt direct path is abs and if not make it abs.
        if not direct.is_absolute():
            direct = pathlib.Path.cwd() / direct
            my_logger.info("Made direct absolute to " + str(direct))
        direct.mkdir(parents=True, exist_ok=True)  # create the directory if needed.

        files_to_copy = [self.config_path.relative_to(self.model_dir), self._post_process_output]

        if extra_files is not None:
            files_to_copy = files_to_copy + extra_files

        files_copied = genericLib.copy_files(self.model_dir, direct, files_to_copy)
        cp_model = copy.deepcopy(self)  # make a copy of self
        miss_files = set(files_to_copy) - set(files_copied)
        if len(miss_files) > 0:
            my_logger.warning(f"{miss_files} missing and not copied")

        cp_config_path = direct / (self.config_path.relative_to(self.model_dir))
        if update_paths:
            cp_model.update_history(f'Copied {len(files_copied)} from {self.model_dir} to {direct}')
            cp_model.model_dir = direct
            cp_model.config_path = cp_config_path
            cp_model.update_history(f"Updated model_dir and config_path")

        cp_model.dump(cp_config_path)  # dump it out! (needed as have changed things so orig copy will have old values)
        return cp_model

    def archive(self,
                archive: tarfile.TarFile,
                root_dir: pathlib.Path,
                extra_files: typing.Optional[typing.List[pathlib.Path | str]] = None):
        """

        :param archive: archive to be added to
        :param root_dir: root to which all files (in archive file) are stored relative to.
           If not None, then the name in the archive will be relative to this path.
        :param extra_files -- extra model things to archive. Should be relative to self.model_dir
        :return: None

        Dump models to tempdir  and only archives files that exist.
        Adds self.config_path and self.model_dir / self._post_process_output to archive.
          If your model wants to include more things in the archive, then overload this method.
          If you call it first using the super method then you just need to add you own stuff to archive!

        Example:
        with tarfile.open(archive_file, "w") as archive:
            model.archive(archive, rootDir=pathlib.Path("my_root_dir")
        """

        # dump the model. TODO: Make dump take a fp or a path. If it has a fileptr then just write to it.
        with tempfile.TemporaryDirectory() as tmpdir:
            # dump the model (but no change to internal values)
            # handle models that were read in and so, potentially, outside rootDir
            tmpdir_pth = pathlib.Path(tmpdir) / self.config_path.name
            self.copyConfig(direct=tmpdir_pth, extra_files=extra_files, update_paths=False)

            for path in tmpdir_pth.rglob("*"):  # get all files in the copied  model dir.
                if path.is_dir():
                    continue  # skip dirs
                arc_path = path.relative_to(tmpdir_pth)
                archive.add(path, arc_path)
                my_logger.debug(f"Added {path} to archive as {arc_path}")

    def reprocess(self, post_process: typing.Optional[dict] = None) -> pd.Series:
        """
        Update already processed observations. State must be processed.
        :param post_process: post-processing info. See set_post_process() method for requirements.
          If not provided then existing post_process info will be used.
        :return: new simulated obs.

        """
        if not self.is_processed():
            raise ValueError(f"Expecting state as processed not {self.status}")

        self.set_post_process(post_process=post_process)
        sim_obs = self.process(update=True)
        return sim_obs

    def to_dict(self) -> dict:
        """
        Convert a Model to a dict  converting paths to PurePaths.
        Most of the work is done by calling the super class to_dict method.
        :return:
        """
        dct = super().to_dict()

        for key in ['config_path', 'model_dir', 'reference']:  # vars to make into purePaths
            dct[key] = pathlib.PurePath(dct[key])
        # deal with configs. Might need similar for Engine
        dct['configs'] = self.configs.to_dict()  # convert configs to a
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> "Model":
        """
        Convert a dict representing the Model to a Model object.
        :param dct: dict to be converted to a model
        :return:a tempModel instance.
        """
        dct2 = cls.convert_pure_paths(dct)
        try:
            dct2['configs'] = namelist_var.GroupConfig.from_dict(
                dct2['configs'])  # convert configs back to a configs object.
        except KeyError:  # old style configs so just let __init__ deal with it.
            pass
        obj = cls(name=dct2.pop('name'), reference=dct2.pop('reference'))  # create a default instance
        obj.fill_attrs(dct2)
        return obj

    def read_nl_value(self, nl_var: NamelistVar,
                      raise_error: bool = True) -> type_allowed_fortran:
        """
        Read value from namelist.
        :param nl_var: namelist variable to read
        :param raise_error: Pass through to configs.read_value
        :return: value obtained by reading the namelist.
        """
        value = self.configs.read_value(nl_var, raise_error=raise_error)
        return value

    def gen_params(self,
                   parameters: typing.Optional[dict] = None) -> dict[NamelistVar,type_allowed_fortran]:
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
                   backup: bool = True):

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
        self.configs.write_values(nl, backup=backup,
                                  create=True)  # and write them all out. Creating new namelist info if needed.

    def read_param(self, parameter: str,
                   raise_error: bool = True) -> type_allowed_fortran:
        """
        Read parameter value from model instance.
        :param parameter: parameter wanted
        :param raise_error: passed through to read_nl_value.
        :return: value. Depends on what is in the model...
        """
        # get the namelist info.
        stuff = self.get_param_info(parameter)[0]  # just want the first element of the list.
        if callable(stuff):  # is it a callable? If so run it in inverse mode.
            result = stuff(self, None)
            my_logger.debug(f"Called {stuff.__qualname__} with inverse and got {result} ")
        else:
            result = self.read_nl_value(stuff, raise_error=raise_error)
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

        stuff = self.get_param_info(parameter)
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

    @classmethod
    def register_param_with_partial(cls,
                                    name: str,
                                    method: typing.Callable,
                                    *args,  # positional arguments to pass to the function,
                                    **kwargs  # keyword arguments to pass to the function
                                    ):

        """
        Register a parameter with a partial method in the param_info.
        Also adds the method to the class with the given name.
        :param name: name of the parameter to register.
        :param method: method to register. This should be a method of the class.
          partial will be used to wrap the method with the given args and kwargs.
        :param args: arguments to pass to partial
        :param kwargs:kwargs to pass to partial
        :return: method
        """

        def make_wrapped_method(method, qualname):
            def wrapper(self, *fargs, **fkwargs):
                return method(self, *fargs, **fkwargs)  # call the method with the class as first argument.

            wrapper.__qualname__ = qualname
            wrapper.__name__ = qualname.split('.')[-1]
            return wrapper

        partial_method = functools.partial(method, *args, **kwargs)
        wrapped_method = make_wrapped_method(partial_method, cls.__name__ + '.' + name)  # wrap the method.

        cls.param_info.register(name, wrapped_method)
        setattr(cls, name, wrapped_method)  # add the method to the class.
        return wrapped_method

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
        status:type_status = 'PERTURBED'
        self.check_status(status)
        if parameters is None:
            parameters = {}
            my_logger.debug("Setting perturb parameters to empty dict")

        self.parameters_no_key.update(copy.deepcopy(parameters))  # set parameters_no_key to the perturbed parameters
        self.set_params(backup=False)  # set parameter values but with no backup done.
        self.update_history(f'Perturbed using {parameters}')  # so at least we can find out what was done
        self.perturb_count += 1
        self.set_status(status)
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
        cls.param_info.update_from_file(filepath, duplicate=duplicate)

    def reload(self):
        """
        Reload model in place..
        """
        new_model_dict = vars(self.load(self.config_path))
        self.fill_attrs(new_model_dict)

    def kill(self, kill_model: bool = True) -> list[str]:
        """
        Kill model related processes
        :param kill_model: If True kill the model job and post-processing job. If False just the post-processing job.
        :return: list of killed job ids. Though submission of models might have things that are not jobids.
        """
        jobs_killed = []  # list of killed jobs
        if kill_model and len(self.model_jids) > 0:  # got some models to kill
            curr_model_id = self.model_jids[-1]
            status = self.model_job_status()
            if status not in ['notFound', None]:
                cmd = self.engine.kill_job(curr_model_id)
                self.run_cmd(cmd)
                my_logger.debug(f"Killed model job id:{curr_model_id}")
                self.update_history(f'Killed model job id:{curr_model_id}')  # update history
                jobs_killed.append(curr_model_id)
            else:
                my_logger.debug(f"Job {curr_model_id} not found.")
        status = self.pp_job_status()
        if status not in ['notFound', None]:  # got a post-processing job.
            cmd = self.engine.kill_job(self.pp_jid)
            self.run_cmd(cmd)
            my_logger.debug(f"Killed post-processing job id:{self.pp_jid}")
            self.update_history(f'Killed post-processing job id:{self.pp_jid}')  # update history
            jobs_killed.append(self.pp_jid)
        else:
            my_logger.debug(f"No post-processing job to kill. pp_jid is {self.pp_jid} and status is {status}")
        return jobs_killed

    def pp_job_status(self) -> typing.Optional[str]:
        """
        Get the status of the post-processing job. Really done so can mock this for testing!
        :return: status of the post-processing job. None if pp_jid is None
        """
        if self.pp_jid is None:
            return None
        else:
            return self.engine.job_status(self.pp_jid)  # return the job status

    def model_job_status(self) -> typing.Optional[str]:
        """
        Get the status of the model job. Really done so can mock this for testing!
        :return: status of the model job. None if no model job submitted.
        Probably will be overwritten in subclasses to return the status of the currently running model job.
        """
        if self.model_jids:  # got some model jid's. Return status of last one
            return self.engine.job_status(self.model_jids[-1])
        else:
            return None

    def calendar(self) -> str:
        """
        Return cftime calendar string for this model.
        :return: string representing the calendar for this model which will be 'standard'
        """
        return 'standard'

    def ssh_command(self, cmd: list[str | pathlib.PurePath],
                    remote_machine: typing.Optional[str] = None,
                    remote_model_dir: typing.Optional[pathlib.PurePath] = None) -> list[str]:
        """
        generate  cmd to run via ssh on remote system. Does not actually run it. Use self.run_cmd to do that.
        :param cmd: command to be run on remote machine as a list of strings.
        :param remote_machine: remote machine name. Any form that ssh accepts.
            If None then input cmd is returned as is.
        :param remote_model_dir: remote path for model_dir. Should be a PurePath.
         If provided then any element of cmd that begins with value of model_dir on local machine
         will be replaced with remote_model_dir and whole thing converted to posix path for remote machine.


        :return: command (a list of strings) that will run cmd on remote_machine via ssh.
        :raises ValueError: if remote_machine is not None and not a str or
        if remote_model_dir is not None and not a PurePath
        """
        if remote_machine is None:
            return cmd  # nothing to be done.

        # check variable types are correct.
        if not isinstance(remote_machine, str):
            raise ValueError(f"remote_node {remote_machine} is not a str")
        if (remote_model_dir is not None) and (not isinstance(remote_model_dir, pathlib.PurePath)):
            raise ValueError(f"remote_model_dir {remote_model_dir} is not a PurePath")
        # generate the remote command.
        remote_cmd: list[str] = []
        for c in cmd:
            if not isinstance(c, (str, pathlib.PurePath)):
                raise ValueError(f"Element {c} of cmd is not a string or a pure path")
            if (remote_model_dir is not None) and isinstance(c, pathlib.PurePath) and c.is_relative_to(self.model_dir):
                # have remote_model_dir and c is a PurePath that is relative to self.model_dir
                # need to replace local model_dir with remote_model_dir
                relative_path = c.relative_to(self.model_dir)
                remote_path = remote_model_dir / relative_path
                remote_cmd.append(remote_path.as_posix())  # convert to posix path for remote machine
            elif isinstance(c, pathlib.PurePath):
                remote_cmd.append(c.as_posix())  # convert to posix path for remote machine
            else:
                remote_cmd.append(c)  # just use as is.

        remote_cmd = ' '.join(remote_cmd)  # make into a single string
        cmd = ['ssh', '-q', '-o', 'batchmode=yes', '-o', 'StrictHostKeyChecking=yes', remote_machine, remote_cmd]
        return cmd  #

    @staticmethod
    def new_path(path: pathlib.PurePath,
                 target_dir: pathlib.PurePath,
                 root_dir: typing.Optional[pathlib.PurePath] = None) -> pathlib.PurePath:
        """
        Create a new path by replacing the root_dir in path with target_dir.
        :param path:path to be modified
        :param target_dir: Target directory to replace root_dir with.
            If root_dir is None or path is not relative to root_dir then path.name is appended to target_dir
        :param root_dir:root dir to use
        :return:new_path with root_dir replaced by target_dir or target_dir/path.name if root_dir is None or path is not relative to root_dir
          AND add_path which is the part added to target_dir.
        """
        if root_dir is not None and path.is_relative_to(root_dir):
            add_path = path.relative_to(root_dir)
        else:
            add_path = path.name
        new_path = target_dir / add_path
        return new_path

    def install_remote_command(self,
                               remote_machine: typing.Optional[str] = None,
                               remote_model_dir: typing.Optional[pathlib.PurePath] = None) -> typing.Optional[
        list[list[str | pathlib.PurePath]]]:
        """
        Generate list of cmds to install on remote machine. Use run_cmd to actually do it.
        WIll return None if no remote machine or remote_dir
        :param remote_machine: remote machine -- remote machine to install on.
        :param remote_model_dir: remote directory to install to.
        if either remote_machine or remote_model_dir is None then nothing is done and None is returned.
        Commands returned are
        1) Create remote dir
        2) rsync model_dir to remote_dir

        :raises ValueError: if remote_dir is not Nome and not a PurePath or remote_dir is not None and not a str

        Thoughts -- could work with remote_machine not set by just returning cmd to rsync to remote_model_dir (with potential path adjustment).
        But for now require both to be set.
         A more generic way of doing this is to have a script that gets ran to do the installation.
           That then offloads the details of how to do the install to that script.
           For example could use rsync or scp or globus

        :return: list of commands to run. Each element is a  list of strings/purePaths to run.
        """
        if (remote_model_dir is None) or (remote_machine is None):
            my_logger.debug(f"Remote machine or  remote dir are not set.")
            return None  # nothing to be done.
        # check variable types are correct.
        if not isinstance(remote_model_dir, pathlib.PurePath):
            raise ValueError(f"remote_dir {remote_model_dir} is not a PurePath")
        if not isinstance(remote_machine, str):
            raise ValueError(f"remote_machine {remote_machine} is not a string")

        my_logger.debug(f"Will install {self.model_dir} to {remote_model_dir} on {remote_machine}")

        remote_path = remote_model_dir.as_posix().rstrip('/')  # get as posix path for remote machine.
        cmd0 = self.ssh_command(['mkdir', '-p', remote_path],
                                remote_machine=remote_machine)  # cmd to create remote dir if needed.
        # now create cmd to rsync the model dir to the remote dir. rsync will create remote_dir if it does not exist.
        ssh_opts = ["-o", "BatchMode=yes", "-o",
                    "StrictHostKeyChecking=yes"]  # run in batch mode with strict host key checking
        ssh_command = "ssh " + " ".join(shlex.quote(opt) for opt in ssh_opts)
        cmd = ['rsync', '-a', '-q', "-e", ssh_command, str(self.model_dir) + '/', f"{remote_machine}:{remote_path}"]
        # note no trailing slash so we copy the model_dir to remote_path NOT into remote_path (as would happen with a trailing slash)

        return [cmd0, cmd]

    def update_params(self, parameters: typing.Optional[list[str]] = None,
                      update: bool = True) -> pd.Series:
        """
        Update the parameters in a model by reading them from disk.
        Model is not written to disk.
        :param parameters: parameters to update. Existing parameters will be updated too.
        :param update: If True then update the model. If False just return the updated parameters.
        :return: Pandas series containing updated values
        """
        # check been instantiated!
        if self.is_instantiable():
            raise ValueError("Model not instantiated. Cannot update parameters.")
        if parameters is None:
            parameters = []
        params_to_update = set(list(self.parameters.keys()) + parameters)  # list of unique params
        new_params = self.read_params(list(params_to_update), fail=True)  # read the param values
        my_logger.debug(f'Updated params are {params_to_update}')
        if update:
            self.parameters.update(new_params)  # update
            self.update_history(f"Updated parameters {params_to_update}")
        result = pd.Series(new_params).rename(self.name)

        return result

    def config_name(self) -> str:
        """
        Returns the configuration name which is the reference_name + ensembleMember (or 0)
        :return: name
        """
        ensemble_member = self.parameters.get('ensembleMember', 0)
        config_name = f"{self.reference_name}#{ensemble_member}"
        return config_name

    def update_reference_name(self, reference_name: str,
                              dump: bool = True):
        """
        Update the reference name for this model. This will update, if necessary, the reference_name attribute.
        :param reference_name: new reference_name to use.
        :param dump: If True dump the model config to disk to save the updated reference_name. If False then just update the reference_name in memory.
        :return: None
        """
        if self.reference_name != reference_name:  # only update if reference name different to avoid unnecessary updates and dumps.
            my_logger.warning(f"Updating reference from {self.reference_name} to {reference_name}")
            self.reference_name = reference_name
            self.update_history(f"Updated reference_name  to {reference_name}")
            if dump:
                self.dump(self.config_path)  # dump the model to disk to save the updated reference.


Model.register_class(Model)  # register ourselves!

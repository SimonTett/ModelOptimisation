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
from __future__ import annotations # TODO remove use of this.

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
from namelist_var import NamelistVar,GroupConfig,type_allowed_fortran
from engine import abstractEngine

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

type_status = typing.Literal['CREATED', 'INSTANTIATED', 'SUBMITTED',
'RUNNING', 'FAILED', 'PERTURBED', 'CONTINUE',
'SUCCEEDED', 'PROCESSED']  # allowed strings for status


class Model(ModelBaseClass, journal):
    # type definitions for attributes.
    name: str
    config_path: typing.Union[pathlib.Path, pathlib.PurePath]
    reference:typing.Union[pathlib.Path, pathlib.PurePath]
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
    submit_script: typing.Union[pathlib.Path ,pathlib.PurePath]
    continue_script: typing.Union[pathlib.Path, pathlib.PurePath]
    set_status_script: typing.Union[pathlib.Path, pathlib.PurePath]
    status: type_status
    simulated_obs: typing.Optional[pd.Series]
    _post_process_input: typing.Optional[str]
    _post_process_output: typing.Optional[str]
    configs: GroupConfig

    """
    Abstract model class. Any class that inherits from this will have name lookup.
    Also provides top-level methods and class methods.
    Model.load_model() will load a model from disk. An object of the appropriate class will be returned as long as that
    class inherits from Model.
    As it inherits from journal  it has methods to update history,output and run commands.
    public attributes: (Be careful if you  change them)
        model_dir -- directory where model information is stored
        reference --  where the reference configuration came from.
        config_path -- where the configuration is to be written to (or was read from).
        name -- name of the model
        status -- status of the model
        post_process -- post-processing information. See Model.set_post_process for details.
        fake -- If True model is faked.
        perturb_count -- no of times perturbation has been done.
        parameters -- dict of parameters/values. Used to generate key and set values.
        parameters_no_key -- dict of parameters/values. Overrides parameters to set values and do form part of the key
        set_status_script -- path to script that sets_status. Your model will need to call this.
        engine -- submission engine.
        pp_jid -- post-processing job id. This gets released when model status changes to SUCCEEDS
        model_jids -- list of model job ids.
        configs - cache of configuration files (namelists or equivalent)
        pertub_count -- no of times model has been perturbed.
        submission_count -- no of times model has been submitted.
        
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
                       RUNNING=["SUBMITTED","RUNNING"],  # running the model should have been submitted or already been running
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
        Load a configuration
        :param model_path:  where the configuration is stored
          config_path will be set to model_path
          model_dir will be set to model_path.parent.
          warnings given if these are changes.
        :return: loaded model
        """
        model = super().load(model_path)  # Using json "magic". See generic_json for what actually happens.

        if not model.config_path.samefile(model_path):
            my_logger.warning(f"Model {model} model_path changed to {model_path}")
            model.config_path = model_path  # Replace config_path with where we actually loaded it from.

        if not model.model_dir.samefile(model_path.parent):
            my_logger.warning(f"Model {model} model_dir changed to {model_path.parent} ")
            model.model_dir = model_path.parent # update directory with where we actually loaded it from.
        return model


    # methods now.
    def __init__(self,
                 name: str,
                 reference: pathlib.Path,
                 post_process: typing.Optional[dict] = None,
                 model_dir: pathlib.Path = pathlib.Path.cwd(),
                 config_path: typing.Optional[pathlib.Path] = None,
                 status: type_status = "CREATED",
                 parameters: typing.Optional[dict] = None,
                 engine: typing.Optional[abstractEngine] = None,
                 run_info: typing.Optional[dict] = None,
                 fake: bool = False,
                 study: typing.Optional["Study"] = None):
        # TODO add in verbose option so that set_model_status script has verbose options provided in.
        """
        Initialize the Model class.

        :param name -- name of model
        :param reference -- reference directory. Should be a pathlib.Path
                keyword arguments
        :param model_dir --- where model will be created and any files written.
             Should be a pathlib.Path. Will, if needed, be created. If node cwd will be used.
             Must be different from reference
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
        :param study -- a study. This is there in case model wants to interrogate it at init time.
        It is recommended that study **not** be stored as an attribute.
           If you do take great care and worry about recursion as study stores models.
           Note that this implementation does not take use study
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
        self.model_dir = model_dir
        if status not in self.allowed_status:
            raise ValueError(f"Status {status} not in " + " ".join(self.allowed_status))

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

        if engine is None: # no submission engine provided. So create one. Default will be SLURM.
            engine = abstractEngine.create_engine(engine_name=self.run_info.get('submit_engine','SLURM'),
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

        # Set status
        self.status = status
        if self.status == 'CREATED':  # creating model for the first time
            self.update_history("CREATING model")
            # and simulated obs.
        self.simulated_obs = None
        self.configs = GroupConfig(root_dir=self.model_dir) # grouped configs for writing out generic namelists

    @classmethod
    def get_param_info(cls,parameter:str) -> list[NamelistVar|typing.Callable]:
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
                    break # exit the loop as we have found the parameter.
        if stuff is None:
            raise KeyError(f"Parameter {parameter} not found.\n Allowed parameters are: " +
                           " ".join(cls.known_parameters()))
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
        post_process will be deep-copyed to self.post_process with script, interp, input_file, output_file removed
        self.post_process_cmd_script will hold the command to run the post-processing.
        :return: None
        """
        if post_process is None:
            return
        pp = copy.deepcopy(post_process)
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

    def __eq__(self, other,vars_to_ignore:typing.Optional[typing.List[str]] = None):
        """
        Compare two objects and identify their differences by comparing their attributes.
        :param other: The other object to compare against.
        :param vars_to_ignore: List of variables to ignore in the comparison. Will always have configs appended
        :return: Set of differing attributes between the objects.
        """
        if vars_to_ignore is None:
            vars_to_ignore = []
        vars_to_ignore.append('configs')
        result = super().__eq__(other,vars_to_ignore=vars_to_ignore)
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

        if type(self) != type(other):
            return {'Different types:', type(self), type(other)}

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
        s = f"Type: {self.class_name()} Name: {self.name}" \
            f" Status: {self.status} Nparams: {len(self.parameters)} Last Modified:{last_hist_key}"
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
                result[parameter] = self.read_param(parameter)
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
                     direct:typing.Optional[pathlib.Path] = None,
                     copy_ref:bool=True):
        """
        Create a new model by copying reference. If self.fake is True then no copy is done.
         Overwrite (and call superclass) for your own model.
         :param direct: path to directory to create. If None then self.model_dir will be used.
         :param copy_ref: If True copy reference directory to dir
        For example if you want to modify your reference model.
        :return:nothing.
        """
        if direct is None:
            direct = self.model_dir
        direct.mkdir(parents=True, exist_ok=True)  # create the directory if needed.
        my_logger.info(f"Created {direct}")
        if not self.fake:
            # empty the directory (if we are creating)
            for file in direct.iterdir():
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()
            if copy_ref:
                shutil.copytree(str(self.reference), str(direct), symlinks=True, dirs_exist_ok=True)  # copy from reference.
                my_logger.info(f"Copied {self.reference} to {direct}")

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

        if check_allowed: # only check
            if new_status not in self.allowed_status:
                raise ValueError(f"Status {new_status} should be one of " + " ".join(self.allowed_status))
            if new_status == 'CREATED':
                raise ValueError(f"Do not set status to CREATED")
        
        expected_status = self.status_info[new_status]
        if check_existing and (self.status not in expected_status):
            raise ValueError(
                f"Expected current status {self.status}  to be one of " + " ".join(
                    expected_status) + f" as changing to {new_status}")
        my_logger.debug(f"Changing status from {self.status} to {new_status}")
        self.update_history(f"Status set to {new_status} in {self.model_dir}")
        self.status = new_status
        self.dump_model()  # write to disk

    def instantiate(self,fake:bool = False) -> None:
        """
        Run create_model and set_params, update status.
        If fake is True then do not actually create the model.
         Just create the directory. Actual behaviour depends on class
         implementations of create_model, modify_model and set_params.
        Will verify that (if defined) post-processing script exists failing if not
        Also, should make any changes needed to those files.
        :return:
        """
        self.fake = self.fake or fake
        # if fake is True then we are faking it unless we are already faking it!
        self.create_model()  # create model
        self.modify_model()  # do any modifications to model needed before setting params.
        self.set_params()  # set the params
        self.check() # check the model is ok. Very model dependent.
        # set permissions to rxw,rx,rx for submit and continue script.
        if not fake:
            for file in [self.submit_script, self.continue_script]:
                if file is not None:
                    (self.model_dir / file).chmod(0o766)  # set permission
        else:
            self.fake = True # we are faking now!

        self.set_status('INSTANTIATED')

    def modify_model(self):
        """
        Modify model.This method is minimal; call from your own clqss
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
        job_params = self.engine.extract_job_submission_params(self.run_info,default_values=dict(runTime=1800)) # extract stuff needed to submit the job.
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
        pp_jid = None  # unless we do something will have no pp job.

        # deal with fake_function.
        if self.fake and fake_function is  None:
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
        output = self.run_cmd(cmd)  # and run the command
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
        # Then indiviudal model classes can run the generic code first and then do their own thing.
        if self.status in ['INSTANTIATED', 'PERTURBED']:
            script = self.submit_script
        elif self.status == 'CONTINUE':
            script = self.continue_script
        else:
            raise ValueError(f"Status {self.status} not expected ")
        runCode = self.run_info.get('runCode')
        runTime = self.run_info.get('runTime', 2000)  # 2000 seconds as default.
        # need to (potentially) modify model script so runTime and runCode are set.
        # but in this case just use the submit.
        outdir = self.model_dir / 'model_output'
        outdir.mkdir(parents=True, exist_ok=True)
        my_logger.debug(f"Created {outdir}")

        cmd = self.engine.submit_cmd([str(self.model_dir / script)], f"{self.name}{len(self.model_jids):05d}", outdir,
                                     run_code=runCode, time=runTime, rundir=self.model_dir)
        return cmd

    def running(self) -> typing.Optional[str]:
        """
        Set status to running, store current job id
        :return: current job id
        """
        if not self.fake:  # faking so no job id.
            my_jid = self.engine.my_job_id()
            my_logger.debug(f"My jobid is {my_jid}")

        else:
            my_jid = None

        self.model_jids.append(my_jid)

        self.set_status('RUNNING')
        return my_jid

    def guess_failed(self) -> bool:
        """
        Guess of a model has failed.
          STATUS = RUNNING or SUBMITTED and status of last model jobid is unKnown likely means model has failed.
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
             This allows it ot read in and act on a StudyConfig file.
         arg#2 can be .json or .csv or .nc

        :param update If True update the post-processed. State must be Processed.
        :return: output from post-processing.
        """
        status: type_status = 'PROCESSED'
        if update:  # handle updating.
            if self.status != 'PROCESSED':
                raise ValueError(f"Updating and status is {self.status} != PROCESSED")

        if self.fake:  # faking?
            my_logger.debug("Faking")
            self.set_status(status)  # just update the status which saves the state
            return

        input_file = self.model_dir / self._post_process_input  # generate json file to hold post process info
        my_logger.debug(f"Dumping post_process to {input_file}")
        output = dict(postProcess=self.post_process)  # wrap post process in dict
        with open(input_file, 'w') as fp:
            json.dump(output, fp,indent=2)
        # dump the post-processing dict for the post-processing to  pick up.

        post_process_output = self.model_dir / self._post_process_output
        result = self.run_cmd(self.post_process_cmd_script, cwd=self.model_dir)  #

        # get in the simulated obs which also sets them 
        self.read_simulated_obs(post_process_output)
        
        if update:  # if we are updating there is no status change -- we have already checked we are processed.
            self.update_history("Reprocessed model")

        my_logger.debug(f"Sim obs are {self.simulated_obs}")
        self.set_status(status)  #  update the status (and dump state to disk)
        return result # Should this actually return the simulated observations??

    def read_simulated_obs(self, post_process_file: pathlib.Path):
        """
        Read the post processed data.
         This default implementation reads simulated obs from netcdf, json or csv data and
         stores it in the Model as a pandas series. Tests that nothing is null.
         :param post_process_file: path to the post processed data containing the simulated observations,
        :return: a pandas series of the simulated
        """

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
        self.simulated_obs = obs

        # check for nulls
        null = obs.isnull()
        if np.any(null):
            raise ValueError("Obs contains null values at: " + ", ".join(obs.index[null]))

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
        if not self.fake: # not faking
            self.kill()


        shutil.rmtree(self.model_dir, ignore_errors=True)
        my_logger.info(f"Deleted everything in {self.model_dir}")
        self.config_path.unlink(missing_ok=True)
        return True

    def key(self, fpFmt: str = '%.4g') -> str:
        """
        Generate key from keys and values in self.parameters.
        This should be unique (to some rounding on float parameters)
        :param fpFmt -- format to convert float to string. (Default is %.4g)
        :return: a tuple as an index. tuple is key_name, value in sorted order of key_name.
        """
        keys = []
        paramKeys = sorted(self.parameters.keys())  # fixed ordering
        # Iterate over sorted parameter names.
        for k in paramKeys:  # iterate over keys in sorted order.
            keys.append(k)
            v = self.parameters[k]
            if isinstance(v, float):
                keys.append(fpFmt % v)  # float point number so use formatter.
            else:  # just append the value.
                keys.append(repr(v))  # use the object repr method.
        keys = tuple(keys)  # convert to tuple
        return str(keys)  # and then to a string.

    @register_param("ensembleMember")
    def ens_member(self, ensMember: typing.Optional[int]) -> None:
        """
        Do nothing as perturbing initial conditions is model-specific. But needed
         for test cases.
        :param ensMember: ensemble member. The ensemble member wanted.
        :return: None (for now) as nothing done.
        """

        inverse = (ensMember is None)
        if inverse:
            my_logger.warning("Can not invert ensMember")
            return None

        my_logger.warning(f"Nothing set for {ensMember}. Override in your own model")
        return None


    def copy(self, direct: pathlib.Path,
             extra_files: typing.Optional[typing.List[pathlib.Path | str]] = None,
             link_paths: typing.Optional[typing.List[pathlib.Path | str]] = None):
        """
        Copy model to new place

        :param direct: Where new model_dir is
        :param extra_files: Extra files to be copied
        :param link_paths: Paths to be linked (rather than copied). Useful, for large files that will be read.
           On some OS's links are only allowed if you are on the same filesystem.
        :return: Modified model.
        """
        if extra_files is None:
            extra_files = []
        if link_paths is None:
            link_paths = []

        if direct.samefile(self.model_dir):  # Check we are not wiping out ourselves.
            raise ValueError(f"Copying {direct} to {self.model_dir} which is the same path")

        # make needed directories. Do in one go so to speed things up (a bit).
        dirs_to_make = {direct} | \
                       {(direct / p).parent for p in extra_files} | \
                       {(direct / p).parent for p in link_paths}  # unique set of directories needed
        for direct in dirs_to_make:
            direct.mkdir(exist_ok=True, parents=True)  # make any needed directories.
            my_logger.debug(f"Created {dir}")

        cp_model = copy.deepcopy(self)
        cp_model.model_dir = direct
        cp_model.config_path = direct / self.config_path.relative_to(self.model_dir)

        # copy extra paths + post_process_output
        paths_to_copy = [self.model_dir / self._post_process_output] + [self.model_dir / p for p in extra_files]
        for path in paths_to_copy:
            if path.exists():
                cp_path = direct / path.relative_to(self.model_dir)  # where it goes
                shutil.copy2(path, cp_path)  # copy it
                msg = f"Copied {path} to {cp_path}"
                my_logger.debug(msg)
                cp_model.update_history(msg)

        # Do links. Note potential problems with filesystems. Will fix if they are a problem.
        for p in link_paths:
            path = self.model_dir / p
            new_path = direct / p
            path.hardlink_to(new_path)
            msg = f"Linked {new_path} to {path}"
            my_logger.debug(msg)
            cp_model.update_history(msg)

        cp_model.update_history(f"Copied from {self.config_path} to {cp_model.config_path}")
        cp_model.dump_model()
        return cp_model

    def archive(self,
                archive: tarfile.TarFile,
                rootDir: pathlib.Path,
                extra_files: typing.Optional[typing.List[pathlib.Path | str]] = None):
        """

        :param archive: archive to be added to
        :param rootDir: root to which all files (in archive file) are stored relative to.
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
        if extra_files is None:
            extra_files = []
        # dump the model. TODO: Make dump take a fp or a path. If it has a fileptr then just write to it.
        with tempfile.TemporaryDirectory() as tmpdir:
            # dump the model (but no change to internal values)
            tmpfile = pathlib.Path(tmpdir) / self.config_path.name
            self.dump(tmpfile)
            arc_path = self.config_path.relative_to(rootDir)
            archive.add(tmpfile, arc_path)
            my_logger.debug(f"Added {self} to archive as {arc_path}")

        paths_to_archive = [self.model_dir / self._post_process_output] + [self.model_dir / p for p in extra_files]

        for path in paths_to_archive:
            arc_path = path.relative_to(rootDir)
            if path.exists():
                archive.add(path, arc_path)  # archive the file with name relative to root
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
    def from_dict(cls, dct: dict) -> Model:
        """
        Convert a dict representing the Model to a Model object.
        :param dct: dict to be converted to a model
        :return:a tempModel instance.
        """
        dct2 = cls.convert_pure_paths(dct)
        dct2['configs'] = namelist_var.GroupConfig.from_dict(dct2['configs'])  # convert configs back to a configs object.
        obj = cls(name=dct2.pop('name'), reference=dct2.pop('reference'))  # create a default instance
        obj.fill_attrs(dct2)
        return obj


    def read_nl_value(self,nl_var:NamelistVar,
                      raise_error:bool = True) -> type_allowed_fortran:
        """
        Read value from namelist.
        :param nl_var: namelist variable to read
        :param raise_error: Pass through to configs.read_value
        :return: value obtained by reading the namelist.
        """
        value = self.configs.read_value(nl_var,raise_error=raise_error)
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
        self.configs.write_values(nl,backup=backup,create=True)  # and write them all out. Creating new namelist info if needed.

    def read_param(self, parameter: str,
                   raise_error:bool = True) -> type_allowed_fortran:
        """
        Read parameter value from model instance.
        :param parameter: parameter wanted
        :param raise_error: passed through to read_nl_value.
        :return: value. Depends on what is in the model...
        """
        # get the namelist info.
        stuff = self.get_param_info(parameter)[0]# just want the first element of the list.
        if callable(stuff):  # is it a callable? If so run it in inverse mode.
            result = stuff(self, None)
            my_logger.debug(f"Called {stuff.__qualname__} with inverse and got {result} ")
        else:
            result = self.read_nl_value(stuff,raise_error=raise_error)
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
        if parameters is None:
            parameters = {}
            my_logger.debug("Setting perturb parameters to empty dict")

        self.parameters_no_key.update(copy.deepcopy(parameters))  # set parameters_no_key to the perturbed parameters
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
        cls.param_info.update_from_file(filepath, duplicate=duplicate)

    def reload(self):
        """
        Reload model in place..
        """
        new_model_dict = vars(self.load(self.config_path))
        self.fill_attrs(new_model_dict)

    def kill(self,kill_model:bool=True) -> list[str]:
        """
        Kill model related processes
        :param kill_model: If True kill the model job and post-processing job. If False just the post-processing job.
        :return: list of killed job ids. Though submission of models might have things that are not jobids.
        """
        jobs_killed = []  # list of killed jobs
        if kill_model and len(self.model_jids) > 0:  # got some models to kill
            curr_model_id = self.model_jids[-1]
            status = self.model_job_status()
            if status not in ['notFound',None]:
                cmd = self.engine.kill_job(curr_model_id)
                self.run_cmd(cmd)
                my_logger.debug(f"Killed model job id:{curr_model_id}")
                self.update_history(f'Killed model job id:{curr_model_id}')  # update history
                jobs_killed.append(curr_model_id)
            else:
                my_logger.debug(f"Job {curr_model_id} not found.")
        status = self.pp_job_status()
        if status not in ['notFound',None]:  # got a post-processing job.
            cmd = self.engine.kill_job(self.pp_jid)
            self.run_cmd(cmd)
            my_logger.debug(f"Killed post-processing job id:{self.pp_jid}")
            self.update_history(f'Killed post-processing job id:{self.pp_jid}')  # update history
            jobs_killed.append(self.pp_jid)
        else:
            my_logger.debug(f"No post-processing job to kill. pp_jid is {self.pp_jid} and status is {status}")
        return jobs_killed

    def pp_job_status(self) ->typing.Optional[str]:
        """
        Get the status of the post-processing job. Really done so can mock this for testing!
        :return: status of the post-processing job. None if pp_jid is None
        """
        if self.pp_jid is None:
            return None
        else:
            return self.engine.job_status(self.pp_jid) # return the job status

    def model_job_status(self) -> typing.Optional[str]:
        """
        Get the status of the model job. Really done so can mock this for testing!
        :return: status of the model job. None if no model job submitted.
        Probably will be overwritten in subclasses to return the status of the currently running model job.
        """
        if self.model_jids: # got some model jid's. Return status of last one
            return self.engine.job_status(self.model_jids[-1])
        else:
            return None

    def calendar(self) -> str:
        """
        Return cftime calendar string for this model.
        :return: string representing the calendar for this model which will be 'standard'
        """
        return 'standard'


Model.register_class(Model)  # register ourselves!

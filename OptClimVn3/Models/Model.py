"""
Class for abstract Model and namelists.
"""
from __future__ import annotations

import copy
import logging
import os
import pathlib
import shutil
import typing

import numpy as np
import pandas as pd
import xarray

import json
from model_base import journal
from ModelBaseClass import ModelBaseClass, register_param
from namelist_var import namelist_var
import engine


# from Study import Study


class Model(ModelBaseClass, journal):
    """
    Abstract model class. Any class that inherits from this will have name lookup.

    Also provides top-level methods and class methods.
    Model.load_model() will load a model from disk. An object of the appropriate class will be returned as long as that
    class inherits from Model.
    As it inherits from history it has methods to update history,output nad run commands.

    """
    # logging.getLogger(__name__) # setup logger for the class.
    post_proccess_json = "post_process.json"  # where post-process info gets written
    status_info = dict(CREATED=None,
                       INSTANTIATED=["CREATED"],  # Instantiate a model requires it to have been created
                       SUBMITTED=['INSTANTIATED', 'PERTURBED', 'CONTINUE'],
                       # Submitting needs it to have been instantiated, perturbed or to be continued.
                       RUNNING=["SUBMITTED"],  # running needed it should have been submitted
                       FAILED=["RUNNING"],  # Failed means it should have been running
                       PERTURBED=["FAILED"],  # Allowed to perturb a model after it failed.
                       CONTINUE=["FAILED", "PERTURBED"],
                       # failed can just be continued or can be perturbed. For example ran out of time or disk space
                       # full
                       SUCCEEDED=["RUNNING"],  # SUCCEEDED means it should have been running
                       PROCESSED=['SUCCEEDED'])  # Processed means it should have succeeded.
    # Q Perturbed comes in two flavours. Perturb and continue or perturb and restart. How to handle that?
    allowed_status = set(status_info.keys())

    @classmethod
    def from_dict(cls, dct: dict):
        """
        Initialises using name and reference values in dct (popping them out so they don't get added twice)
        Then copies keys to attributes in obj
        but only those that  exist after initialisation.
        This is really a factory method
        :param dct: dict containing information needed by class_name.from_dict()
        :return: initialised object
        """
        obj = cls(name=dct.pop('name'), reference=dct.pop('reference'))  # create an default instance
        for name, value in dct.items():
            if hasattr(obj, name):
                setattr(obj, name, value)
            else:
                logging.warning(f"Did not setattr for {name} as not in obj")
        return obj

    # methods now.
    def __init__(self,
                 name: str,
                 reference: pathlib.Path,
                 post_process: typing.Optional[dict] = None,
                 model_dir: pathlib.Path = pathlib.Path.cwd(),
                 config_path: typing.Optional[pathlib.Path] = None,
                 status: str = "CREATED",
                 fake: bool = False,
                 submission_count: int = 0,
                 run_count: int = 0,
                 perturb_count: int = 0,
                 parameters: typing.Optional[dict] = None,
                 post_process_cmd: typing.Optional[typing.List] = None,
                 simulated_obs: typing.Optional[pd.Series] = None,
                 study: typing.Optional["Study"] = None):
        """
        Initialise the Model class.

        :param name -- name of model
        :param reference -- reference directory. Should be a pathlib.Path
                keyword arguments
        :param model_dir --- where model will be created and any files written.
             Should be a pathlib.Path. Will, if needed, be created. If node cwd will be used.
             Must be different from reference
        :param config_path: Where configuration will be created.
               If not defined (or None) will be model_dir/(self.name+".mcfg")
        :param status -- model status. Default = "CREATED"
        :param fake -- if True model is faking running/post-processing
        :param submission_count -- no of times model has been submitted
        :param run_count -- no of times model has been running
        :param perturb_count -- no of times model has been perturbed.
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
        :param post_process_cmd -- command to run or release_job the post-processing.
        Only used when status gets set to COMPLETED.
        :param simulated_obs -- value of simulated_obs.
        :param study -- a study. This is there in case model wants to interrogate it at init time.
        It is recommended that study **not** be stored as an attribute.
           If you do take great care and worry about recursion as study stores models.
           Note that this implementation does not take use study

        All (except study) are stored as attributes and are publicly available
          though user should be careful if they modify them.
        public attributes:
        model_dir -- directory where model information is stored
        reference --  where the reference configuration came from.
        config_path -- where the configuration is to be written to (or was read from).
        name -- name of the model
        status -- status of the model
        post_process -- post-processing information. See Model.set_post_process for details.
        post_process_cmd_script -- cmd to be run for post-processing.
        post_process_cmd -- cmd to run the post-processing. Depends on your submission/job management system
        fake -- If True model is faked.
        perturb_count -- no of times perturbation has been done.
        parameters -- dict of parameters/values
        set_status_script -- path to script that sets_status. Your model will need to call this.

        Private attributes:
          _post_process_input -- name of input file for post-procesing
          _post_process_output -- name of input file for post-processing

          Note that update_history and store_output (see Journal for doc for those) set up private attributes.

        """
        # set up default values.
        self.post_process = {}  # where all post-processing info stored.
        self.post_process_cmd_script = None  # cmd (list) to be run for post-processing.
        self._post_process_input = None  # filename where input for post-processing goes
        self._post_process_output = None  # filename where output for post-processing goes.
        self.fake = fake  # did we fake the model? Changed at submission.
        self.perturb_count = perturb_count  # how many times have we perturbed the model?
        self.submission_count = submission_count  # how mamy times have we submitted the model?
        self.run_count = run_count  # how many times has the model been run

        self.update_history(None)  # init history.
        self.store_output(None, None)  # init store output

        if parameters is None:
            parameters = {}
        else:
            parameters = copy.deepcopy(parameters)
        # TODO check that parameters exist in lookup.

        self.name = name

        if config_path is None:
            config_path = model_dir / (self.name + '.mcfg')

        self.config_path = config_path
        if (model_dir == reference) or (model_dir.exists() and reference.samefile(model_dir)):
            raise ValueError(f"Model_dir {model_dir} is the same as reference {reference}")

        self.reference = reference
        self.model_dir = model_dir
        if status not in self.allowed_status:
            raise ValueError(f"Status {status} not in " + " ".join(self.allowed_status))

        self.parameters = parameters
        self.post_process_cmd = post_process_cmd
        # attributes that get defined based on inputs or are just fixed.
        if post_process is not None:
            self.set_post_process(post_process)
        # setup submit and continue script
        self.submit_script = "submit.sh"
        self.continue_script = "continue.sh"
        # setup path to where script that sets status is. TODO: Is this actually needed?
        root = self.expand("$OPTCLIMTOP/OptClimVn3")
        script_pth = root / "scripts/set_model_status.py"
        self.set_status_script = script_pth

        self.status = status
        if self.status == 'CREATED':  # creating model for the first time
            self.update_history("CREATING model")
        self.simulated_obs = simulated_obs

    def set_post_process(self, post_process: typing.Optional[dict] = None):

        """
        Set up post_process info.
        :param post_process -- None or  dict containing information for post-processing.
            If None will immediately return without doing anything.
        If dict must include:
            script -- full path to script to run. Will be passed through self.expand to expand vars and user id.
             Will be checked for existence, and if script_interp is not None,  for execute permission.
            If None/not present will raise an error
        post_process can include
            interp -- if not None the name of the interpreter.
            input_file -- name of input file for post-processing. If None will be input.json
                 post-processing info will be written to that file so post processor has access to it
                Will be stored in self._post_process_input
            output_file -- name of output file for post-processing. If None will be sim_obs.json
                This is where simulated observations go (which are then read in).
                Will be stored in self._post_process_output
        post_process will be deepcopyed to self.post_process with script, interp, input_file, output_file removed
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

    def compare_objects(self, other):
        """
        Compare two objects and identify their differences by comparing their attributes.
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
            elif type(vself[k]) != type(vother[k]):
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

    def gen_params(self, parameters: typing.Optional[dict] = None):
        """
        Get iterable  of namelist/vars  to set.
        :param parameters: If none use self.parameters else use this
        Calls self.param_info.gen_parameters using self.parameters.
        then verifies  have iterable of namelist,value pairs.
        .or some models you may want to override this method to deal with
          other ways of setting parameters.
           This implementation deals with namelists and functions that return None,
        :return: a iterable of  namelist, value pairs
        Example: model.gen_param_set()
        """
        if parameters is None:
            parameters = copy.deepcopy(self.parameters)
        else:
            self.update_history(f"Setting parameters using parameters {parameters} rather than self.parameters")

        param_set_info = self.param_info.gen_parameters(self, **parameters)
        # expecting a list of namelist/something, value pairs
        for (nl, value) in param_set_info:
            if not isinstance(nl, namelist_var):  # can only deal with namelists
                raise NotImplementedError(f"Implement code for {type(nl)}  or override gen_params")
        return param_set_info  # checked we have only namelists.

    def set_params(self, parameters: typing.Optional[dict] = None):

        """
        Set parameters by patching namelist. Override if you want more than namelists.
        :param self: Model instance
        :param parameters -- dict(or None) of parameters to use.
        :return: Nothing
        """
        nl = self.gen_params(parameters=parameters)
        namelist_var.nl_modify(nl, dirpath=self.model_dir)  # patch the namelists.

    def changed_nl(self):
        """
        Return namelists that have been changed. Note does not actually change anything about the model.
        :return: set of namelists indexed by file.
        """
        nl = self.gen_params()
        modNL = namelist_var.modify_namelists(nl, dirpath=self.model_dir, update=True, clean=True)  # purge the cache
        return modNL

    def create_model(self):
        """
        Create a new model by copying reference. Overwrite (and call superclass) for your own model.
        For example if you want to modify your reference model.
        :return:nothing.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)  # create the directory if needed.
        shutil.copytree(self.reference, self.model_dir, symlinks=True, dirs_exist_ok=True)  # copy from reference.

    def set_status(self, new_status: str, check_existing: bool = True) -> None:
        """
        Set the status of Model.
        Checks that new status is allowed and consistent with current status
        Then sets the status. Writes out the model configuration.
        See self.status_info for allowed status names and what is expected.
        :param new_status: new status for model
        :param check_existing: Check current status is as expected.
        """

        if new_status not in self.allowed_status:
            raise ValueError(f"Status {new_status} should be one of " + " ".join(self.allowed_status))
        if new_status == 'CREATED':
            raise ValueError(f"Do not set status to CREATED")
        expected_status = self.status_info[new_status]
        if check_existing and (self.status not in expected_status):
            raise ValueError(
                f"Expected current status {self.status}  to be one of " + " ".join(
                    expected_status) + f" as changing to {new_status}")
        logging.debug(f"Changing status from {self.status} to {new_status}")
        self.update_history(f"Status set to {new_status} in {self.model_dir}")
        self.status = new_status
        self.dump_model()  # write to disk

    def instantiate(self) -> None:
        """
        Run create_model and set_params, update status.
        Will verify that (if defined) post-processing script exists failing if not
        Also, should make any changes needed to those files.
        :return:
        """
        self.create_model()  # create model
        self.modify_model()  # do any modifications to model needed before setting params.
        self.set_params()  # set the params
        # set permissions to rxw,rx,rx for submit and continue script.
        for file in [self.submit_script, self.continue_script]:
            if file is not None:
                (self.model_dir / file).chmod(0o766)  # set permission

        self.set_status('INSTANTIATED')

    def modify_model(self):
        """
        Modify model. This does nothing and is designed to be overwritten in classes that inherit from it.
        Those should call this first as it checks that set_status_script exists and updates history

        :return: None
        """
        if not self.set_status_script.exists():
            raise ValueError(f"Need {self.set_status_script} does not exists")

        self.update_history(f"modifying model")

        return None

    # def setup_model_env(self) -> bool:
    #     """
    #     Method to set up stuff for model_status_script to get model_config
    #     This function does it by setting up env var OPTCLIM_MODEL_PATH and
    #       writing out the json file model_dir/OPTCLIM_MODEL_PATH.json with model_config stored in it.
    #       Uses generic json dump to do so.
    #       Idea is that this gets run just before a model is actually submitted. As model run
    #         is then expected to run the status update script.
    #     both model_config and model_dir should exist. If they do not or are not a file or dir respectively
    #       a value error will be raised.
    #     :param model_config: path to model configuration.
    #     :param model_dir: path to model_dir. Model assumed to run in that
    #     :return: True if successful; False if Not though will actually fail!
    #     """
    #     if not self.model_dir.exists():
    #         raise ValueError(f"Model_dir {self.model_dir} does not exists")
    #     if not self.model_dir.is_dir():
    #         raise ValueError(f"Model_dir {self.model_dir} is not a directory")
    #     if not self.config_path.exists():
    #         raise ValueError(f"Model config {self.config_path} does not exist")
    #     if not self.config_path.is_file():
    #         raise ValueError(f"Model config {self.config_path} is not a file")
    #     config_info = dict(config_path=self.config_path)
    #     outpath = self.model_dir / "OPTCLIM_MODEL_PATH.json"
    #     with open(outpath, 'wt') as fp:
    #         generic_json.dump(config_info, fp)
    #
    #     os.environ['OPTCLIM_MODEL_PATH'] = str(self.config_path)  # set up the environment.
    #     logging.debug(f"Wrote config path to {outpath}. $OPTCLIM_MODEL_PATH = {os.environ['OPTCLIM_MODEL_PATH']}")
    #     return True  # we worked!

    def submit_model(self, run_info: dict,
                     engine: engine.abstractEngine,
                     fake_function: typing.Optional[typing.Callable] = None,
                     outputDir: typing.Optional[pathlib.Path] = None
                     ) -> typing.Optional[str]:
        """
        Submit  a model and its post-processing.
        Post-processing gets submitted first but held.
        Model gets cmd to release_job post-processing included before it gets submitting.

        :param run_info -- dict of information for running. runTime & runCode used here.
        :param engine: -- engine (see engine) for details.
          Contract for this is that it takes in a list (cmd) and transforms it.
           If None then the cmd will simply be ran.
        :param fake_function -- if provided no submission  will be done. Instead, this function will be used to generate fake obs.
          Designed for testing code that runs whole algorithms. Takes one argument -- dict of parameters.
        :param outputDir -- path to where output goes. Used for post-processing.
          If None default from engine.submit will be used
        :return: The jobid of the post-process submission. None if nothing submitted (fake_fn set or continuing)

        Example:
         model.submit_model(config.run_info(),engine=SGE_engine)
        """
        status = 'SUBMITTED'
        # deal with fake_function.
        if fake_function is not None:  # handle fake function
            self.post_process_cmd = None  # no cmd to run as we just run it!
            self.simulated_obs = fake_function(self.parameters).rename(self.name)  # compute the simulated_obs
            if not isinstance(self.simulated_obs, pd.Series):
                raise ValueError(f"{fake_function} did not return pandas series. Returned {self.simulated_obs}")

            # check for nulls
            null = self.simulated_obs.isnull()
            if np.any(null):
                raise ValueError("Fake function produced null values at: " + ", ".join(self.simulated_obs.index[null]))
            self.fake = True  # we are faking it!
            logging.info(f"Using fake functions {fake_function.__name__}")
            logging.info(f"Faking {self.name}")
            # work through rest of order.
            self.set_status(status)
            self.running()  # running stuff
            self.succeeded()  # succeeded stuff
            self.process()  # and process.
            return None  # nothing submitted.
        # real stuff.
        # first sort out the post-processing.
        if not self.continuable():
            if self.post_process_cmd is not None:  # check self.post_process_cmd is None and fail if not!
                raise ValueError(f"Have post_process_cmd {self.post_process_cmd} should be None")
            pp_cmd = [str(self.set_status_script),  str(self.config_path),
                      'PROCESSED']  # post-process cmd.
            run_time = self.post_process.get('runTime', 1800)  # get the runTime.
            run_code = self.post_process.get('runCode', run_info.get('runCode'))
            # and the run_code -- default is value in run_info but use value from post_process if we have it.
            pp_cmd = engine.submit_cmd(pp_cmd, f"PP_{self.name}",
                                       outdir=outputDir,
                                       hold=True,
                                       time=run_time,
                                       run_code=run_code)  # generate the submit cmd.
            # note the post-processing is submitted "held".It needs to be released once the model
            # has actually finished. That could require multiple simulations. So we don't tag it on the model
            # and explicitly release_job it.
            output = self.run_cmd(pp_cmd)  # submit the post-processing job. Note it is held.
            logging.debug(f"post-processing run {pp_cmd} and got {output}")
            pp_jid = engine.job_id(output)  # extract the job-ID.
            release_cmd = engine.release_job(pp_jid)  # and generate the release_job job cmd.
            self.post_process_cmd = release_cmd  # store it! So when model releases the post-processing cmd it will run.
            logging.debug(f"self.post_process_cmd  is {self.post_process_cmd}")
        else:
            pp_jid = None  # no post-processing got submitted.
        # submit the model!
        cmd = self.submit_cmd(run_info, engine)
        # actual running model can update status.
        output = self.run_cmd(cmd)
        logging.debug(f"Model submission: ran {cmd} and got {output}")
        self.submission_count += 1
        self.set_status(status)

        return pp_jid  # return the post-processing jid

    def submit_cmd(self, run_info: dict,
                   engine: engine.abstractEngine) -> typing.List[str]:
        """"
        Generate the submission command. Over-ride this for your own model.
        :param run_info -- run_info block.
        :param engine -- engine info.
        If status is INSTANTIATED or PERTURBED then this runs engine.submit_cmd on  [self.submit_script] and
        if CONTINUE runs on  [self.continue_script]
        output should go to model_dir/'model_output' which will be created if it does not exist.
        """
        if self.status in ['INSTANTIATED', 'PERTURBED']:
            script = self.submit_script
        elif self.status == 'CONTINUE':
            script = self.continue_script
        else:
            raise ValueError(f"Status {self.status} not expected ")
        script = self.model_dir / script
        runCode = run_info.get('runCode')
        runTime = run_info.get('runTime',2000) # 2000 seconds as default.
        # need to (potentially) modify model script so runTime and runCode are set.
        # but in this case just use the submit.
        outdir = self.model_dir / 'model_output'
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = engine.submit_cmd([str(script)], f"{self.name}{self.run_count:05d}", outdir,
                                run_code=runCode, time=runTime)

        return cmd

    def running(self):
        """
        Set status to running & increment run_count
        :return:
        """
        self.run_count += 1
        self.set_status('RUNNING')

    def set_failed(self):
        """
        Set status to failed.
        :return:
        """
        self.set_status('FAILED')

    def perturb(self, parameters: typing.Optional[dict] = None):
        """
        Set status to PERTURBED. Will need to be continued or submitted which requires submission information.
        :param parameters: dict of parameters & values to use to generate random perturbation.
        This will update existing parameters.
        Will aso increase perturb_count by 1 so algorithm can adjust if multiple perturbations done,
        :return:None

        This likely needs overwriting for specific models as parameters will be set.
        For implementation in actual model class have something like:
        def perturb(self):

            parameters=dict(rand_init=random())
            super().perturb(parameters)
        """
        if parameters is None:
            parameters = {}
            logging.debug("Setting pertrub parameters to empty dict")
        self.parameters.update(parameters)  # update parameters
        self.set_params()  # set parameter values
        self.update_history(f'Perturbed using {parameters}')  # so at least we can find out what was done
        self.perturb_count += 1
        self.set_status('PERTURBED')
        logging.debug(f"set parameters to {parameters}")

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
        Run the post-processing job by running self.post_process_cmd
        Then set status to SUCCEEDED
        :return: output from running self.post_process_cmd
        """

        status = 'SUCCEEDED'

        if self.post_process_cmd is not None:
            # release_job the post-processing job. But really system specific.
            # just run the post-processing cmd as a sub-shell.
            # That hopefully, eventually, does model.post_process()!
            # On eddie this will be something like qrls NNNNNNNN.x
            result = self.run_cmd(self.post_process_cmd, cwd=self.model_dir)
            # will raise an error if it failed.
            logging.info(f"Ran post-processing cmd {self.post_process_cmd}")
            output = result
        else:
            logging.info("No post-processing processing cmd")
            output = None  # no output.

        self.set_status(status)
        return output

    def process(self):
        """
        Run the post-processing, store output and set status to COMPLETED.
        "Contract" for a post-processing script
        1) takes a json file as input (arg#1) and puts output in file (arg#2).
        2) It is being ran in the model_directory.
         arg#1 needs json.load to read the json file. Code should expect a dict and use the postProcess entry.
             This allows it ot read in and act on a StudyConfig file.
         arg#2 can be .json or .csv or .nc
        :return: output from post-processing.
        """
        status = 'PROCESSED'
        if self.fake:  # faking?
            self.set_status(status)  # just update the status
            return

        input_file = self.model_dir / self._post_process_input  # generate json file to hold post process info
        logging.debug(f"Dumping post_process to {input_file}")
        output = dict(postProcess=self.post_process)  # wrap post process in dict
        with open(input_file, 'w') as fp:
            json.dump(output, fp)
        # dump the post-processing dict for the post-processing to  pick up.

        post_process_output = self.model_dir / self._post_process_output
        result = self.run_cmd(self.post_process_cmd_script, cwd=self.model_dir)  #

        # get in the simulated obs which also sets them 
        self.read_simulated_obs(post_process_output)
        self.set_status(status)
        return result

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
            logging.debug("netcdf file got " + " ".join(obs.keys()))

        elif fileType == '.json':  # json file
            with open(post_process_file, 'r') as fp:
                obs = json.load(fp)
            logging.debug("json file got " + " ".join(obs.keys()))

        elif fileType == '.csv':  # data is a csv file.
            obsdf = pd.read_csv(post_process_file, header=None, index_col=False)
            obs = obsdf.to_dict()
            logging.debug("csv file got " + " ".join(obs.keys()))

        else:  # don't know what to do. So raise an error
            raise NotImplementedError(f"Do not recognize {fileType}")

        logging.info(f"Read {fileType} data from {post_process_file}")
        obs = pd.Series(obs).rename(self.name)
        self.simulated_obs = obs

        # check for nulls
        null = obs.isnull()
        if np.any(null):
            raise ValueError("Obs contains null values at: " + ", ".join(obs.index[null]))

        return obs  # return the obs.

    def read_values(self, parameters: str | typing.List[str] | None, fail: bool = True) -> dict:
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
            parameters = self.param_info.known_parameters()  # get all parameters

        for parameter in set(parameters):  # set means we iterate over unique parameters
            try:
                result[parameter] = self.param_info.read_param(self, parameter)
            except (KeyError, FileNotFoundError):
                if fail:
                    raise
                logging.warning(f"Parameter {parameter} not found in {self.name}")
                result[parameter] = None

        return result

    def is_submittable(self) -> bool:
        """
        Return True if model is submittable -- which means its status is CONTINUE or INSTANTIATED or PERTURBED
        :return:
        """
        return self.status in ["CONTINUE", "INSTANTIATED", "PERTURBED"]

    def failed(self) -> bool:
        """
        Return True if model has failed
        :return:
        """
        return self.status in ['FAILED']

    def continuable(self) -> bool:
        """

        :return: Return True if model is continuable.
        """

        return self.status in ['CONTINUE']

    def delete(self):
        """
        Delete all on disk stuff. Do by deleting all files in self.model_dir and self.config_path
        :return: None
        """
        shutil.rmtree(self.model_dir, ignore_errors=True)
        logging.info(f"Deleted everything in {self.model_dir}")
        self.config_path.unlink(missing_ok=True)
        return True

    def archive(self, archive_path):
        """
        Archive parts of model_dir & config_path to archive_path. This version will raise NotImplementedError as
          needs to be specialised for individual models
        :param archive_path:
        :return:None
        """
        raise NotImplementedError("Implement archive for your model")

    @register_param("ensembleMember")
    def ens_member(self, ensMember: typing.Optional[int]) -> None:
        """
        Do nothing as perturbing initial conditions is model specific. But needed
         for test cases.
        :param ensMember: ensemble member. The ensemble member wanted.
        :return: None (for now) as nothing done.
        """

        inverse = (ensMember is None)
        if inverse:
            logging.warning("Can not invert ensMember")
            return None

        logging.warning(f"Nothing set for {ensMember}. Override in your own model")

        return None


Model.register_class(Model)  # register ourselves!

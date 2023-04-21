"""
Class for abstract Model and namelists.
"""
from __future__ import annotations

import datetime
import logging
import os
import pathlib
import shutil
import subprocess
import typing
import copy
import time
import xarray
import pandas as pd


import generic_json
from Models.model_base import model_base
from Models.namelist_var import namelist_var
from Models.param_info import param_info



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
     See register_param function above which tags functions while the init_subclass uses thoses tags to put that function in
     the param_info attribute.
    This class inherits from model base which through its __init_subclass__  whicch sets up things for dumping and loading instances
    to disk as a json file.
    """
    class_registry = dict()  # where class information for model_init is used.

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
        for name, member in cls.__dict__.items():
            # Loop through all members of the subclass and populate param_info
            # accordingly. These should all be functions.
            logging.debug(f"Processing member {name}")
            if getattr(member, '_is_param', False):
                param_name = getattr(member, '_name')
                my_param_info.register(param_name, member)
                logging.info(f"Registered param {param_name}")

        cls.param_info = copy.deepcopy(my_param_info)
        ModelBaseClass.register_class(cls)
        # register the class for subsequent creation. This allows model_init to work.
        logging.info(f"Registered {cls.__name__}")

    @classmethod
    def register_class(cls, newcls: typing.Class):
        ModelBaseClass.class_registry[newcls.__name__] = newcls

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
            newcls = Model.class_registry[class_name]
        except KeyError:
            raise ValueError(f"Failed to find {class_name}. Allowed classes are " + " ".join(cls.class_registry.keys()))
        result = newcls(*args, **kwargs)
        logging.debug(f"Created {class_name}")
        return result

    @classmethod
    def load_model(cls, model_path: pathlib.Path):
        """
        Load a configuration
        :param model_dir: directory where the configuration (model_config_name) is stored
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


class Model(ModelBaseClass):
    """
    Abstract model class. Any class that inherits from this will have name lookup.
    Also provides top-level methods and class methods.
    Model.load_model() will load a model from disk. An object of the appropriate class will be returned as long as that
    class inherits from Model.

    """
    # logging.getLogger(__name__) # setup logger for the class.
    model_config_name = 'Model_config.json'  # name of the config file.
    post_proccess_json = "post_process.json"  # where post-process info gets written
    status_info = dict(CREATED=None,
                       INSTANTIATED=["CREATED"],  # Instantiate a model requires it to have been created
                       SUBMITTED=['INSTANTIATED', 'PERTURBED'],
                       # Submitting needs it to have been instantiated or perturbed
                       RUNNING=["SUBMITTED"],  # running needed it should have been submitted
                       FAILED=["RUNNING"],  # Failed means it should have been running
                       PERTURBED=["FAILED"],  # Allowed to perturb a model after it failed.
                       SUCCEEDED=["RUNNING"],  # SUCCEEDED means it should have been running
                       PROCESSED=['SUCCEEDED'])  # Processed means it should have succeeded.
    # Q Perturbed comes in two flavours. Pertrub and continue or perturb and restart. How to handle that?
    allowed_status = set(status_info.keys())

    # methods now.
    def __init__(self, *, name: typing.Optional[str] = None,
                 reference: typing.Optional[pathlib.Path] = pathlib.Path.cwd(),
                 model_dir: typing.Optional[pathlib.Path] = pathlib.Path.cwd(),
                 config_path: typing.Optional[pathlib.Path] = None,
                 status: typing.Optional[str] = "CREATED",
                 fake: bool = False,
                 perturb_count: int = 0,
                 parameters: dict | None = None,
                 post_process: dict | None = None,
                 history: dict | None = None,
                 output: dict | None = None,
                 post_process_cmd: str | None = None,
                 simulated_obs: pd.Series | None = None):
        """
        Initialise the Model class.

        All are keyword arguments
        :param name -- name of model
        :param reference -- reference directory. Should be a pathlib.Path
        :param model_dir --- where model will be created and any files written.
             Should be a pathlib.Path. Will, if needed, be created.
        :param config_dir: Where configuration will be created.
               If not defined will be model_dir/(self.name+".mcfg")
        :param status -- model status. Default = "CREATED"

        :param post_process -- dict for post-processing. If available, will be made available to post-processing code.
            Must contain at least the following keys:
                script -- full path to the script that will be ran
                outputPath -- local path to where the output of the script will be written.
                if contains script_interp -- this is the interpreter used for the script.
                    Handy for windows but better to use #! in your script!
          This should be constructed in the  setup using the StudyConfig. Goal would be to provide some useful information
          identify which parameters are variable, parameter ranges for those. Expected obs names,
          Suggests a method for StudyConfig for this.
          Which adds some information to post_process block and provides it as a dict.

        :param history -- dict of history. Keys are times as ISO strings and values are arbitrary.
        :param output -- dict of output from running commands indexed by status.
        :param post_process_cmd -- command to run the post-processing. Only used when status gets set to COMPLETED.

        All are stored as attributes and are publicly available though user should be careful if they modify them.
        public attributes:
        model_dir -- directory where model information is stored
        reference --  where the reference configuration came from.
        config_path -- where the configuration is to be written to (or was read from).
        name -- name of the model

        status -- status of the model
        history -- model history
        post_process -- post-processing information.
        output -- output from sub shell commands indexed by status.
        post_process_script -- Script/exe to be run for post-processing. If None no post-processing will be done.
        post_process_script_interp -- interpreter for the post-processing script. If None not used.
        post_process_cmd -- cmd to run the post-processing. Depends on your submission/job management system

        fake -- If True model is faked.
        perturb_count -- no of times perturbation has been done.
        parameters -- dict of parameters/values

        Things that should be changed in inherited classes (and are not written out)
            submit_script -- name of script to submit for a new simulation
            continue_script -- name of script to submit for a continuing simulation

        """
        # set up default values.
        self.fake = fake  # did we fake the model? Changed at submission.
        self.perturb_count = perturb_count  # how many times have we perturbed the model?
        if output is None:
            output = {}
        else:
            if not isinstance(output,dict):
                raise TypeError(f"Output should be a dict not type {type(output)}")
            output = copy.deepcopy(output)

        if history is None:
            history = {}
        else:
            history = copy.deepcopy(history)

        if parameters is None:
            parameters = {}
        else:
            parameters = copy.deepcopy(parameters)

        if post_process is None:
            post_process = {}
        else:
            post_process = copy.deepcopy(post_process) # do not modify input.
        if name is not None:
            self.name = name
        else:
            self.name = model_dir.name

        if config_path is None:
            config_path = model_dir / (self.name + '.mcfg')

        self.config_path = config_path

        self.reference = reference
        self.model_dir = model_dir
        if status not in self.allowed_status:
            raise ValueError(f"Status {status} not in " + " ".join(self.allowed_status))

        self.parameters = parameters
        self.post_process = post_process
        self.history = history
        self.output = output
        self.post_process_cmd = post_process_cmd
        # attributes that get defined based on inputs or are just fixed.

        self.submit_script = 'submit.sh'
        self.continue_script = 'continue.sh'
        if post_process is not None:
            self.post_process_script = post_process.pop('script', None)
            self.post_process_script_interp = post_process.pop('script_interp', None)
            post_process_output = post_process.pop('outputPath', None)
            if post_process_output is not None:
                self.post_process_output = post_process_output
            else:
                self.post_process_output = None

        # code below tests that post_process_script exists and is executable.
        if self.post_process_script is not None:
            # check it is read/executable by us raising errors if not.
            self.post_process_script = self.expand(self.post_process_script)
            ok = self.post_process_script.is_file()
            if not ok:
                raise ValueError(f"Script file {self.post_process_script} does not exist")
            ok = os.access(self.post_process_script, os.R_OK | os.X_OK)
            if not ok:
                raise ValueError(f"Script file {self.post_process_script} does not have read and execute permission")
        self.status = status
        if self.status == 'CREATED':  # creating model for the first time
            self.update_history("CREATING model")
        self.simulated_obs = simulated_obs

    def __repr__(self):
        """
        String that represents model. Shows type,name,status, no of parameters and when history last updated
        :return: str
        """
        last_hist = list(self.history.keys())[-1]
        s = f"Type: {self.class_name()} Name: {self.name}" \
            f" Status: {self.status} Nparams: {len(self.parameters)} Last Modified:{last_hist}"
        return s

    def dump_model(self):
        """
        dump a model configuration to self.model_dir/model_config_name
        :return: whatever dump does
        """

        return self.dump(self.config_path)  # call the  *dump* method.

    def gen_params(self, parameters: dict | typing.NoneType = None):
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
            parameters = self.parameters
        else:
            self.update_history(f"Setting parameters using parameters {parameters} rather than self.parameters")
        param_set_info = self.param_info.gen_parameters(self, **parameters)
        # expecting a list of namelist/something, value pairs
        for (nl, value) in param_set_info:
            if not isinstance(nl, namelist_var):  # can only deal with namelists
                raise NotImplementedError(f"Implement code for {type(nl)}  or override gen_params")
        return param_set_info  # checked we have only namelists.

    def set_params(self, parameters: dict | None = None):

        """
        Set parameters by patching namelist. Override if you want more than namelists.
        :param self: Model instance
        :param parameters -- dict(or None) of parameters to use.
        :return: Nothing
        """

        nl = self.gen_params(parameters=parameters)
        namelist_var.nl_patch(nl, dirpath=self.model_dir)  # patch the namelists.

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
            raise ValueError(f"Expected current status ({self.status}) to be " + " ".join(expected_status))
        logging.debug(f"Changing status from {self.status} to {new_status}")
        self.update_history(f"Status set to {new_status} in {self.model_dir}")
        self.status = new_status
        self.dump_model()  # write to disk

    def update_history(self, message: str):
        """
        Update the history directory. Key will be *now* using UTC.
        Routine updates existing values so that multiple updates in a short time will preserve history.
        Short time defined as less than precision of str(now))
        :param message:message text to be stored.
        :return:
        """
        dtkey = str(datetime.datetime.now(tz=datetime.timezone.utc))  # datetime for now as string
        h = self.history.get(dtkey, [])  # get any existing history for this time
        h += [message]  # add on the message
        self.history[dtkey] = h  # store it back again.
        logging.debug(f"Updated history at {dtkey} ")

    def instantiate(self) -> None:
        """
        Run create_model and set_params, update status.
        Will verify that (if defined) post-processing script exists failing if not
        Also, should make any changes needed to those files.
        :return:
        """
        # test for existence of post-processing script
        if (self.post_process_script is not None) and (not self.post_process_script.exists()):  # check script exists.
            raise ValueError(f"{self.post_process_script} does not exist")

        self.create_model()  # create model
        self.set_params()  # set the params
        self.set_status('INSTANTIATED')

    def run_model(self, submit_cmd: typing.Callable, post_process_cmd: str | None = None,
                  new: bool = True, fake_function: typing.Callable | None = None) -> str:
        """
        run  a model.
        :param submit_cmd: A function that submits a script.
          Contract for this is that it takes in the path to a script and does whatever magic is needed to
          submit it. Any errors are simply raised. Output from submission cmd is returned.
        :param new-- If True a new simulation will be started; If False the simulation will be continued.
        :param post_process_cmd: Command to run post-processing. Generated by the submission system. None if not wanted.
           This command will be run as part of the  succeeded method.
        :return: The output from the submit_cmd.
        Example:
         model.run_model(SGE_submit,post_process_cmd = 'ssh login01.eddie.ecdf.ed.ac.uk qrls 890564.2',new=False)
        """

        if new:
            script = self.submit_script
        else:
            script = self.continue_script

        status = 'SUBMITTED'

        if fake_function is not None:  # handle fake function
            def fake_submit_cmd(x):  # generate a submit_cmd that does nothing
                return f"Faking submission of {x}"

            submit_cmd = fake_submit_cmd  # overwrite submit_cmd
            self.post_process_cmd = None  # no cmd to run as we just run it!
            self.simulated_obs = fake_function().rename(self.name)  # compute the simulated_obs
            if not isinstance(self.simulated_obs, pd.Series):
                raise ValueError(f"{fake_function} did not return pandas series. Returned {self.simulated_obs}")
            self.fake = True  # we are faking it!
            logging.info(f"Using fake functions {fake_function.__name__}")
        else:
            self.post_process_cmd = post_process_cmd  # the post-processing cmd needed to run post-processing.
            logging.debug(f"Set post_process_cmd  to {post_process_cmd}")

        output = submit_cmd(script)  # submit which ever script is meant to get ran.
        logging.debug(f"Ran submit_cmd: {submit_cmd.__qualname__}")
        try:
            soutput = self.output[status]
        except KeyError:
            soutput = []
        soutput.append(output)
        self.output[status]=soutput
        self.set_status(status)

        if self.fake:
            logging.info(f"Faking {self.name}")
            # work through rest of order.
            delay = 0.01  # delay 1/100 of second between updates
            time.sleep(delay)
            self.running()  # running stuff
            time.sleep(delay)
            self.succeeded()  # succeeded stuff
            time.sleep(delay)
            self.process()  # and process.

        return output

    def running(self):
        """
        Set status to running
        :return:
        """
        self.set_status('RUNNING')

    def failed(self):
        """
        Set status to failed.
        :return:
        """
        self.set_status('FAILED')

    def perturb(self, parameters):
        """
        Set status to PERTURBED. Will need to be continued or submitted which requires submission information.
        :param parameters: dict of parameters & values to use to generate random perturbation.
        This will update existing parameters
        Will aso increase perturb_count by 1 so algorithm can adjust if multiple perturbations done,
        :return:None

        This likely needs overwriting for specific models as parameters will be set.
        For implementation in actual model class have something like:
        def perturb(self):

            parameters=dict(rand_init=random())
            super().perturb(parameters)
        """
        self.parameters.update(parameters)  # update parameters
        self.set_params()  # set parameter values
        self.update_history(f'Perturbed using {parameters}')  # so at least we can find out what was done
        self.perturb_count += 1
        self.set_status('PERTURBED')
        logging.debug(f"set parameters to {parameters}")

    def succeeded(self):
        """
        Run the post-processing job by running self.post_process_cmd
        Then set status to SUCCEEDED
        :return: output from running self.post_process_cmd
        """

        status = 'SUCCEEDED'

        if self.post_process_cmd is not None:
            # release the post-processing job. But really system specific.
            # just run the post-processing cmd as a sub-shell.
            # That hopefully, eventually, does model.post_process()!
            # On eddie this will be something like ssh login01.eddie.ecdf.ed.ac.uk qrls NNNNNNNN.x
            result = subprocess.check_output(self.post_process_cmd, shell=True, cwd=self.model_dir)
            if result.returncode != 0:  # failed
                logging.error(" ".join(result.args) + f" failed with status {result.exitcode}")
                logging.error("StdOut: " + "\n".join(result.stdout))
                if result.stdout is not None:
                    logging.error("StdErr: " + "\n".join(result.stder))
                result.check_return_code()  # aise an error if process failed
            logging.info(f"Ran post-processing cmd {self.post_process_cmd}")
            output = result.stdout
        else:
            logging.info("No post-processing processing cmd")
            output = None  # no output.

        try:
            soutput = self.output[status]
        except KeyError:
            soutput = []
        soutput.append(output)
        self.output[status]=soutput
        self.set_status(status)
        return output

    def process(self):
        """
        Run the post-processing, store output and set status to COMPLETED.
        "Contract" for a post-processing script
        1) takes a json file as input (arg#1) and puts output in file (arg#2).
        2) It is being ran in the model_directory.
         arg#1 needs generic_json.load to read the json file. This handles the necessary decoding.
         arg#2 can be .json or .csv or .nc
        :return: output from post-processing.
        """
        status = 'PROCESSED'
        if self.fake:  # faking?
            self.set_status(status)  # just update the status
            return

        input_file = self.model_dir / self.post_proccess_json  # generate json file to hold post process info
        with open(input_file, 'w') as fp:
            generic_json.dump(self.post_process, fp)
        # dump the post-processing dict for the post-processing to  pick up.

        post_process_output = self.model_dir / self.post_process_output
        cmd = [str(self.post_process_script), str(input_file), str(post_process_output)]
        if self.post_process_script_interp is not None:  # have to specify interpreter for script
            cmd.insert(0, self.post_process_script_interp)
        result = subprocess.run(cmd, cwd=self.model_dir, shell=True, text=True, capture_output=True)  # run in model_dir
        if result.returncode != 0:  # failed
            logging.error(" ".join(result.args) + f" failed with status {result.returncode}")
            if result.stdout is not None:
                logging.error("StdOut: " + result.stdout)
            if result.stderr is not None:
                logging.error("StdErr: " + result.stderr)
            result.check_returncode()  # raise an error if process failed

        try:
            soutput = self.output[status]
        except KeyError:
            soutput = []
        soutput.append(result.stdout)
        self.output[status]=soutput
        # get in the simulated obs which also sets them 
        self.read_simulated_obs(post_process_output)
        self.set_status(status)
        return result

    def read_simulated_obs(self, post_process_file: pathlib.Path):
        """
        Read the post processed data.
         This default implementation reads simulated obs from netcdf, json or csv data and
         stores it in the Model as a pandas series
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
                obs = generic_json.load(fp)
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

        return obs  # return the obs.

    def read_values(self, parameters: str | typing.List[str]|None,fail: bool =True) -> dict:
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
            except (KeyError,FileNotFoundError):
                if fail:
                    raise
                logging.warning(f"Parameter {parameter} not found in {self.name}")
                result[parameter] = None

        return result


Model.register_class(Model)  # register ourselves!


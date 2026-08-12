"""
Support for handling model submission.
Porting hints:
1) Work out how, on your cluster, submission is done and modify define_submission.
2) See eddie_ssh for an example of a submit_cmd (which you might need if your workers cannot submit directly)
3) System assumes your model is setup to run on your cluster using a script which is setup for whatever Q system
   your computer uses.

"""
from __future__ import annotations

import copy

import logging
import pathlib

import string
import sys
import tempfile
import typing
import shutil
import importlib
import tarfile


from typing import Optional, List, Callable, Mapping

import numpy as np
import pandas as pd

import engine
import generic_json
from Model import Model
from Models import Model
from model_base import model_base, journal
from Study import Study
from StudyConfig import dictFile
import genericLib
# check we are version 3.8 or above.

if (sys.version_info.major < 3) or (sys.version_info.major == 3 and sys.version_info.minor < 10):
    raise Exception("Only works at 3.10+ ")

__version__ = '0.95'

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")


class SubmitStudy(Study, model_base, journal):
    # typing information for class attributes
    refDir: pathlib.PurePath
    model_name: str
    module_name: typing.Optional[str]
    run_info: dict
    engine: engine.abstractEngine
    config_path: pathlib.Path
    name_values: typing.Optional[list[int]]
    iter_keys: dict
    next_iter_cmd: typing.Optional[list[str|pathlib.PurePath]]
    next_iter_jids: list[str]
    next_command:typing.Optional[typing.Literal['stop']] # next command to run. Only None or stop are allowed.

    """
     provides methods to support working out which models need to be submitted. Creates new models and submits them.
    If you want to view a study just use the Study class. 
    Attributes, beyond model_base, Study & journal ones, are:
        refDir -- path for reference directory. A pure path is allowed but most uses will crash.
        model_name -- name of the model being used
        module_name -- name of the module being used. 
        run_info -- information for submitting runs.
        engine -- functions to handle different job submission engines.
        config_path -- path to where config is stored.A pure path is allowed but most uses will crash.
        name_values -- used to generate name. Set to None to reset counter.
        iter_keys -- dict indexed by key with iteration count.
        next_iter_cmd -- the command to run the next iteration.
        next_iter_jids -- the jobs ids of all submitted next_iter_cmd jobs
    """

    """
    Issues: 1) loading should change the config path or make it an absolute paths.
    """

    fn_type = Callable[[Mapping], pd.Series]  # type hint for fakeFn
    from StudyConfig import  OptClimConfigVn3
    def __init__(self,
                 config: Optional[OptClimConfigVn3],
                 name: Optional[str] = None,
                 rootDir: Optional[pathlib.Path] = None,
                 refDir: Optional[pathlib.Path] = None,
                 models: Optional[List[Model]] = None,
                 model_name: Optional[str] = None,
                 config_path: Optional[pathlib.Path] = None,
                 next_iter_cmd: Optional[list[str|pathlib.Path]] = None
                 ):
        """
        Create ModelSubmit instance
        :param config: configuration information
        :param name: name of the study. If None name of config is used.
        :param rootDir : root dir where new directories and configuration files are to be created.
          If None will be current dir/config.name().
        :param refDir: Directory where a reference model is. If None, then config.referenceConfig() will be used.
        :param model_name: Name of a model type to create. If None value in config is used
        :param models -- list of models.
        :param config_path -- where configuration should be stored. If None default is root_dir/name
        :param next_iter_cmd -- command to run next iteration.
        :return: instance SubmitStudy with the following public attributes :
        """
        super().__init__(config, name=name, models=models, rootDir=rootDir)
        #self.rootDir.mkdir(parents=True, exist_ok=True)  # create it if need be.
        # no need to create rootDir as could be updated and when files are created mkdir happens then    
        if refDir is None:
            refDir = self.expand(str(config.referenceConfig()))
        self.refDir = refDir

        if model_name is not None:  # This is fixed. Even if configuration changed the model_name is fixed.
            self.model_name = model_name
        else:
            self.model_name = config.model_name()

        self.module_name = None
        # see if we have model_name in the list of known models. If we don't then try and load from module
        # TODO: make this more robust so that it can handle models that are not in the list of known models and uses syntax Module.class
        if self.model_name not in Model.known_models():
            self.module_name = config.module_name(model_name=self.model_name)
            my_logger.debug(f"Loading {self.module_name}")
            importlib.import_module(self.module_name)  # and load the module.
        else:
            my_logger.debug(f"Already have {self.model_name} so not loading module")

        self.run_info = copy.deepcopy(config.run_info())  # copy run_info as modifying it.
        # Set up submit engine for this node.
        eng_name = self.run_info.pop('submit_engine',None)
        ssh_node = self.run_info.pop('ssh_node', None)
        if eng_name is None:
            eng  = engine.abstractEngine.guess_engine(ssh_node=ssh_node)
        else:
            eng = engine.abstractEngine.create_engine(eng_name, ssh_node=ssh_node)

        self.engine = eng

        # engine & submit for this computer.

        if config_path is None:
            config_path = self.rootDir / (self.name + '.scfg')
        self.config_path = config_path

        self.name_values = None  # init the counters for names.
        self.update_history(None)  # init history.
        self.store_output(None, None)  # init store output

        self.iter_keys = dict()  # key iteration pairs.
        self.next_iter_cmd = next_iter_cmd
        self.next_iter_jids = []  # no next jobs (yet)
        self.next_command  = None

    def update_config(self, config: "OptClimConfigVn3"):
        """
        Partially set up self with the configuration. This allows updating following a change to the configuration.
          Sets up run_info in addition to whatever the superclass method does.
        To update from configuration simply do self.update_config(config).
        :param config: Configuration to be used.
        :return: nada
        """
        my_logger.debug("Setting configuration")
        super().update_config(config)  # call the superclass

        self.run_info = copy.deepcopy(config.run_info())  # copy run_info as modifying it.
        my_logger.debug(f"Set run_info to {self.run_info}")
        self.update_history(f"Updated configuration from {config.fileName()}") # fix history

    def lock(self,timeout:float=0.0,poll_interval:typing.Optional[float] = None):
        """
        Lock the configuration file using genericLib.ContextFileLock.
        :param timeout -- time in seconds to timeout -- see ContextFileLock
        :param poll_interval -- poll_interval in seconds -- see ContextFileLock

        :return: context file object

        Example usage is:
        with SubmitStudy.lock(timeout=10) as lock:
          do stuff with lock
        """
        return genericLib.ContextFileLock(self.config_path,timeout=timeout,poll_interval=poll_interval)

    def __repr__(self):
        """
        String that represents a SubmitStudy. Calls superclass method and adds on info about history
        :return: string
        """
        last_hist_key = self.last_history_key()
        if last_hist_key:
            last_hist = "Last changed at " + last_hist_key
        else:
            last_hist = "No Hist"
        s = super().__repr__() + " " + last_hist

        return s

    def compute_simulated_observations(self,param:dict,
                                       use_cache:bool = True) -> \
            tuple[list[Model|None],typing.Optional[pd.Series]]:
        """"
        Compute simulated obs for a param. If param not in dict then return None.
        :param param: dict of params to compute simulated observations for
        :param use_cache -- whether to use cached simulated observations or not.
        :return: one element list of Model and simulated observations as pd.Series. All obs are returned.
          TODO: refactor this approach so that model is not returned as can just be retrieved using param.
        """
        key = self.key(param) # problem is (I think) that self.key(param) is not the same as self.key_for_model(param)
        # Key failing to match key_for_model as do not have reference_name
        #raise NotImplementedError("Key gen going wrong. Likely because missing reference_name. FIXME please")
        if key not in self.model_index:
            model = self.create_model(param,dump=False ) # create the model.
            return [model],None
        model = self.model_index[key]
        sim_obs = model.compute_simulated_observations(use_cache=use_cache)
        return [model],sim_obs

    def create_model(self, params: dict,
                     dump: bool = True) -> typing.Optional[Model]:
        """
        Create a model, update list of created models and index of models.
        If self.next_command is 'stop' immediately returns None.
        name is generated using self.gen_name() and will be checked to see if it already exists.
        If it does then a new name will be generated (and so on).

        :param   params: dictionary of parameters to create the model.
         The following parameters are special and handled differently:
           * reference -- the reference directory. If not there (or None) then self.refDir is used.
           * model_name -- the model type to be created. If not in params then  self.model_name is used.
           These support more complex algorithms where multiple models need to be ran.
        These will be augmented by fixedParams
        If you need functionality beyond this you may want to inherit from SubmitStudy and
          override create_model to meet your needs
        :param dump: If True dump  self (using self.dump_config method)
        :param reference_name: Name of the reference config. If None the default Model behaviour is used.
        :return: Model created.

        Will raise ValueError if model_dir or config path already exist.
        """
        if self.next_command == 'stop':
            my_logger.warning("Stopping. Returning None")
            return None
        existing_names = [model.name for model in self.model_index.values() ] # list of existing model names
        while True: # loop until we find a name that does not exist.
            name = self.gen_name()
            if name not in existing_names:
                break
            else:
                my_logger.debug(f"Name {name} already exists. Generating new name")
        model_dir = self.rootDir / name
        if model_dir.exists():
            raise ValueError(f"model_dir {model_dir} already exists")
        config_path = model_dir / (name + '.mcfg')  # create model config in model dir
        if config_path.exists():
            raise ValueError(f"config_path {config_path} already exists")
        param_dir = copy.deepcopy(params)
        reference = self.expand(param_dir.pop('reference', str(self.refDir)))
        model_name = param_dir.pop('model_name', self.model_name)
        reference_name = param_dir.pop('reference_name', None)
        post_process = self.config.getv('postProcess')
        run_info = self.config.run_info()
        model = Model.model_init(model_name, name=name,
                                 reference=reference,
                                 reference_name=reference_name,
                                 model_dir=model_dir,
                                 config_path=config_path,
                                 parameters=param_dir,
                                 post_process=post_process,
                                 study_config_path=self.config.fileName(),
                                 engine=self.engine,
                                 run_info=run_info
                                 )
        key = self.key_for_model(model)
        if key in self.model_index:
            raise ValueError(f"Already got key for {key} and parameters {model.parameters}")
        self.model_index[key] = model
        self.update_history(f"Created Model {model}")
        if dump:
            self.dump_config()  # and configuration
        my_logger.debug(f"Created model {model} with parameters {model.parameters}")
        return model

    def update_iter(self, models: List[Model]) -> int:
        """
        Update iteration information.
        :param models: models to add to iteration info.
        :return: current value of iteration count
        """
        existing_counts = list(self.iter_keys.values())
        if len(existing_counts) == 0:
            iter_count = 0
        else:
            iter_count = int(np.max(existing_counts)) + 1  # need as a native python int rather than numpy.

        for m in models:
            key = self.key_for_model(m)
            self.iter_keys[key] = iter_count

        return iter_count

    def iterations(self) -> List[List[Model]]:
        """

        :return: list of lists of  models. Outer list is order by iteration. Inner list is models on each iteration.
            For example result[0] is list of all models from iteration 0.
        """
        # TODO: Work out how a version of this can go into Study.
        # Only way I can currently see of doing this is by converting a SubmitStudy object

        if len(self.iter_keys) == 0:
            return [[]] # return empty list
        iter_keys= list(self.iter_keys.values())
        iter_count = np.max(iter_keys) + 1
        result = [None] * iter_count  # initialize list to iter_count Nones
        # The obvious result = [[]]*iter_count does not work....
        for key, iterc in self.iter_keys.items():
            if result[iterc] is None:  # None make it an empty list
                result[iterc] = []
            result[iterc].append(self.model_index[key])
        return result

    def dump_config(self, dump_models: bool = False):
        """
        Dump the configuration to config_path.
        Unless dump_models is True  models are not dumped. This done to make code run faster as model.set_status(XX) saves the model.
        :param dump_models: If True dump all models
        :return: Nothing
        """
        self.dump(self.config_path)
        if dump_models:
            for model in self.model_index.values():
                model.dump_model()

    @classmethod
    def load(cls, config_path: typing.Union[pathlib.Path, str],
                         error:generic_json.type_error='error',
                         Study: bool = False) -> typing.Union[Study, SubmitStudy]:
        """
        Load a SubmitStudy (or anything that inherits from it) from a file.
        The object will have its config_path replaced by config_path passed in.
        :param config_path: path to configuration to load
        :param error -- how to handle errors when loading. See generic_json.load for details.
        :param Study: If True return a Study object. These are read-only (unless you modify by hand the attributes)
        :return: object
        """
        config_path = cls.expand(config_path)
        # convert str to path and or expand user or env vars.

        obj:SubmitStudy = super().load(config_path,check_types=[cls],error=error) # call the super class load.

        obj.config_path=config_path # modify config path

        if not (isinstance(obj.rootDir,pathlib.Path) and obj.rootDir.exists() and config_path.parent.samefile(obj.rootDir)):
            my_logger.info(f"Modifying config rootDir from  {obj.rootDir} to {config_path.parent}")
            obj.config_path = config_path
            obj.rootDir= config_path.parent
            obj.update_history(f"Modified config path from  {obj.rootDir} to {config_path.parent}")

        if Study:  # convert to a study
            obj = obj.to_study()

        return obj

    def instantiate(self):
        """
        Instantiate all created models. And update_iter so we can see what was done.
        :return: True if all were instantiated. False otherwise
        """

        models = [model for model in self.model_index.values() if model.status == 'CREATED']
        for model in models:
            model.instantiate()  # model state will be written out.
        iter_count = self.update_iter(models)  # update iteration info
        self.update_history(f'Instantiated {len(models)} models on iteration {iter_count}')
        my_logger.info(f"Instantiated {len(models)} models")
        return iter_count



    def to_dict(self) -> dict:
        """
        Convert StudyConfig instance to dict. engine will be saved with the computer name
       from_dict will replace these.
        :return: a dict. Keys are attributes.
        """

        dct = super().to_dict()
        # REPLACE all paths with PurePaths
        for var in dct.keys():
            if isinstance(dct[var], pathlib.PurePath):
                dct[var] = pathlib.PurePath(dct[var])

        my_logger.debug(f"Replacing models in model_index with config_path")
        m2 = dict()
        for key, model in dct['model_index'].items():
            m2[key] = pathlib.PurePath(model.config_path)
        dct['model_index'] = m2
        dct['config'] = self.config.to_dict()  # convert Config to a dict.
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> SubmitStudy:
        """
        Convert a dct back to a SubmitStudy. Does the following:
           decodes config -- needs special handling. FIXME: re-engineer StudyConfig so it needs less special handling....
           Creates the object
           Copies over attributes from dct to any existing attributes
           Sets up submission engine using its name
           Loads up models from paths that are saved.
        :param dct: dict containing attributes to be converted
        :return: a SubmitStudy object
        """
        # TODO: (if needed) have some way of loading up model info if the whole lot been moved.
        # deal with config
        config_dct = dct.pop('config')  # extract the config info.
        config = dictFile(Config_dct=config_dct[
            'Config']).to_StudyConfig()  # convert the config entry to a dictFile then convert to a StudyConfig.
        config._filename = config_dct['_filename']
        # TODO Very messy code. Good to sort out StudyConfig but that needs a big re-engineering job..

        # deal with translation and conversion to paths (if possible)
        dct = cls.convert_pure_paths(dct)

        # create the SubmitStudy object
        obj = cls(config)
        obj.fill_attrs(dct, convert_pure_paths=True)  # fill in the rest of the objects attributes. Converting pure path

        # load up models.
        model_index = dict()
        right_pure_path_type = type(pathlib.PurePath())  # (will give Windows/Posix as appropriate)
        for key, ppath in obj.model_index.items():  # iterate over the paths (which is how we represent the models)
            path = cls.translate_path(ppath)  # this will be a path
            if not (isinstance(path, pathlib.Path) or type(
                    path) == right_pure_path_type):  # not the right kind of pure path ?
                my_logger.warning(f"Path {ppath} not of correct type={type(path)} not {right_pure_path_type}. Skipping")
                continue

            path = pathlib.Path(path)  # make path version which we can then load.
            if not path.is_absolute(): # A relative path. Append rootdir
                
                new_path = obj.rootDir/path
                if not new_path.exists(): # path does not exist. Try with slightly different path
                    new_path = obj.rootDir.parent/path
                path  = pathlib.Path(new_path)
                
            if path.exists():
                my_logger.debug(f"Loading model from {path}")
                # verify key is as expected.
                model = Model.load_model(path)  # load the model.
                got_key = obj.key_for_model(model)
                if key != got_key:  # key changed.
                    dct = cls.key_to_dict(key)
                    dct['reference'] = dct.get('reference',model.reference)
                    dct['reference_name'] = dct.get('reference_name', model.reference_name)
                    try:
                        dct['reference']= pathlib.PurePath(dct['reference'])
                    except KeyError:
                        pass

                    fixed_key = cls.key(dct)
                    if fixed_key != got_key:
                        raise ValueError(f"Key has changed from (after fixing): \n{fixed_key} to: \n{got_key} \n for model {model}")
                    else:
                        my_logger.info(f"Key for model {model} has been fixed")
                model_index[got_key] = model
            else:

                my_logger.warning(f"Failed to find {path} so ignoring.")

        obj.model_index = model_index  # overwrite the index
        return obj

    def delete(self):
        """
        Clean up SubmitStudy configuration by deleting all models and removing self.config_path.
        The Internal structure will be updated so gen_name goes back to start and will return xxxx0...0
        """
        self.kill()   #  step 1 -- kill any  jobs
        # Step 2 -- delete models
        for key, model in self.model_index.items():
            model.delete()  # delete the model.
        self.model_index = dict()
        self.iter_keys = dict()
        # step 2 -- update internal state
        # remove the config_path.
        self.config_path.unlink(missing_ok=True)  # remove the config path.
        # remove the directory.
        shutil.rmtree(self.rootDir, ignore_errors=True)
        # reset values count (used to generate name) to 0.
        self.name_values = None  # start again!
        self.update_history("Deleted")

    def copyConfig(self,direct:pathlib.Path,

             extra_files:typing.Optional[list[pathlib.Path]]=None,
             keep_list:typing.Optional[list[pathlib.Path]]=None,
             new_config_name: typing.Optional[str] = None,
             update_paths:bool = True) -> SubmitStudy:
        """
        Copy SubmitStudy to a new directory. By default, only the config file is copied.
        :param direct: directory where study is to be copied. Will be created if it does not exist
        :param extra_files: list of extra files (paths provided relative to rootDir) to be copied to new directory.
        :param keep_list: list of files to keep in the new directory.
        :param update_paths -- If True then any path parameters will be updated to reflect new directory structure.
        :param new_config_name: If not None then the config file will be renamed to new_config_name and name updated.
        :return: Copied SubmitStudy.
        """

        # check that direct is an abs path. If not make it abs.
        if not direct.is_absolute():
            direct = pathlib.Path.cwd() / direct
            my_logger.info(f"Converting direct to absolute path {direct}")
        direct.mkdir(parents=True, exist_ok=True)  # create directory if need be.

        config_path = self.config_path.resolve().relative_to(self.rootDir)
        if new_config_name is not None:
            config_path = config_path.parent / new_config_name
            my_logger.info(f"Renaming config file to {config_path}")
        files_to_copy = [config_path]  # default is just the config file.
        if extra_files is not None:
            files_to_copy += extra_files

        files_to_copy = list(set(files_to_copy))  # make unique

        files_copied = genericLib.copy_files(self.rootDir, direct, files_to_copy, keep_list=keep_list)
        missing = set(files_to_copy) - set(files_copied)
        if len(missing) > 0:
            my_logger.warning(f"Failed to copy  {missing} from {self.rootDir} to {direct}")
        cp_submit_study = copy.deepcopy(self)  # copy the submit study
        if new_config_name is not None:
            cp_submit_study.name=config_path.stem
        cp_config_path = direct /files_to_copy[0] # new config path
        # now copy the model(s) to the new directory
        model_index = dict()  # empty  model index
        for key,model in self.model_index.items():
            new_dir = direct/(model.model_dir.relative_to(self.rootDir) )# new directory for model.
            m =  model.copyConfig(new_dir, update_paths=update_paths) # model path(s) changed so need to change model.
            model_index[key] = m


        if update_paths:
            cp_submit_study.rootDir = direct
            cp_submit_study.config_path = cp_config_path
            cp_submit_study.update_history(f"Copied {len(files_copied)} from {self.rootDir} to {direct}")
            cp_submit_study.model_index = model_index
            cp_submit_study.update_history(f"Copied {len(self.model_index)} models to {direct} ")

        cp_submit_study.dump(cp_config_path)  # dump the new config
        # and we are done!
        return cp_submit_study

    def update_params(self,update_parameters:list[str]) -> dict[str,str]:
        """
        Update parameters in config & models based on update_parameters list.
        Update done in place by changing model_index
        :param update_parameters: parameters to update
        :return: key mappings from old key to new key.  This to allow other things to be updated.
        """
        key_mappings  = dict()
        model_info = dict()
        for key,model in self.model_index.items():
            model.update_params(update_parameters)
            new_key = self.key_for_model(model)
            model_info[new_key] = model
            key_mappings[key] = new_key
        self.model_index = model_info # update the model index
        self.update_history(f"Updated parameters {update_parameters} in all models")
        return key_mappings

    def archive(self,
                archive: tarfile.TarFile,
                extra_paths: typing.Optional[List[pathlib.Path]] = None) -> list[pathlib.Path]:
        """
        Archive SubmitStudy and all its model configurations to archive.
        :param archive: archive to be written into.
        :param extra_paths -- a list of extra paths to be archived. For example, final_json and monitor paths.
          Should be specified relative to rootDir.
        :return files that got archived.

        """

        # archive ourselves!
        with tempfile.TemporaryDirectory() as tmpdir:
            # need to copy ourselves to a temporary directory first.
            tpth = pathlib.Path(tmpdir)
            self.copyConfig(tpth,extra_files=extra_paths,update_paths=False) # just copy -- no changes.
            files = list(set(tpth.rglob('*')))  # get all unique files
            for f in files:
                if f.is_dir():
                    continue
                arcname = f.relative_to(tpth)
                archive.add(f, arcname)
            my_logger.info(f"Added {self} to archive")

        return files


    def delete_model(self, model):
        """
        Delete a model and remove it from the indices
        :param model: model to delete
        :return:
        """
        key = self.key_for_model(model)
        m = self.model_index.pop(key)
        if m != model:
            raise ValueError(f"Something wrong popped model {m} is not the same as model: {model}")
        model.delete()
        self.iter_keys.pop(key)  # remove it from the iteration info.
        my_logger.info("Deleted model with key {key}")

    def gen_name(self, reset=False):
        """
        generate the next name .  Will be self.config.baseRunID() + maxDigit chars. Chars are 0-9,a-z
        and will increment every time called. First time it is called then internal counter will be reset
          to zero. Counter is incremented before name is generated.
        :param reset: Reset internal counter to zero so starting sequence again.
        For example IA0Az
         :return: name
        """
        # initialisation

        base = self.config.baseRunID()  # get the baserun
        max_digits = self.config.maxDigits()  # get the maximum length of string for model.
        chars = string.digits + string.ascii_lowercase  # windows does not case distinguish. Silly windows.
        radix = len(chars)
        # increment counter or restart
        if (reset is True) or (self.name_values is None):  # reset the counter
            self.name_values = [0] * max_digits
        elif (max_digits > 0):  # increase the values if we have digits.
            self.name_values[0] += 1
            for indx in range(0, len(self.name_values)):
                if self.name_values[indx] >= radix:
                    self.name_values[indx] = 0
                    try:
                        self.name_values[indx + 1] += 1
                    except IndexError:
                        raise ValueError(f"values too large {self.name_values}")
        else:  # just return the base name
            return base

        # now to create digit_str
        digit_str = ''
        for v in self.name_values[::-1]:
            digit_str += chars[v]
        name = base + digit_str

        # give a warning if run out of names

        if (max_digits > 0) & (self.name_values == [radix] * max_digits):
            my_logger.warning(f"Ran out of names name_values = {self.name_values}")
        return name  # return name

    def models_to_instantiate(self) -> list[Model]:
        """
        return a list of  models that need instantiation.
        :return:list of models that need instantiation
        """
        models_to_instantiate = [model for model in self.model_index.values() if model.is_instantiable()]

        return models_to_instantiate

    def models_to_submit(self) -> list[Model]:
        """
        return a list of  models that need submission.
        :return:list of models that need submission
        """
        models_to_submit = [model for model in self.model_index.values() if model.is_submittable()]

        return models_to_submit

    def models_to_continue(self) -> list[Model]:
        """

        :return: a list of models that are marked to continue
        """
        models_to_continue = [model for model in self.model_index.values() if model.is_continuable()]

        return models_to_continue

    def failed_models(self) -> list[Model]:
        """

        :return: list of models that have failed
        """

        return [model for model in self.model_index.values() if model.is_failed()]

    def running_models(self) -> list[Model]:
        """

        :return: List of models that are running
        """
        return [model for model in self.model_index.values() if model.is_running()]

    def submitted_models(self) -> list[Model]:
        """

        :return: List of models that are running
        """
        return [model for model in self.model_index.values() if model.is_submitted()]

    def processed_models(self) -> list[Model]:
        """

        :return: List of models that have processed
        """
        return [model for model in self.model_index.values() if model.is_processed()]

    def succeeded_models(self) -> list[Model]:
        """
        :return: List of models that have succeeded
        """
        return [model for model in self.model_index.values() if model.is_succeeded()]

    def submit_all_models(self, fake_fn: Optional[Callable] = None):
        """
        Submit models, the post-processing, and the next iteration in the algorithm to job control system.
        :param fake_fn:Function to fake model runs -- will skip most stages, including post-processing.
          fake and anything to be continued will generate an error.  No pp or next submission will be done if provided,
        :return: number of models submitted

        Does the following:
            1) Submits the models & post processing jobs
            2) If any post-processing jobs were submitted then submits  self.next_iter_cmd
               so once the  post-processing jobs has completed the next bit of the algorithm gets ran.
            3) When all the post-processing jobs are done the resubmission will be ran.

        This algorithm is not particularly robust to failure -- if anything fails the various jobs will be sitting around
        Releasing them will be quite tricky! You can always kill everything, remove any continuing models and start again.
        The models and study will contain info on jobs so you might be able to fix/kill by hand.
        """


        model_list = self.models_to_submit()  # models that need submitting!
        if len(model_list) == 0:  # nothing to do. We are done (no post-processing or resubmission to be submitted)
            return 0

        models_to_continue = self.models_to_continue()  # models that need continuing.
        config = self.config
        configName = config.name()

        maxRuns = self.config.maxRuns()

        output_dir = self.rootDir / 'jobOutput'  # directory where output goes for post-processing and next stage.
        # try and create the outputDir
        output_dir.mkdir(parents=True, exist_ok=True)
        my_logger.debug(f"Created {output_dir}") 

        if len(models_to_continue) > 0:  # (re)submit  models that need continuing and exit
            if fake_fn is not None:
                raise ValueError('Faking and continuing not allowed')
            if (maxRuns is not None) and (maxRuns < len(models_to_continue)):
                models_to_continue = models_to_continue[0:maxRuns]
                my_logger.debug(f"Truncating models_to_continue to {maxRuns}")

            for model in models_to_continue:
                pp_jid = model.submit_model()
                my_logger.debug(f"Continuing {model.name}  ")

            my_logger.info(f"Continued {len(models_to_continue)} models")
            self.update_history(f"Continued {len(models_to_continue)} models")
            self.dump_config()  # and write out the Study
            return len(models_to_continue)
            # nothing else to do -- next stage is still sitting  in the Q waiting to be released.
            # Will be submitted once all the post-processing jobs have been run.

        # No runs to continue, so let's submit new runs
        # Deal with maxRuns.
        if (maxRuns is not None) and (maxRuns < len(model_list)):  # need to truncate no of runs?
            my_logger.debug(f"Reducing to {maxRuns} models.")
            model_list = model_list[0:maxRuns]

        # submit models! Faking if necessary.
        pp_jids = []  # list of job ids from post-processing
        for model in model_list:  # submit model and post-processing
            pp_jids.append(model.submit_model(fake_function=fake_fn))

        if fake_fn:
            my_logger.info(f"Faked {len(model_list)} jobs")
            self.update_history(f"Faked {len(model_list)} jobs")
            # if faking will have Nones so remove them from pp_jids. This allows the possibility of mixing them
            pp_jids = [pp_jid for pp_jid in pp_jids if pp_jid is not None]
            # note that engine.submit handles an empty hold list.
        else:
            my_logger.info(f"Submitted {len(model_list)} jobs")
            self.update_history(f"Submitted {len(model_list)} models")

        # now (re)submit this entire script so that the next iteration in the algorithm can be ran
        # pp_jids are the jobs ids from the post-processing. We remove the None when if Faking it.

        if (self.next_iter_cmd is not None) and (len(pp_jids) > 0):
            # submit the next job in the iteration if have one and submitted post-processing.
            run_info = config.run_info() # get out the run_info
            submit_params = self.engine.extract_job_submission_params(run_info) # and extract the submission parameters
            iter_count = np.max(list(self.iter_keys.values()))  # iteration we are at.
            next_job_name = f"{configName}_{iter_count}"
            run_next_submit = self.engine.submit_cmd(self.next_iter_cmd, next_job_name,
                                                     outdir=output_dir,
                                                     hold=pp_jids,
                                                     **submit_params)
            output = self.run_cmd(run_next_submit)
            my_logger.info(f"Next iteration cmd is {run_next_submit} with output:{output}")
            jid = self.engine.job_id(output)  # extract the actual job id.
            my_logger.info(f"Job ID for next iteration is {jid}")
            self.next_iter_jids.append(
                jid)  # append jid to list of jobs. That way if have problems in previous jobs can get info back.
            self.update_history(f"Submitted next job with ID {jid}")

        self.dump_config()  # and write ourselves out
        return len(model_list)  # all done now

    def guess_failed(self):
        """
        Set status of running or submitted models to failed using model.guess_failed()
        :return: List of models that were guessed to have failed. Their status will be FAILED.
        """
        models_guess_fail = []
        for model in self.running_models():
            failed = model.guess_failed()  # guess if running model has actually failed.
            if failed:
                models_guess_fail.append(model)
        for model in self.submitted_models():
            failed = model.guess_failed()
            if failed:
                models_guess_fail.append(model)
        my_logger.info(f"{len(models_guess_fail)} Models were set to FAILED.")
        return models_guess_fail

    def to_study(self) -> Study:
        """
        Convert to a study.
        Study instances only have read access to info. Useful if you don't want to accidentally modify state.
          config_path will be set to None to further reduce risk.
        :return: Study
        """

        study = Study(self.config, name=self.name, rootDir=self.rootDir)
        for key, var in vars(self).items():
            if hasattr(study, key):
                setattr(study, key, copy.deepcopy(var))  # make a copy of var and add it as an attribute to study

        return study

    def process(self,
                reprocess:bool = False) -> list[Model]:
        """
        Process the study by running model.process on all models that are in SUCCEEDED state and for which the
          pp_jid is NotFound or None.
        :param reprocess: If True then in addition re-run the post-processing that are PROCESSED.
        This method is really for dealing with cases where the post-processing has, for some reason, failed.
        :return: list of models that were processed or re-processed.
        """

        # deal with models that need processing.
        models = self.succeeded_models() # list of succeeded models.
        # only want those models that have a pp_jid that is None or NotFound. These are in error state.
        #TODO -- add error state to model so when it fails it stores that.
        models = [model for model in models if (model.pp_job_status() == 'notFound' )or (model.pp_job_status() is None)]
        if reprocess:  # reprocess all models that have processed as well as those that have succeeded.
            models  += self.processed_models()
        for model in models: # actually process the models.
            model.process()

        count_models_processed = len(models)  # count how many models we processed.
        if count_models_processed > 0: # processed any models?
            my_logger.info(f"Processed {count_models_processed} models")
            self.update_history(f"Processed {count_models_processed} models")
            self.dump_config() # and write ourselves out
        return models


    def resub_status(self) -> typing.Optional[str]:
        """"
        Get the status of the next iteration job.
        :return: status of the next iteration job. If no next iteration job then return None
        """
        if len(self.next_iter_jids) == 0:
            return None
        curr_resub_id = self.next_iter_jids[-1]
        status = self.engine.job_status(curr_resub_id)
        return status

    def kill(self) -> List[str]:
        """
        Kill all jobs that are running or submitted.
           This implementation calls model.kill() for each model in the model_index and kills the next iteration job
        :return: list of job ids that we attempted to kill.
        """
        killed = []
        for model in self.model_index.values():
            killed += model.kill() # record the job ids that were killed.
        # kill next iteration job
        status = self.resub_status()  # get the status of the next iteration job
        if  ((status is not  None) and (status != 'notFound') and (len(self.next_iter_jids) > 0)):
            curr_resub_id = self.next_iter_jids[-1]
            cmd = self.engine.kill_job(curr_resub_id)
            self.run_cmd(cmd)
            killed.append(curr_resub_id)
            my_logger.info(f'Killed resubmission job id:{curr_resub_id}')
            self.update_history(f"Killed resubmission job id:{curr_resub_id}")

        my_logger.info(f"Killed {len(killed)} jobs")
        self.update_history(f"Killed {len(killed)} jobs")
        return killed

    def iter_cmd(self, iter_cmd:typing.Optional[ list[str | pathlib.Path]]) -> \
            typing.Optional[list[str | pathlib.Path]]:
        """
        If iter_cmd is Truthy set next_iter_cmd and update history. If it is the same as current value no change will be made.
        :param next_iter_cmd: Command to be used -- a list of strings or pathlib.Path objects.
        :return:whatever next_iter_cmd was set to.
        """
        if (iter_cmd and ((self.next_iter_cmd is None) or (self.next_iter_cmd != iter_cmd))):
            # only update self.next_iter_cmd if  either self.next_iter_cmd is empty/None or
            #  self.next_iter_cmd is different from iter_cmd
            self.update_history(f'Modifying {self.next_iter_cmd} to {iter_cmd}')
            logging.debug(f'Setting next_iter_cmd to {iter_cmd}')
            self.next_iter_cmd = iter_cmd

        return iter_cmd





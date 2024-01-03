"""
Support for handling model submission.
Porting hints:
1) Work out how, on your cluster, submission is done and modify define_submission.
2) See eddie_ssh for an example of a submit_cmd (which you might need if your workers cannot submit directly)
3) System assumes your model is setup to run on your cluster using a script which is setup for whatever Q system
   your computer uses.

# TODO: Have some way of copying configuration somewhere else which handles messy business of changing paths.
# Mainly so can run a model with different obs making use of the existing simulations.
# Can use archive functionality...
# have an UPDATE method which updates all model simulated observations by rerunning the processing with, potentially, updated
# configuation file/code.
# Will update the obs too.
"""
from __future__ import annotations

import copy
import json
import logging
import pathlib
import platform
import string
import sys
import tempfile
import typing
import shutil
import importlib
import tarfile
import os

from typing import Optional, List, Callable, Mapping

import numpy as np
import pandas as pd

import engine
import generic_json
from Model import Model
from model_base import model_base, journal
from Study import Study
from StudyConfig import OptClimConfigVn3, dictFile

# check we are version 3.8 or above.

if (sys.version_info.major < 3) or (sys.version_info.major == 3 and sys.version_info.minor < 8):
    raise Exception("Only works at 3.8+ ")

__version__ = '0.9'

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
    next_iter_cmd: typing.Optional[list[str]]
    next_iter_jids: list

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

    fn_type = Callable[[Mapping], pd.Series]  # type hint for fakeFn

    def __init__(self,
                 config: Optional[OptClimConfigVn3],
                 name: Optional[str] = None,
                 rootDir: Optional[pathlib.Path] = None,
                 refDir: Optional[pathlib.Path] = None,
                 models: Optional[List[Model]] = None,
                 model_name: Optional[str] = None,
                 config_path: Optional[pathlib.Path] = None,
                 next_iter_cmd: typing.Optional[typing.List[str]] = None):
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
        self.rootDir.mkdir(parents=True, exist_ok=True)  # create it if need be.
        if refDir is None:
            refDir = self.expand(str(config.referenceConfig()))
        self.refDir = refDir

        if model_name is not None:  # This is fixed. Even if configuation changed the model_name is fixed.
            self.model_name = model_name
        else:
            self.model_name = config.model_name()

        self.module_name = None
        # see if we have model_name in the list of known models. If we don't then try and load from module
        if self.model_name not in Model.known_models():
            self.module_name = config.module_name(model_name=self.model_name)
            my_logger.info(f"Loading {self.module_name}")
            importlib.import_module(self.module_name)  # and load the module.
        else:
            my_logger.info(f"Already have {self.model_name} so not loading module")

        self.run_info = copy.deepcopy(config.run_info())  # copy run_info as modifying it.
        eng = engine.abstractEngine.create_engine(self.run_info.pop('submit_engine'),
                                                  ssh_node=self.run_info.pop('ssh_node', None))

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

    def update_config(self, config: OptClimConfigVn3):
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

    def create_model(self, params: dict, dump: bool = True) -> typing.Optional[Model]:
        """
        Create a model, update list of created models and index of models.

        :param   params: dictionary of parameters to create the model.
         The following parameters are special and handled differently:
           * reference -- the reference directory. If not there (or None) then self.refDir is used.
           * model_name -- the model type to be created. If not in params then then self.model_name is used.
           These support more complex algorithms where multiple models need to be ran.
        These will be augmented by fixedParams
        If you need functionality beyond this you may want to inherit from SubmitStudy and
          override create_model to meet your needs
        :param dump: If True dump  self (using self.dump_config method)
        :return: Model created (or model that already exists). Returns None if would make more than max_model_sims
        """

        name = self.gen_name()
        model_dir = self.rootDir / name
        if model_dir.exists():
            raise ValueError(f"model_dir {model_dir} already exists")
        config_path = model_dir / (name + '.mcfg')  # create model config in model dir
        if config_path.exists():
            raise ValueError(f"config_path {config_path} already exists")
        paramDir = copy.deepcopy(params)
        reference = paramDir.pop('reference', self.refDir)
        model_name = paramDir.pop('model_name', self.model_name)
        post_process = self.config.getv('postProcess')
        run_info = self.config.run_info()
        study = self.to_study()  # convert SubmitStudy to Study
        model = Model.model_init(model_name, name=name,
                                 reference=reference,
                                 model_dir=model_dir,
                                 config_path=config_path,
                                 parameters=paramDir,
                                 post_process=post_process,
                                 study=study,
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
        my_logger.info(f"Created model {model} with parameters {model.parameters}")
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
    def load_SubmitStudy(cls, config_path: [pathlib.Path, str],
                         Study: bool = False) -> typing.Union[Study, SubmitStudy]:
        """
        Load a SubmitStudy (or anything that inherits from it) from a file.
        The object will have its config_path replaced by config_path passed in.
        :param config_path: path to configuration to load
        :param Study: If True return a Study object. These are read-only (unless you modify by hand the attributes)
        :return: object
        """
        config_path = cls.expand(config_path)
        # convert str to path and or expand user or env vars.

        obj:SubmitStudy = cls.load(config_path,check_types=[SubmitStudy])

        #TODO -- consider removing these as archive handles the rewritting needed to make work.
        # Instead trigger an error???
        if not config_path.samefile(obj.config_path):
            my_logger.info(f"Modifying config path from  {obj.config_path} to {config_path}")
            obj.config_path = config_path
            obj.update_history(f"Modified config path from  {obj.config_path} to {config_path}")

        if not config_path.parent.samefile(obj.rootDir):
            my_logger.info(f"Modifying config rootDir from  {obj.rootDir} to {config_path.parent}")
            obj.config_path = config_path
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

    def models_to_instantiate(self) -> List[Model]:
        """
        return a list of  models that need instantiation.
        :return:list of models that need instantiation
        """
        models_to_instantiate = [model for model in self.model_index.values() if model.is_instantiable()]

        return models_to_instantiate

    def models_to_submit(self) -> List[Model]:
        """
        return a list of  models that need submission.
        :return:list of models that need submission
        """
        models_to_submit = [model for model in self.model_index.values() if model.is_submittable()]

        return models_to_submit

    def models_to_continue(self) -> List[Model]:
        """

        :return: a list of models that are marked to continue
        """
        models_to_continue = [model for model in self.model_index.values() if model.is_continuable()]

        return models_to_continue

    def failed_models(self) -> List[Model]:
        """

        :return: list of models that have failed
        """

        return [model for model in self.model_index.values() if model.is_failed()]

    def running_models(self) -> List[Model]:
        """

        :return: List of models that are running
        """
        return [model for model in self.model_index.values() if model.is_running()]

    def submitted_models(self) -> List[Model]:
        """

        :return: List of models that are running
        """
        return [model for model in self.model_index.values() if model.is_submitted()]

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
                my_logger.warning(f"Path {ppath} not of correct type. Skipping")
                continue

            path = pathlib.Path(path)  # make path version which we can then load.
            if path.exists():
                my_logger.debug(f"Loading model from {path}")
                # verify key is as expected.
                model = Model.load_model(path)  # load the model.
                got_key = obj.key_for_model(model)
                if key != got_key:  # key changed. TODO. deal with ensembleMember which seems to be truncated.
                    raise ValueError(f"Key has changed from {key} to {got_key} for model {model}")
                model_index[key] = model
            else:
                my_logger.warning(f"Failed to find {path} so ignoring.")

        obj.model_index = model_index  # overwrite the index
        return obj

    def delete(self):
        """
        Clean up SubmitStudy configuration by deleting all models and removing self.config_path.
        The Internal structure will be updated so gen_name goes back to start and will return xxxx0...0
        """
        # Step 1 -- delete models
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
        # kill any resubmission job running
        if len(self.next_iter_jids) > 0:
            curr_resub_id = self.next_iter_jids[-1]
            status = self.engine.job_status(curr_resub_id)
            if status not in ['notFound']:
                cmd = self.engine.kill_job(curr_resub_id)
                self.run_cmd(cmd)
                my_logger.info(f"Killed resubmission job id:{curr_resub_id}")

        self.update_history("Deleted")

    def copy(self,direct:pathlib.Path,
             extra_paths: typing.Optional[List[pathlib.Path]] = None,
             extra_model_paths: typing.Optional[List[pathlib.Path]] = None,
             ):
        """
        Copy
        :param direct: directory where study is to be copied. Will be created if it does not exist
        :param extra_paths: extra paths to be copied. Should be provided relative to rootDir
        :param extra_model_paths: extra_paths for each model to be copied. Passed into model.copy()
        :return: Nada
        """

        if extra_paths is None:
            extra_paths = []
        direct.mkdir(parents=True,exist_ok=True)
        pth = direct/(self.config_path.relative_to(self.rootDir))
        cp = copy.deepcopy(self)
        cp.rootDir = direct
        cp.config_path = pth

        # copy the extra_paths acoss
        for path in extra_paths:
            full_path = self.rootDir / path
            if full_path.exists():
                tgt_path = direct/path# tgt path
                tgt_path.parent.mkdir(parents=True,exist_ok=True) # make directory if needed
                shutil.copy2(full_path,tgt_path)
                my_logger.info(f"Copied  {full_path} to {tgt_path} ")
                cp.update_history(f'Copied {full_path} to {tgt_path}')

        # now copy the model(s) to the new directory
        for key,model in self.model_index.items():
            new_dir = direct/model.model_dir.relative_to(self.rootDir) # new directory for model.
            cp.model_index[key] = model.copy(new_dir,extra_paths=extra_model_paths) # model path(s) changed so need to change model.

        # dump the config.
        cp.update_history(f"Copied from {self.rootDir} to {direct}")
        cp.dump_config(dump_models=True)
        # and we are done!
    def archive(self,
                archive: tarfile.TarFile,
                extra_paths: typing.Optional[List[pathlib.Path]] = None):
        """
        Archive SubmitStudy and all its model configurations to archive.
        :param archive: archive to be written into.
        :param extra_paths -- a list of extra paths to be archived. For example, final_json and monitor paths.
          Should be specified relative to rootDir.
        :return Nada!

        """

        # archive ourselves!
        with tempfile.TemporaryDirectory() as tmpdir:
            # need to dump ourselves
            pth= pathlib.Path(tmpdir)/self.config_path.name
            self.dump(pth)
            archive.add(pth,self.config_path.name)
            my_logger.info(f"Added {self} to archive")
        if extra_paths is None:
            extra_paths = []


        for path in extra_paths:
            full_path = self.rootDir / path
            if full_path.exists():
                archive.add(full_path, path)
                my_logger.info(f"Added {full_path} as {path} to archive")
        for model in self.model_index.values():  # deal with individual models
            model.archive(archive, self.rootDir)

        return


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
        maxDigits = self.config.maxDigits()  # get the maximum length of string for model.
        chars = string.digits + string.ascii_lowercase  # windows does not case distinguish. Silly windows.
        radix = len(chars)
        # increment counter or restart
        if (reset is True) or (self.name_values is None):  # reset the counter
            self.name_values = [0] * maxDigits
        elif (maxDigits > 0):  # increase the values if we have digits.
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

        if (maxDigits > 0) & (self.name_values == [radix] * maxDigits):
            my_logger.warning(f"Ran out of names name_values = {self.name_values}")
        return name  # return name

    def submit_all_models(self, fake_fn: Optional[Callable] = None):
        """
        Submit models, the post-processing and the next iteration in the algorithm to job control system.
        :param fake_fn:Function to fake model runs -- will skip most stages including post-processing.
          fake and anything to be continued will generate an error.  No pp or next submission will be done if provided,
        :return: number of models submitted

        Does the following:
            1) Submits the models & post processing jobs
            2) If any post-processing jobs were submitted then submits  self.next_iter_cmd
               so once the  post-processing jobs has completed the next bit of the algorithm gets ran.
            3) When all the post-processing jobs are done the resubmission will be ran.

        This algorithm is not particularly robust to failure -- if anything fails the various jobs will be sitting around
        Releasing them will be quite tricky! You can always kill everything, remove any continuing models and start again.
        TODO: make this a bit more robust.
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
        # All the pp_jids should be not None. We remove the None whens if Faking it.

        if (self.next_iter_cmd is not None) and (len(pp_jids) > 0):
            # submit the next job in the iteration if have one and submitted post-processing.
            run_info = config.run_info()
            runCode = config.runCode()  # NB with current implementation this is the same as run_info.get('runCode')
            iter_count = np.max(list(self.iter_keys.values()))  # iteration we are at.
            next_job_name = f"{configName}_{iter_count}"
            run_next_submit = self.engine.submit_cmd(self.next_iter_cmd, next_job_name, outdir=output_dir,
                                                     run_code=runCode,
                                                     hold=pp_jids)
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




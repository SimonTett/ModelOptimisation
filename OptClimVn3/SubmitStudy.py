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
import typing

import numpy as np
from typing import Optional, List, Callable, Mapping
import engine
import pandas as pd

from Model import Model
from model_base import model_base, journal
from Study import Study
from StudyConfig import OptClimConfigVn3, dictFile

# check we are version 3.9 or above.

if (sys.version_info.major < 3) or (sys.version_info.major == 3 and sys.version_info.minor < 9):
    raise Exception("Only works at 3.9+ ")

__version__ = '0.9'


class SubmitStudy(model_base, Study, journal):
    """

    provides methods to support working out which models need to be submitted. Creates new models and submits them.
    If you want to view a study just use the Study class
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
        :param refDir: Directory where reference model is. If None then config.referenceConfig() will be used.
        :param model_name: Name of model type to create. If None value in config is used
        :param models -- list of models.
        :param config_path -- where configuration should be stored. If None default is root_dir/name
        :param next_iter_cmd -- command to run next iteration.
        :return: instance SubmitStudy with the following public attributes :


            refDir -- path for reference directory
            model_name -- name of the model being used
            run_info -- information for submitting runs.
            engine -- functions to handle different job submission engines. Currently, only SGE and SLURM are supported.
            config_path -- path to where config is stored.
            name_values -- used to generate name. Set to None to reset counter.
            iter_keys -- dict indexed by key with iteration count.
            iter_count -- current iteration count.
            next_iter_cmd -- the command to run the next iteration.
            next_iter_jids -- the jobs ids of all submitted next_iter_cmd jobs



        """
        super().__init__(config, name=name, models=models, rootDir=rootDir)

        if refDir is None:
            refDir = self.expand(str(config.referenceConfig()))
        self.refDir = refDir

        if model_name is None:
            model_name = self.config.model_name()
        self.model_name = model_name
        self.run_info = copy.deepcopy(config.run_info()) # copy run_info as modifying it.
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

    def create_model(self, params: dict, dump: bool = True) -> Model:
        """
        Create a model, update list of created models and index of models.
        :param   params: dictionary of "variable" parameters which are generated algorithmically.
         The following parameters are special and handled differently:
           * reference -- the reference directory. If not there (or None) then self.refDir is used.
           * model_name -- the model type to be created. If not in params then then self.model_name is used.
           These support more complex algorithms where multiple models need to be ran.
        These will be augmented by fixedParams
        If you need functionality beyond this you may want to inherit from SubmitStudy and
          override create_model to meet your needs
        :param dump: If True dump  self (using self.dump_config method)
        :return: Model created (or that already exists)
        """

        name = self.gen_name()
        model_dir = self.rootDir / name
        if model_dir.exists():
            raise ValueError(f"model_dir {model_dir} already exists")
        config_path = model_dir / (name + '.mcfg')  # create model config in model dir
        if config_path.exists():
            raise ValueError(f"config_path {config_path} already exists")
        paramDir = copy.deepcopy(params)
        paramDir.update(self.config.fixedParams())  # and bring in any fixed params there are
        reference = paramDir.pop('reference', self.refDir)
        model_name = paramDir.pop('model_name', self.model_name)
        post_process = self.config.getv('postProcess')
        run_info = self.config.run_info()
        study = self.to_study()  # convert SubmitStudy to Study
        model = Model.model_init(model_name, name=name,
                                 reference=reference,
                                 model_dir=model_dir,
                                 config_path=config_path,
                                 parameters=params,
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
        logging.info(f"Created model {model} with parameters {params}")
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
        iter_count = np.max(list(self.iter_keys.values())) + 1
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
    def load_SubmitStudy(cls, config_path: pathlib.Path,
                         config: typing.Optional[OptClimConfigVn3] = None,
                         Study: bool = False) -> Study | SubmitStudy:
        """
        Load a SubmitStudy (or anything that inherits from it) from a file. The object will have config_path replaced by config_path.
        :param config_path: path to configuration to load
        :param Study: If True return a Study object. These are read-only (unless you modify by hand the attributes)
        :return: object
        """

        obj = cls.load(config_path)
        if not isinstance(obj, SubmitStudy):
            logging.warning(f"Expected instance of SubmitStudy got {type(obj)}")
        if config is not None:
            logging.info("Updating configuration")
            obj.config = copy.deepcopy(config)
        if Study:  # convert to a study
            obj = obj.to_study()
            return obj
        if not config_path.samefile(obj.config_path):
            logging.info(f"Modifying config path from  {obj.config_path} to {config_path}")
            obj.config_path = config_path
            obj.update_history(f"Modified config path from  {obj.config_path} to {config_path}")

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
        logging.info(f"Instantiated {len(models)} models")
        return iter_count

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

    def to_dict(self) -> dict:
        """
        Convert StudyConfig instance to dict. engine will be saved with the computer name
       from_dict will replace these.
        :return: a dict. Keys are attributes.
        """

        dct = super().to_dict()

        logging.debug(f"Replacing models in model_index with config_path")
        m2 = dict()
        for key, model in dct['model_index'].items():
            m2[key] = model.config_path
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

        # create the SubmitStudy object
        obj = cls(config)
        obj.fill_attrs(dct)  # fill in the rest of the objects attributes.

        # load up models.
        model_index = dict()
        for key, path in obj.model_index.items():  # iterate over the paths (which is how we represent the models)
            if path.exists():
                logging.debug(f"Loading model from {path}")
                # verify key is as expected.
                model = Model.load_model(path)  # load the model.
                got_key = obj.key_for_model(model)
                if key != got_key:  # key changed. TODO. deal with ensembleMember which seems to be truncated.
                    logging.warning(f"Key has changed from {key} to {got_key} for model {model}")
                    raise ValueError
                model_index[got_key] = model
            else:
                logging.warning(f"Failed to find {path} so ignoring.")

        obj.model_index = model_index  # overwrite the index
        return obj

    def delete(self):
        """
        Clean up SubmitStudy configuration by deleting all models and removing self.config_path.
        Internal structure will be updated so gen_name goes back to start and will return xxxx0...0
        """
        # Step 1 -- delete models
        for key, model in self.model_index.items():
            model.delete()  # delete the model.
        self.model_index = dict()
        self.iter_keys = dict()
        # step 2 -- update internal state

        # remove the config_path.
        self.config_path.unlink(missing_ok=True)  # remove the config path.
        # reset values count (used to generate name) to 0.
        # remove the directory.
        self.name_values = None  # start again!
        self.update_history("Deleted")

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
        logging.info("Deleted model with key {key}")

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
            logging.warning(f"Ran out of names name_values = {self.name_values}")
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
        if (maxRuns is not None) and (maxRuns > len(models_to_continue)):
            models_to_continue = models_to_continue[0:maxRuns]
            logging.debug(f"Truncating models_to_continue to {maxRuns}")

        if len(models_to_continue) > 0:  # (re)submit  models that need continuing and exit
            if fake_fn is not None:
                raise ValueError('Faking and continuing not allowed')
            for model in models_to_continue:
                pp_jid = model.submit_model()
                logging.debug(f"Continuing {model.name}  ")

            logging.info(f"Continued {len(models_to_continue)} models and done")
            self.update_history(f"Continued {len(models_to_continue)} models")
            self.dump_config()  # and write out the Study
            return len(models_to_continue)
            # nothing else to do -- next stage is still sitting  in the Q waiting to be released.
            # Will be submitted once all the post-processing jobs have been run.

        # No runs to continue so let's submit new runs
        # Deal with maxRuns.
        if (maxRuns is not None) and (maxRuns > len(model_list)):  # need to truncate no of runs?
            logging.debug(f"Reducing to {maxRuns} models.")
            model_list = model_list[0:maxRuns]

        # submit models! Faking if necessary.
        pp_jids = []  # list of job ids from post-processing
        for model in model_list:  # submit model and post-processing
            pp_jids.append(model.submit_model(fake_function=fake_fn))

        if fake_fn:
            logging.info(f"Faked {len(model_list)} jobs")
            self.update_history(f"Faked {len(model_list)} jobs")
            # if faking will have Nones so remove them from pp_jids. This allows the possibility of mixing them
            pp_jids = [pp_jid for pp_jid in pp_jids if pp_jid is not None]
            # note that engine.submit handles an empty hold list.
        else:
            logging.info(f"Submitted {len(model_list)} jobs")
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
            logging.info(f"Next iteration cmd is {run_next_submit} with output:{output}")
            jid = self.engine.job_id(output)  # extract the actual job id.
            logging.info(f"Job ID for next iteration is {jid}")
            self.next_iter_jids.append(
                jid)  # append jid to list of jobs. That way if have problems in previous jobs can get info back.
            self.update_history(f"Submitted next job with ID {jid}")

        self.dump_config()  # and write ourselves out
        return len(model_list)  # all done now

    def guess_failed(self):
        """
        Set status of running models to failed using model.guess_failed()
        :return: List of models that were guessed to have failed. Their status will be FAILED.
        """
        models_guess_fail = []
        for model in self.running_models():
            failed = model.guess_failed()  # guess if running model has actually failed.
            if failed:
                models_guess_fail.append(model)
        logging.info(f"{len(models_guess_fail)} Models were set to FAILED.")
        return models_guess_fail

    def to_study(self) -> Study:
        """
        Convert to a study.
        Study instances only have read access to info. Useful if you don't want to accidentally modify state.
          config_path will be set to None to further reduce risk.
        :return: Study
        """

        study = Study(self.config)
        for key, var in vars(self).items():
            if hasattr(study, key):
                setattr(study, key, copy.deepcopy(var))  # make a copy of var and add it as an attribute to study

        return study

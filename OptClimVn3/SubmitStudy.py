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
import importlib
import logging
import pathlib
import string
import sys
from typing import Optional, List, Callable, Mapping

import pandas as pd

from Model import ModelBaseClass, Model
from OptClimVn3.Models.model_base import model_base, journal
from Study import Study
from StudyConfig import OptClimConfigVn2, OptClimConfigVn3

# check we are version 3.9 or above.

if (sys.version_info.major < 3) or (sys.version_info.major == 3 and sys.version_info.minor < 9):
    raise Exception("Only works at 3.9+ ")

__version__ = '0.9'

import dataclasses


@dataclasses.dataclass(frozen=True,eq=False) # use the __eq__ in model_base
class engine(model_base):
    """
    class that holds information about submission engine
     see setup_engines if you want to modify or add more engines.
    Bit of hack around dataclasses but
    holds the following attributes (which should largely be functions):
    submit_fn -- function that will submit a job to a job control system.
            Takes one parameter -- name which is the name of the job
    array_fn -- function that will submit an array job to a job control.
            Takes two parameters -- the name and the number of tasks in the array
    release_fn --  function that takes one parameter -- the job id to release
    kill_fn -- function that takes one parameter -- the job id to kill.
    jid_fn -- function that given submission output extracts the job id,
    engine_name -- the name of the engine

    As it inherits from model_base it has a to_dict method and from_dict method.
    """
    submit_fn: callable
    array_fn: callable
    release_fn: callable
    kill_fn: callable
    jid_fn: callable
    engine_name: str

    def to_dict(self) -> dict:
        """
        Convert engine to dict by replacing callables with the callable name.
        from_dict (which loads it will reverse this)
        :return:
        """
        dct = dataclasses.asdict(self)
        for k, v in dct.items():
            if callable(v):
                dct[k] = v.__name__  # replace with name
        logging.debug(f"Converted functions in {dct} to names")
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> engine:
        """
        Convert dict to engine class. Uses engine_name and then runs setup_engine on it.
        then verifies function names are as expected.
        :param dct: dct to convert/check
        :return: Appropriately initialised engine object
        """

        obj = cls.setup_engine(engine_name=dct['engine_name'])

        # verify all OK
        for key, val in vars(obj).items():
            if callable(val) and (val.__name__ != dct[key]):
                raise ValueError(f"Inconsistency in conversion from {dct} for {key}")

        return obj

    @classmethod
    def setup_engine(cls,engine_name: str = 'SGE') -> engine:
        """
        Create an engine (used by submission system)
        time: time in minutes that jobs will run for
        mem: memory required in Mbytes.
        engine_name: name of engine wanted.
        Sets up engines which hold cmds for SGE or slurm respectively. See class engine.
        """

        # SGE-specific command options
        def sge_submit_fn(cmd: List, name: str, outdir: pathlib.Path, rundir: Optional[pathlib.Path] = None,
                          run_code: Optional[str] = None, hold_jid: str | int | bool = False
                          , time: int = 30, mem: int = 4000
                          ):
            """
            Function to submit to SGE
            :param cmd: list of commands to run.
            :param name: name of job
            :param outdir: Directory where output will be put.
            :param rundir: Directory where job will be ran. If None will run in current working dir.
            :param run_code: If provided, code to use to run the job
            :param hold_jid: If provided as a string or integer, this jobid will need to successfully run before cmd is ran.
              If provided as a bool then job will held if hold_jid is True. If False no hold will be done
            :return: the command to be submitted.
            """
            submit_cmd = ['qsub', '-l', f'h_vmem={mem}M', '-l', f'h_rt=00:{time}:00',
                          '-V',
                          "-e", outdir, "-o", outdir,
                          '-N', name]
            # -l h_vmem={mem}M: Request mem Mbytes of virtual memory per job
            # -l h_rt=00:{time}:00: Request a maximum run time of time minutes per job
            # -V: Pass the environment variables to the job
            # -N name: name of job

            if run_code is not None:
                submit_cmd += ['-A', run_code]

            if rundir is None:
                submit_cmd += ['-cwd']  # run in current working dir
            else:
                submit_cmd += ['-wd', str(rundir)]

            if isinstance(hold_jid, bool):
                if hold_jid:  # just hold the job ID
                    submit_cmd += ['-h']  # If False nothing will be held
            else:
                submit_cmd += ['-hold_jid', hold_jid]  # hold it on something

            submit_cmd += cmd
            return submit_cmd

        def sge_array_fn(cmd: List, name: str, outdir: pathlib.Path, njobs: int,
                         rundir: Optional[pathlib.Path] = None,
                         run_code: Optional[str] = None, hold_jid: str | int | bool = True
                         , time: int = 30, mem: int = 4000):
            """
            Submit an array function to SGE. Calls sge_submit_fn for details.
            :param cmd: list of commands to run.
            :param name: name of job
            :param outdir: Directory where output will be put.
            :param njobs: Number of jobs to run in the array
            :param rundir: Directory where job will be ran. If None will run in current working dir.
            :param run_code: If provided, code to use to run the job
            :param hold_jid: If provided as a string or integer, this jobid will need to successfully run before cmd is ran.
              If provided as a bool then job will held if hold_jid is True. If False no hold will be done
            :return: the command to be submitted.
            """
            result = sge_submit_fn([], name, outdir, run_code=run_code, rundir=rundir, hold_jid=hold_jid,time=time,mem=mem) + \
                     ['-t', f'1:{njobs}']  # -t task array
            result += cmd
            return result

        def sge_release_fn(jobid: str | int) -> List:
            """
            SGE cmd to release a job
            :param jobid: jobid to release
            :return: cmd to release job
            """
            return ['qrls', jobid]  # qrls: Command to release a job

        def sge_kill_fn(jobid: str | int) -> List:
            """
            SGE cmd to kill a job
            :param jobid: jobid to kill
            :return: cmd to kill a job.
            """
            return ['qdel', jobid]  # qdel: command to delete a job

        def sge_jid_fn(output:str) -> str:
            """
            Extract jobid from output of a qsub or similar command
            :param output: output from submission (or similar)
            :return: jobid as a string.
            """

            return output.split()[2].split('.')[0]



        # end of SGE functions

        # SLURM-specific command options
        def slurm_submit_fn(cmd: List, name: str, outdir: pathlib.Path, rundir: Optional[pathlib.Path] = None,
                            run_code: Optional[str] = None, hold_jid: str | int | bool = False
                            , time: int = 30, mem: int = 4000):
            """
            Function to submit to SGE
            :param cmd: list of commands to run.
            :param name: name of job
            :param outdir: Directory where output will be put.
            :param rundir: Directory where job will be ran. If None will run in current working dir.
            :param run_code: If provided, code to use to run the job
            :param hold_jid: If provided as a string or integer, this jobid will need to successfully run before cmd is ran.
              If provided as a bool then job will held if hold_jid is True. If False no hold will be done
            :return: the command to be submitted.
            """
            submit_cmd = ['sbatch', f'--mem={mem}', '--mincpus=1', f'--time={time}',
                          '--output', f'{outdir}/%x_%A_%a.out', '--error', f'{outdir}/%x_%A_%a.err',
                          '-J', name]
            # --mem={mem} Request mem mbytes  of memory per job
            # --mincpus=1: Request at least 1 CPU per job
            # --time={time}: Request a maximum run time of time  minutes per job
            # -J name Name of the job
            if run_code is not None:
                submit_cmd += ['-A', run_code]
            if rundir is not None:  # by default Slrum runs in cwd.
                submit_cmd += ['-D', str(rundir)]  # should be abs path.
            if isinstance(hold_jid, bool):
                if hold_jid:
                    submit_cmd += ['-H']  # Hold it
            else:
                submit_cmd += ['-d', f"afterok:{hold_jid}"]
            submit_cmd += cmd
            return submit_cmd

        def slurm_array_fn(cmd: List, name: str, outdir: pathlib.Path, njobs: int,
                           rundir: Optional[pathlib.Path] = None,
                           run_code: Optional[str] = None, hold_jid: str | int | bool = True
                           , time: int = 30, mem: int = 4000):
            """
            Submit an array function to SLURM. Calls slurm_submit_fn.
            :param cmd: list of commands to run.
            :param name: name of job
            :param outdir: Directory where output will be put.
            :param njobs: Number of jobs to run in the array
            :param rundir: Directory where job will be ran. If None will run in current working dir.
            :param run_code: If provided, code to use to run the job
            :param hold_jid: If provided as a string or integer, this jobid will need to successfully run before cmd is ran.
              If provided as a bool then job will held if hold_jid is True. If False no hold will be done
            :return: the command to be submitted.
            """
            result = slurm_submit_fn([], name, outdir, run_code=run_code, rundir=rundir, hold_jid=hold_jid,time=time,mem=mem) + \
                     ['-a', f'1-{njobs}']  # -a =  task array
            result += cmd
            return result

        def slurm_release_fn(jobid: int | str) -> List:
            """
            SLURM cmd to release a job
            :param jobid: The jobid of the job to be released
            :return: a list of things that can be ran!
            """
            return ['scontrol', 'release', jobid]  # Command to release a job

        def slurm_kill_fn(jobid: int | str) -> List:
            """
            Slurm cmd to kill a job
            :param jobid: The jobid to kill
            :return: command to be ran (a list)
            """
            return ['scancel', jobid]

        def slurm_jid_fn(output: str) -> str:
            """
            Extract jobid from output of sbatch or similar command
            :param output: string of output
            :return: jobid as a string.
            """
            raise NotImplementedError
            return output.split()[2].split('.')[0]

        ## end of slurm functions
        if engine_name == 'SLURM':
            engine = cls(submit_fn=slurm_submit_fn, array_fn=slurm_array_fn,
                         release_fn=slurm_release_fn, kill_fn=slurm_kill_fn,
                         jid_fn = slurm_jid_fn,
                         engine_name='SLURM')
        elif engine_name == 'SGE':
            engine = cls(submit_fn=sge_submit_fn, array_fn=sge_array_fn, release_fn=sge_release_fn,
                         kill_fn=sge_kill_fn,jid_fn=sge_jid_fn,
                         engine_name='SGE')
        else:
            raise ValueError(f"Unknown engine {engine_name}")
        return engine


class SubmitStudy(Study, journal):
    """

    provides methods to support working out which models need to be submitted. Creates new models and submits them.
    If you want to view a study just use the Study class
    """

    fn_type = Callable[[Mapping], pd.Series]  # type hint for fakeFn

    def __init__(self,
                 config: Optional[OptClimConfigVn2 | OptClimConfigVn3],
                 name: Optional[str] = None,
                 rootDir: Optional[pathlib.Path] = None,
                 refDir: Optional[pathlib.Path] = None,
                 models: Optional[List[Model]] = None,
                 model_name: Optional[str] = None,
                 fakeFn: Optional[fn_type] = None,
                 computer: Optional[str] = None):
        """
        Create ModelSubmit instance
        :param config: configuration information
        :param name: name of the study. If None name of config is used.
        :param rootDir : root dir where new directories and configuration files are to be created.
          If None will be current dir/config.name().
        :param refDir: Directory where reference model is. If None then config.referenceConfig() will be used.
        :param model_name: Name of model type to create. If None value in config is used
        :param models -- list of models.
        :param fakeFn : if provided then nothing will actually be submitted.
            Instead, the fakeFunction will be ran which will generate fake values for the observations file.
                  fakeFn(params) # given input parameters as pandas Series returns obs as pandas series.


        :param computer -- computer on which model is being ran. If None then config wil be used.
        :return: instance of ModelSubmit
        """
        super().__init__(config, name=name, models=models, rootDir=rootDir)
        self.next_job_id = None  # no next job (yet)
        self.post_process_jid = None  # no post process job (yet)

        if refDir is None:
            refDir = self.expand(config.referenceConfig())
        self.refDir = refDir

        if model_name is None:
            model_name = self.config.get("modelName")
        self.model_name = model_name

        if computer is None:
            computer = self.config.machine_name()
        self.computer = computer  # really needed for dumping/loading.
        self.engine, self.submit_fn = self.submission_engine(computer)  # engine & submit_fn for this computer.

        self.fake_fn = fakeFn
        self.runTime = self.config.runTime()  # extract time
        self.runCode = self.config.runCode()  # extract code
        self.fix_params = self.config.fixedParams()  # parameters that are fixed for all cases.
        self.name_values = None# init the counters for names.
        self.update_history(None) # will initialise the history stuff
        self.store_output(None, None)
        self.update_history(f"Created SubmitStudy {self}")

    def __repr__(self):
        """
        String that represents a SubmitStudy. Calls superclass method and adds on info about history
        :return: string
        """
        if len(self.history) > 0:
            last_hist = "Last changed at "+str(list(self.history.keys())[-1])

        else:
            last_hist = "No Hist"
        s= super().__repr__()+" "+last_hist

        return s



    @classmethod
    def submission_engine(cls, computer):
        """
        This should be modified if porting OptClim to a new computer system

        :param computer: name of computer. If SGE or SLURM then these will be used.
         Currently known computers are eddie, archer or ARC. eddie uses sge while the other two use slurm
        :return: the submission engine and submit_fn (None if not needed)
        """

        submit_fn = None
        if computer in ['SGE', 'SLURM']:  # Generic SGE/SLURM
            engine_name = computer
        elif computer == 'eddie':  # Edinburgh cluster
            engine_name = 'SGE'
        elif computer == 'archer':  # UK national super-computer
            engine_name = 'SLURM'
        elif computer == 'ARC':  # Oxford cluster.
            engine_name = 'SLURM'
        else:
            raise ValueError(f"Unknown computer {computer}")

        return engine.setup_engine(engine_name=engine_name), submit_fn

    def create_model(self, params, dump=True):
        """
        Create a model, update list of created models and index of models.
        :param   params: dictionary of "variable" parameters which are generated algorithmically.
         The following parameters are special and handled differently:
           * reference -- the reference directory. If not there (or None) then self.refDir is used.
           * model_name -- the model type to be created. If not then then then self.model_name is used.
           These support more complex algorithms where multiple models need to be ran.
        These will be augmented by fixedParams
        If you need functionality beyond this you may want to inherit from SubmitStudy and
          override create_model to meet your needs
        :param dump: If True dump model (using model.dump_model method) and self (using self.dump_config method)
        :return: model. The created model. self.model_index will be updated.
        """

        name = self.gen_name()
        model_dir = self.rootDir / name
        if model_dir.exists():
            raise ValueError(f"model_dir {model_dir} already exists")
        config_path = self.rootDir / (name + '.mcfg')
        if config_path.exists():
            raise ValueError(f"config_path {config_path} already exists")
        paramDir = copy.deepcopy(params)
        paramDir.update(self.fix_params)  # and bring in any fixed params there are
        reference = paramDir.pop('reference', self.refDir)
        model_name = paramDir.pop('model_name', self.model_name)
        post_process = self.config.get('post_process')
        model = ModelBaseClass.model_init(model_name, name=name,
                                          reference=reference, model_dir=model_dir, config_path=config_path,
                                          fake=(self.fake_fn is not None), parameters=params, post_process=post_process
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

    def dump_config(self):
        """
        Dump the configuration *and* all its models to self.config_path and their config_paths
        :return: Nothing
        """
        for model in self.model_index.values():
            model.dump_model()
        self.dump(self.config_path)




    def instantiate(self):
        """
        Instantiate all created models.
        :return: True if all were instantiated. False otherwise
        """

        models = [model for model in self.model_index.values() if model.status == 'CREATED']
        for model in models:
            model.instantiate()  # model state will be written out.
            self.update_history(f'Instantiated {model}')
        logging.info(f"Instantiated {len(models)} models")
        return True

    def models_to_submit(self) -> List[Model]:
        """
        return a list of  models that need submission.
        :return:list of models that need submission
        """
        models_to_submit = [model for model in self.model_index.values() if model.is_submittable()]

        return models_to_submit

    def to_dict(self) -> dict:
        """
        Convert StudyConfig instance to dict.
        Replaces submit_fn with function names. from_dict will replace these.
        :return: a dict. Keys are attributes.
        """
        dct = super().to_dict()
        # deal with functions in engine and submit_fn
        if callable(dct['submit_fn']):
            dct['submit_fn'] = dct['submit_fn'].__name__

        return dct
    @classmethod
    def from_dict(cls, dct: dict) -> SubmitStudy:
        """
        Convert a dct back to a SubmitStudy
        :param dct: sct containing attributes to be converted
        :return:
        """
        # first call the super class method then fix the engine.
        obj = super(cls,cls).from_dict(dct)
        obj.engine, obj.submit_fn = obj.submission_engine(obj.computer)  # engine & submit_fn for this computer.
        return obj

    def delete(self):
        """
        Clean up ModelSubmit configuration by deleting all models and removing self.config_path.
        Internal structure will be updated so gen_name goes back to start and will return xxxx0...0
        """
        # Step 1 -- delete models
        for key, model in self.model_index.items():
            model.delete()  # delete the model.
        self.model_index=dict()
        # step 2 -- update internal state

        # remove the config_path.
        self.config_path.unlink(missing_ok=True)  # remove the config path.
        # reset values count (used to generate name) to 0.
        maxDigits = self.config.maxDigits()  # get the maximum length of string for model.
        self.name_values = None # start again!
        self.update_history("Deleted")

    def gen_name(self,reset=False):
        """
        generate the next name .  Will be self.config.baseRunID() + maxDigit chars. Chars are 0-9,a-z,A-Z
        and will increment every time called. First time it is called then internal counter will be reset
          to zero. Counter is incremented before name is generated.
        :param reset: Reset internal counter to zero so starting sequence again.
        For example IA0Az
         :return: name
        """
        # initialisation

        base = self.config.baseRunID()  # get the baserun
        maxDigits = self.config.maxDigits()  # get the maximum length of string for model.
        chars = string.digits + string.ascii_letters
        radix = len(chars)
        # increment counter or restart
        if (reset is True) or (self.name_values is None): # reset the counter
            self.name_values = [0] * maxDigits
        elif (maxDigits > 0 ) :  # increase the values if we have digits.
            self.name_values[0] += 1
            for indx in range(0, len(self.name_values)):
                if self.name_values[indx] >= radix:
                    self.name_values[indx] = 0
                    try:
                        self.name_values[indx + 1] += 1
                    except IndexError:
                        raise ValueError(f"values too large {self.name_values}")
        else: # just return the base name
            return base

        # now to create digit_str
        digit_str = ''
        for v in self.name_values[::-1]:
            digit_str += chars[v]
        name = base + digit_str

        # give a warning if run out of names
        if self.name_values == [radix]*maxDigits:
            logging.warning(f"Ran out of names name_values = {self.name_values}")
        return name  # return name

    def submit_all_models(self, next_iter_cmd: Optional[list] = None,
                          fake_fn: Optional[Callable] = None):
        """
        Submit models, the post-processing and the next iteration in the algorithm to job control system.

        :param next_iter_cmd -- The command to submit to run the next iteration.
         If None then no next iteration will be submitted.
        :param fake_fn:Function to fake model runs -- will skip most stages including post-processing.
          fake and anything to be continued will generate an error.  No pp or next submission will be done if provided,
        :return: status of submission

        Does the following:
            1) Submits the post-processing jobs as a task array in held state.
                 Jobs continuing (as they failed) will not have a post-processing job as their post-processing job will still be
                   in the system. Nor will they have a next job to release (As that will already be in the system).
                   if any continue jobs submit those and be done.
            2) Submits  resubmit so once the array of post-processing jobs has completed the next bit of the algorithm gets ran.
            3) Submits the model simulations -- which once each one has run will release the appropriate post-processing task
            4) When all the post-processing jobs are done the resubmission will be ran.

        This algorithm is not particularly robust to failure -- if anything fails the various jobs will be sitting around
        Releasing them will be quite tricky! You can always kill everything, remove any continuing models and start again.
        TODO: make this a bit more robust.
        The models and study will contain info on jobs so you might be able to fix/kill by hand.
        """

        model_list = self.models_to_submit()  # models that need submitting!
        if len(model_list) == 0:  # nothing to do. We are done (no post-processing or resubmission to be submitted)
            return True
        models_to_continue = [m for m in model_list if m.status == "CONTINUE"]  # marker for continuation
        #run_fn = functools.partial(subprocess.check_output, text=True, cwd=True)  # what we are using to run things.

        config = self.config
        configName = config.name()

        ## work out postprocess script path
        OptClimRoot = importlib.resources.files("OptClimVn3")  # root path for OptClimVn3
        # scriptName = OptClimRoot / "scripts/qsub.sh"  # not sure why this is needed!
        if len(models_to_continue) > 0:  # (re)submit  models that need continuing and exit
            if fake_fn is not None:
                raise ValueError("Faking and continuing")
            for model in models_to_continue:
                output = model.submit_model(submit_fn=self.submit_fn, runTime=self.runTime, runCode=self.runCode)  #
                logging.debug(f"Continuing {model.name} ")
            logging.info(f"Continued {len(models_to_continue)} and done")
            self.update_history(f"Continued {len(models_to_continue)} models")
            self.dump_config()  # and write out the config
            return True  # nothing else to do -- post-processing should be released when model finishes.
        output_dir = self.rootDir / 'jobOutput'  # directory where output goes for post-processing and next stage.
        # try and create the outputDir
        output_dir.mkdir(parents=True, exist_ok=True)
        if fake_fn is not None:  # faking it until we make it! Only need to run some models
            for index, model in enumerate(model_list):
                # need to put the post-processing job release command in the model.
                # model will complain if not in right status
                jid = "NOID" + f".{index + 1}"  # work out the jobid for release of post-processing
                release_pp = self.engine.release_fn(jid)
                if self.submit_fn:
                    release_pp = self.submit_fn(release_pp)
                model.submit_model(self.submit_fn, post_process_cmd=release_pp,
                                   runTime=self.runTime, runCode=self.runCode, fake_function=fake_fn)
                # handling fake_fn

                logging.debug(f"Faking {model.name} which will release {jid}")

            logging.info(f"Faked {len(model_list)} jobs")
            self.update_history(f"Faked {len(model_list)} jobs")
            self.dump_config()  # and write ourselves out!
            return True  # all done now

        # normal post processing stuff
        configFile = self.rootDir / 'tempConfigList.txt'  # name of file containing list of configs files  for post-processing stage
        with open(configFile, 'wt') as f:  # write file paths for model configs to the configFile.
            for m in model_list:
                f.write(str(m.config_path)+"\n")  # Where model state is to be found.
        # generate the post-processing array job.
        pp_jobName = 'PP' + configName
        postProcess = [OptClimRoot / "scripts/post_process.sh", configFile]
        pp_cmd = self.engine.array_fn(postProcess, pp_jobName, output_dir, len(model_list),
                                      run_code=self.runCode, rundir=self.rootDir)
        if self.submit_fn:  # and thing to do to submit a job?
            pp_cmd = self.submit_fn(pp_cmd)
        logging.info(f"postProcess task array cmd is {pp_cmd}")
        # run the post process and get its job id
        output = self.run_cmd(pp_cmd)
        postProcessJID =  self.engine.jid_fn(output) # extract the actual job id as a string
        logging.info(f"postProcess array job id is {postProcessJID}")
        self.post_process_jid = postProcessJID
        self.update_history(f"Submitted postProcess array job id as {postProcessJID}")
        # now (re)submit this entire script so that the next iteration in the algorithm can be ran
        if (next_iter_cmd is not None):
            # submit the next job in the iteration.
            # need to run resubmit through a script because qsub copies script being run
            # so somewhere temporary. So lose file information needed for resubmit. Not sure this needed. TODO check if needed.
            next_job_name = 'RE' + configName
            run_next_submit = self.engine.submit_fn(next_iter_cmd, next_job_name,output_dir,
                                                    run_code=self.runCode, rundir=self.rootDir,
                                                    hold_jid=postProcessJID)
            if self.submit_fn is not None:
                run_next_submit = self.submit_fn(run_next_submit)
            logging.info("Next iteration cmd is ", run_next_submit)
            output = self.run_cmd(run_next_submit)
            jid = self.engine.jid_fn(output)  # extract the actual job id.
            logging.info(f"Job ID for next iteration is {jid}")
            self.next_job_id = jid
            self.update_history(f"Submitted next job with ID {jid}")
        # now submit the models with info on what to release.
        # This should only happen if have models to run!
        for index, model in enumerate(model_list):
            # need to put the post-processing job release command in the model.
            # model will complain if not in right status
            jid = postProcessJID + f".{index + 1}"  # work out the jobid for release of post-processing
            release_pp = self.engine.release_fn(jid)
            if self.submit_fn:
                release_pp = self.submit_fn(release_pp)
            output = model.submit_model(self.submit_fn, post_process_cmd=release_pp,
                               runTime=self.runTime, runCode=self.runCode)  # no fake_fn here as handled above.
            logging.debug(f"Submitting {m.name} which will release {jid}")

        logging.info(f"Submitted {len(model_list)} model jobs")
        self.update_history(f"Submitted {len(model_list)} model jobs")
        self.dump_config()  # and write ourselves out!
        return True


def eddie_ssh(cmd: list[str]) -> list[str]:
    """
    Example submit function for ssh on eddie
    :param cmd: command to submit -- should be a list.
    :return: modified cmd which includes ssh
    """
    cwd = str(pathlib.Path.cwd())
    s = f"cd {cwd}; " + " ".join(cmd)
    return ['ssh', 'login01.eddie.ecdf.ed.ac.uk', s]

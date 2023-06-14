from __future__ import annotations

import dataclasses
import logging
from Models.model_base import model_base
import typing
import pathlib
@dataclasses.dataclass(frozen=True, eq=False)  # use the __eq__ in model_base
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
        # and convert models to paths.
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
    def setup_engine(cls, engine_name: str = 'SGE') -> engine:
        """
        Create an engine (used by submission system)
        time: time in minutes that jobs will run for
        mem: memory required in Mbytes.
        engine_name: name of engine wanted.
        Sets up engines which hold cmds for SGE or slurm respectively. See class engine.
        """

        # SGE-specific command options
        def sge_submit_fn(cmd: typing.List, name: str,
                          outdir: typing.Optional[pathlib.Path],
                          rundir: typing.Optional[pathlib.Path] = None,
                          run_code: typing.Optional[str] = None,
                          hold_jid: typing.List[str]| str | bool = False
                          ,time: int = 30, mem: int = 4000,
                          n_cores: int = 1
                          ):
            """
            Function to submit to SGE
            :param mem: Memory (in Mybtes) needed for job
            :param time: Time (in seconds) needed for job
            :param cmd: list of commands to run.
            :param name: name of job
            :param outdir: Directory where output will be put. If None will be set to cwd/output.
            :param rundir: Directory where job will be ran. If None will run in current working dir.
            :param run_code: If provided, code to use to run the job
            :param hold_jid: If provided as a string or list of strings,
            this (these) jobids will need to successfully run before cmd is ran.
              If provided as a bool then job will held if hold_jid is True. If False no hold will be done
            :param n_cores: No of cores to use.
            :return: the command to be submitted.
            """

            if outdir is None:
                outdir = pathlib.Path.cwd()/'output'
                logging.debug(f"Set outdir to {outdir}")

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

            if isinstance(hold_jid, bool) and hold_jid:
                submit_cmd += ['-h']  # If False nothing will be held
            if isinstance(hold_jid,str):
                submit_cmd += ['-hold_jid', hold_jid]  # hold it on something
            if isinstance(hold_jid,list) and (len(hold_jid) > 0): # need a non-empty list.
                submit_cmd += ['-hold_jid', ",".join(hold_jid)]  # hold it on multiple jobs
            if n_cores > 1:  # more than 1 core wanted.
                submit_cmd += ['-pe ', f'mpi {n_cores}']  # ask for mpi env.
            submit_cmd += cmd
            return submit_cmd

        def sge_array_fn(cmd: typing.List, name: str, outdir: pathlib.Path, njobs: int,
                         rundir: typing.Optional[pathlib.Path] = None,
                         run_code: typing.Optional[str] = None, hold_jid: typing.List[str] | str | bool = True
                         , time: int = 30, mem: int = 4000, n_cores: int = 1):
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
            :param n_cores: No of cores to use. See sge_submit_fn
            :return: the command to be submitted.
            """
            result = sge_submit_fn([], name, outdir, run_code=run_code, rundir=rundir,
                                   hold_jid=hold_jid, time=time, mem=mem, n_cores=n_cores) + \
                     ['-t', f'1:{njobs}']  # -t task array
            result += cmd
            return result

        def sge_release_fn(jobid: str | int) -> typing.List:
            """
            SGE cmd to release a job
            :param jobid: jobid to release
            :return: cmd to release job
            """
            return ['qrls', jobid]  # qrls: Command to release a job

        def sge_kill_fn(jobid: str | int) -> typing.List:
            """
            SGE cmd to kill a job
            :param jobid: jobid to kill
            :return: cmd to kill a job.
            """
            return ['qdel', jobid]  # qdel: command to delete a job

        def sge_jid_fn(output: str) -> str:
            """
            Extract jobid from output of a qsub or similar command
            :param output: output from submission (or similar)
            :return: jobid as a string.
            """

            return output.split()[2].split('.')[0]

        # end of SGE functions

        # SLURM-specific command options
        def slurm_submit_fn(cmd: typing.List, name: str,
                            outdir: typing.Optional[pathlib.Path] = None,
                            rundir: typing.Optional[pathlib.Path] = None,
                            run_code: typing.Optional[str] = None,
                            hold_jid: typing.List[str]|str |  bool = False
                            , time: int = 30, mem: int = 4000, n_cores: int = 1):
            """
            Function to submit to SGE
            :param cmd: list of commands to run.
            :param name: name of job
            :param outdir: Directory where output will be put. If None then will be cwd/output
            :param rundir: Directory where job will be ran. If None will run in current working dir.
            :param run_code: If provided, code to use to run the job
            :param hold_jid: If provided as a string, this jobid will need to successfully run before cmd is ran.
              If a list (of jobid's) then all jobs will need to be run
              If provided as a bool then job will held if hold_jid is True. If False no hold will be done
            :param n_cores; No of cores to be requested
            :return: the command to be submitted.
            """
            if outdir is None:
                outdir = pathlib.Path.cwd()/'output'
                logging.debug(f"Set outdir to {outdir}")
            submit_cmd = ['sbatch', f'--mem={mem}', f'--mincpus={n_cores}', f'--time={time}',
                          '--output', f'{outdir}/%x_%A_%a.out', '--error', f'{outdir}/%x_%A_%a.err',
                          '-J', name]
            # --mem={mem} Request mem mbytes  of memory per job
            # --mincpus={n_cores}: Request at least n_cores CPU per job
            # --time={time}: Request a maximum run time of time  minutes per job
            # -J name Name of the job
            if run_code is not None:
                submit_cmd += ['-A', run_code]
            if rundir is not None:  # by default Slrum runs in cwd.
                submit_cmd += ['-D', str(rundir)]  # should be abs path.
            if isinstance(hold_jid, bool) and hold_jid:# got a book and its True. Just hold the job
                    submit_cmd += ['-H']  # Hold it
            if isinstance(hold_jid,str): # String -- hold on one job
                submit_cmd += f'--dependency=afterok:{hold_jid}'
            if isinstance(hold_jid,list) and (len(hold_jid) > 0):
                # hold on multiple jobs (empty list means nothing to be held)
                submit_cmd += "--dependency=afterok:"+":".join(hold_jid)
            submit_cmd += cmd
            return submit_cmd

        def slurm_array_fn(cmd: typing.List, name: str, outdir: pathlib.Path, njobs: int,
                           rundir: typing.Optional[pathlib.Path] = None,
                           run_code: typing.Optional[str] = None, hold_jid: typing.List[str]|str |  bool = True
                           , time: int = 30, mem: int = 4000, n_cores: int = 1):
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
            :param n_cores -- no of cores to be ran. See slurm_submit)fn
            :return: the command to be submitted.
            """
            result = slurm_submit_fn([], name, outdir, run_code=run_code, rundir=rundir,
                                     hold_jid=hold_jid, time=time, mem=mem, n_cores=n_cores) + \
                     ['-a', f'1-{njobs}']  # -a =  task array
            result += cmd
            return result

        def slurm_release_fn(jobid: int | str) -> typing.List:
            """
            SLURM cmd to release a job
            :param jobid: The jobid of the job to be released
            :return: a list of things that can be ran!
            """
            return ['scontrol', 'release', jobid]  # Command to release a job

        def slurm_kill_fn(jobid: int | str) -> typing.List:
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
                         jid_fn=slurm_jid_fn,
                         engine_name='SLURM')
        elif engine_name == 'SGE':
            engine = cls(submit_fn=sge_submit_fn, array_fn=sge_array_fn, release_fn=sge_release_fn,
                         kill_fn=sge_kill_fn, jid_fn=sge_jid_fn,
                         engine_name='SGE')
        else:
            raise ValueError(f"Unknown engine {engine_name}")
        return engine


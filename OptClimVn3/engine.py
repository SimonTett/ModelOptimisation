""""
Provide generic functions for job submission, job release, killing a job and extracting a jobid
Provides implementations for SGE andn SLURM. You might find your version of SGE or SLURM has subtle changes
to extract the job-id. If so extend the relevant class and modify setup_engine.
"""

from __future__ import annotations

import logging
import typing
import pathlib
from abc import ABCMeta, abstractmethod


def setup_engine(engine_name: typing.Literal['SGE', 'SLURM'] = 'SGE',
                 connect_fn: typing.Optional[typing.Callable] = None) -> abstractEngine:
    """
    Create an engine (used by submission system)
    :param engine_name: name of engine wanted.
    :param connect_fn: name of function to generation connection to host where sge/slurm exists.
    Sets up engines which hold cmds for SGE or slurm respectively. See class engine.
    """
    if engine_name == 'SGE':
        return sge_engine(connect_fn=connect_fn)
    elif engine_name == 'SLURM':
        return slurm_engine(connect_fn=connect_fn)
    else:
        raise ValueError(f"Do not know what to do with engine_name = {engine_name}")

class abstractEngine(metaclass=ABCMeta):
    """
    Abstract class for Engines. Inherit and implement for your own class.
    """

    def __init__(self, connect_fn: typing.Optional[typing.Callable] = None):
        """
        Initialise an Engine instance
        :param connect_fn_fn: function for submission. Will be ran on all methods.
        """
        self.connect_fn = connect_fn

    def __eq__(self,other):
        """
        Test if two engines are the same.
        types must be the same and connect_fn the same. If fn then have the same name
        :param other: other engine
        :return: True or False
        """

        if type(self) != type(other):
            print(f"Types differ {type(self)} != {type(other)}")
            return False
        if type(self.connect_fn) != type(other.connect_fn):
            print(f"Connect fns types differ {type(self.connect_fn) != type(other.connect_fn)}")
            return False
        if self.connect_fn is None:
            return True # we are both None.
        if callable(self.connect_fn) and (self.connect_fn.__name__ != other.connect_fn.__name__):
            print(f"Connect fn names differ: {self.connect_fn.__name__} != {other.connect_fn.__name__} ")
        if self.connect_fn != other.connect_fn:
            print(f"connect_fn differ {self.connect_fn} != {other.connect_fn}")
            return False

        return True

    @abstractmethod
    def submit_fn(self,
                  cmd: typing.List, name: str,
                  outdir: typing.Optional[pathlib.Path] = None,
                  rundir: typing.Optional[pathlib.Path] = None,
                  run_code: typing.Optional[str] = None,
                  hold_jid: typing.List[str] | str | bool = False,
                  time: int = 1800,
                  mem: int = 4000,
                  n_cores: int = 1,
                  n_tasks: typing.Optional[int] = None
                  ) -> typing.List[str]:
        """
        Submit function.
        :param mem: Memory (in Mybtes) needed for job
        :param time: Time (in seconds) needed for job
        :param cmd: list of commands to run.
        :param name: name of job
        :param outdir: Directory where output will be put.
             If None will be set to cwd/output.
        :param rundir: Directory where job will be ran.
           If None will run in current working dir.
        :param run_code: If provided, code to use to run the job
        :param hold_jid: If provided as a string or list of strings,
        this (these) jobids will need to successfully run before cmd is ran.
          If provided as a bool then job will held if hold_jid is True.
          If False no hold will be done
        :param n_cores: No of cores to use.
        :param n_tasks: no of tasks to run. Will submit an array job.
        :return: the command to be submitted.
        """
        pass

    @abstractmethod
    def release_fn(self, jobid: str) -> typing.List[str]:
        """
        cmd to release a job
        :param jobid: jobid to release
        :return: cmd to release job
        """
        pass

    @abstractmethod
    def kill_fn(self, jobid: str) -> typing.List[str]:
        """
        cmd to kill a job
        :param jobid: jobid to kill
        :return: cmd to kill a job.
        """
        pass

    @abstractmethod
    def jid_fn(self, output: str) -> str:
        """
        Extract jobid from output of a qsub or similar command
        :param output: output from submission (or similar)
        :return: jobid as a string.
        """
        pass

    # end of abstract functions


class sge_engine(abstractEngine):
    """
    Engine class for SGE
    """



    def submit_fn(self, cmd: typing.List, name: str,
                  outdir: typing.Optional[pathlib.Path] = None,
                  rundir: typing.Optional[pathlib.Path] = None,
                  run_code: typing.Optional[str] = None,
                  hold_jid: typing.List[str] | str | bool = False,
                  time: int = 1800,
                  mem: int = 4000,
                  n_cores: int = 1,
                  n_tasks: typing.Optional[int] = None
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
        :param n_tasks: If not None the size of the task array to be ran.
        :return: the command to be submitted.
        """

        if outdir is None:
            outdir = pathlib.Path.cwd() / 'output'
            logging.debug(f"Set outdir to {outdir}")

        submit_cmd = ['qsub', '-l', f'h_vmem={mem}M', '-l', f'h_rt={time}',
                      '-V',
                      "-e", outdir, "-o", outdir,
                      '-N', name]
        # -l h_vmem={mem}M: Request mem Mbytes of virtual memory per job
        # -l h_rt={time}: Request a maximum run time of time seconds per job
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
        if isinstance(hold_jid, str):
            submit_cmd += ['-hold_jid', hold_jid]  # hold it on something
        if isinstance(hold_jid, list) and (len(hold_jid) > 0):  # need a non-empty list.
            submit_cmd += ['-hold_jid', ",".join(hold_jid)]  # hold it on multiple jobs
        if n_cores > 1:  # more than 1 core wanted.
            submit_cmd += ['-pe ', f'mpi {n_cores}']  # ask for mpi env.
        if n_tasks is not None:  # want to run a task array
            submit_cmd += ['-t', f'1:{n_tasks}']
        submit_cmd += cmd
        if callable(self.connect_fn):
            submit_cmd = self.connect_fn(submit_cmd)
        return submit_cmd

    def release_fn(self, jobid: str) -> typing.List[str]:
        """
        SGE cmd to release a job
        :param jobid: jobid to release
        :return: cmd to release job
        """
        cmd = ['qrls', jobid]  # qrls: Command to release a job
        if callable(self.connect_fn):
            cmd = self.connect_fn(cmd)
        return cmd

    def kill_fn(self, jobid: str) -> typing.List[str]:
        """
        SGE cmd to kill a job
        :param jobid: jobid to kill
        :return: cmd to kill a job.
        """
        cmd = ['qdel', jobid]  # qdel: command to delete a job
        if callable(self.connect_fn):
            cmd = self.connect_fn(cmd)
        return cmd

    def jid_fn(self, output: str) -> str:
        """
        Extract jobid from output of a qsub or similar command
        :param output: output from submission (or similar)
        :return: jobid as a string.
        """

        return output.split()[2].split('.')[0]


class slurm_engine(abstractEngine):
    """
    Engine for SLURM
    """

    def submit_fn(self, cmd: typing.List, name: str,
                  outdir: typing.Optional[pathlib.Path] = None,
                  rundir: typing.Optional[pathlib.Path] = None,
                  run_code: typing.Optional[str] = None,
                  hold_jid: typing.List[str] | str | bool = False,
                  time: int = 1800,
                  mem: int = 4000,
                  n_cores: int = 1,
                  n_tasks: typing.Optional[int] = None):
        """
        Function to submit command to SLURM

        :param cmd: list of commands to run.
        :param name: name of job
        :param outdir: Directory where output will be put. If None then will be cwd/output
        :param rundir: Directory where job will be ran. If None will run in current working dir.
        :param run_code: If provided, code to use to run the job
        :param hold_jid: If provided as a string, this jobid will need to successfully run before cmd is ran.
          If a list (of jobid's) then all jobs will need to be run
          If provided as a bool then job will held if hold_jid is True. If False no hold will be done
        :param mem: Memory (in Mybtes) needed by job
        :param time: Time (in seconds) needed by job,
        :param n_cores: No of cores to be requested
        :param n_tasks: If not not None the number of array tasks to run.
        :return: the command to be submitted.
        """
        if outdir is None:
            outdir = pathlib.Path.cwd() / 'output'
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
        if isinstance(hold_jid, bool) and hold_jid:  # got a book and its True. Just hold the job
            submit_cmd += ['-H']  # Hold it
        if isinstance(hold_jid, str):  # String -- hold on one job
            submit_cmd += f'--dependency=afterok:{hold_jid}'
        if isinstance(hold_jid, list) and (len(hold_jid) > 0):
            # hold on multiple jobs (empty list means nothing to be held)
            submit_cmd += "--dependency=afterok:" + ":".join(hold_jid)
        if n_tasks is not None:
            submit_cmd += ['-a', f'1-{n_tasks}']  # -a =  task array
        submit_cmd += cmd
        if callable(self.connect_fn):
            submit_cmd = self.connect_fn(submit_cmd)
        return submit_cmd

    def release_fn(self, jobid: str) -> typing.List:
        """
        SLURM cmd to release a job
        :param jobid: The jobid of the job to be released
        :return: a list of things that can be ran!
        """
        cmd = ['scontrol', 'release', jobid]  # Command to release a job
        if callable(self.connect_fn):
            cmd = self.connect_fn(cmd)
        return cmd

    def kill_fn(self, jobid: str) -> typing.List:
        """
        Slurm cmd to kill a job
        :param jobid: The jobid to kill
        :return: command to be ran (a list)
        """
        cmd = ['scancel', jobid]
        if callable(self.connect_fn):
            cmd = self.connect_fn(cmd)
        return cmd

    def jid_fn(self, output: str) -> str:
        """
        Extract jobid from output of sbatch or similar command
        :param output: string of output
        :return: jobid as a string.
        """

        return output.split()[2].split('.')[0]



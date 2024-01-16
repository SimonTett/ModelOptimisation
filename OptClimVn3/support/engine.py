"""
Provide generic functions for job submission, job release_job, killing a job and extracting a jobid
Provides implementations for SGE and SLURM. You might find your version of SGE or SLURM has subtle changes
to extract the job-id. If so extend the relevant class and modify setup_engine.
"""

from __future__ import annotations

import logging
import os
import subprocess
import typing
import pathlib
from abc import ABCMeta, abstractmethod
from model_base import model_base, journal  # so can save things. The default to_dict, from_dict should work.

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

class abstractEngine(model_base, journal):
    """
    Abstract class for Engines. Inherit and implement for your own class.
    """
    allowed_eng = typing.Literal['SGE', 'SLURM']  # allowed engines

    @classmethod
    def create_engine(cls, engine_name: allowed_eng = 'SGE',
                      ssh_node: typing.Optional[str] = None) -> abstractEngine:
        """
        Create an engine (used by submission system)
        :param engine_name: name of engine wanted.
        :param ssh_node: node to ssh to where engine can submit things
        Sets up engines which hold cmds for SGE or slurm respectively. .
        """

        if engine_name == 'SGE':
            return sge_engine(ssh_node=ssh_node)
        elif engine_name == 'SLURM':
            return slurm_engine(ssh_node=ssh_node)
        else:
            raise ValueError(f"Do not know what to do with engine_name = {engine_name}")

    def __init__(self, ssh_node: typing.Optional[str] = None):
        """
        Initialize an Engine instance
        :param: ssh_node -- if not None then the name of the host to run commands on.
        """
        self.ssh_node = ssh_node

    def connect_fn(self,
                   cmd: list[str],
                   rundir: typing.Optional[pathlib.Path | str] = None) -> list[str]:
        """
        Connect command. Assumes that filesystems match on all nodes and runs commands on self.ssh_node
         If self.ssh_node is None -- just returns the cmd
        :param cmd: command to run
        :param rundir: Where to run the command.
           If not provided command will be run in current working dir when the generated command is run.
            This assumes you are running the command with  journal.run_cmd(). See doc for that method.
            If rundir is a shell var then it will be expanded when the command is run if journal.run_cmd is used.
        :return: modified cmd which includes ssh
        """
        if self.ssh_node is None:
            return cmd

        if rundir is None:
            rundir = '$PWD'  # path at time of running (on linux systems)
        s = f"cd {str(rundir)}; export PYTHONPATH=$PYTHONPATH ; export PATH=$PATH; export OPTCLIMTOP=$OPTCLIMTOP; " + \
            " ".join([str(c) for c in cmd])
        # run_cmd (which will eventually run the command) will expand env variables with values, at the time it is run.
        my_logger.debug(f"{s} will be run on {self.ssh_node}")
        return ['ssh', self.ssh_node, s]

    def __eq__(self, other):
        """
        Test if two engines are the same.
        types must be the same and ssh_nodes must also be the same
        :param other: other engine
        :return: True or False
        """

        if type(self) != type(other):
            print(f"Types differ {type(self)} != {type(other)}")
            return False
        if self.ssh_node != other.ssh_node:
            print(f"ssh_nodes differ {self.ssh_node} != {other.ssh_node}")

        return True

    @abstractmethod
    def submit_cmd(self,
                   cmd: typing.List[str],
                   name: str,
                   outdir: typing.Optional[pathlib.Path] = None,
                   rundir: typing.Optional[pathlib.Path] = None,
                   run_code: typing.Optional[str] = None,
                   hold: typing.List[str] | str | bool = False,
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
        :param rundir: Directory where job will run.
           If None will run in current working dir when command is run.
        :param run_code: If provided, code to use to run the job.
        :param hold: If provided as a string or list of strings,
        this (these) jobids will need to successfully run before cmd is ran.
          If provided as a bool then job will held if hold_jid is True.
          If False no hold will be done
        :param n_cores: No of cores to use.
        :param n_tasks: no of tasks to run. Will submit an array job.
        :return: the command to be submitted.
        """
        pass

    @abstractmethod
    def release_job(self, jobid: str) -> typing.List[str]:
        """
        cmd to release_job a job
        :param jobid: jobid to release_job
        :return: cmd to release_job job
        """
        pass

    @abstractmethod
    def kill_job(self, jobid: str) -> typing.List[str]:
        """
        cmd to kill a job
        :param jobid: jobid to kill
        :return: cmd to kill a job.
        """
        pass

    @abstractmethod
    def job_id(self, output: str) -> str:
        """
        Extract jobid from output of a qsub or similar command
        :param output: output from submission (or similar)
        :return: jobid as a string.
        """
        pass

    @abstractmethod
    def job_status(self, job_id: str, full_output: bool = False) -> str:
        """
        Return the status of a job.
        :param job_id: job id for status to be checked.
        :param full_output If True will return (raw) full output
        :return: One of 'notFound','Running','Held','Error','Queuing','Suspended',f'Failed {status} or full output
        """
        pass

    @abstractmethod
    def my_job_id(self) -> str :
        """
        Return job id of calling process.
        :return: the job id of the process. Fails if not found
        """
        pass
    # end of abstract functions


class sge_engine(abstractEngine):
    """
    Engine class for SGE
    """

    def submit_cmd(self, cmd: typing.List, name: str,
                   outdir: typing.Optional[pathlib.Path] = None,
                   rundir: typing.Optional[pathlib.Path] = None,
                   run_code: typing.Optional[str] = None,
                   hold: typing.List[str] | str | bool = False,
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
        :param rundir: Directory where job will be ran. If None will run in current working dir when command is run.
        :param run_code: If provided, code to use to run the job
        :param hold: If provided as a string or list of strings,
        this (these) jobids will need to successfully run before cmd is ran.
          If provided as a bool then job will held if hold_jid is True. If False no hold will be done
        :param n_cores: No of cores to use.
        :param n_tasks: If not None the size of the task array to be ran.
        :return: the command to be submitted.
        """

        if outdir is None:
            outdir = pathlib.Path.cwd() / 'output'
            my_logger.debug(f"Set outdir to {outdir}")

        submit_cmd = ['qsub', '-l', f'h_vmem={mem}M', '-l', f'h_rt={time}',
                      '-V',
                      "-e", str(outdir) + "/", "-o", str(outdir) + "/",
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

        if isinstance(hold, bool) and hold:
            submit_cmd += ['-h']  # If False nothing will be held
        if isinstance(hold, str):
            submit_cmd += ['-hold_jid', hold]  # hold it on something
        if isinstance(hold, list) and (len(hold) > 0):  # need a non-empty list.
            submit_cmd += ['-hold_jid', ",".join(hold)]  # hold it on multiple jobs
        if n_cores > 1:  # more than 1 core wanted.
            submit_cmd += ['-pe ', f'mpi {n_cores}']  # ask for mpi env.
        if n_tasks is not None:  # want to run a task array
            submit_cmd += ['-t', f'1:{n_tasks}']
        submit_cmd += cmd
        submit_cmd = self.connect_fn(submit_cmd, rundir=rundir)
        return submit_cmd

    def release_job(self, jobid: str) -> typing.List[str]:
        """
        SGE cmd to release_job a job
        :param jobid: jobid to release_job
        :return: cmd to release_job job
        """
        cmd = ['qrls', jobid]  # qrls: Command to release a job

        cmd = self.connect_fn(cmd)
        return cmd

    def kill_job(self, jobid: str) -> typing.List[str]:
        """
        SGE cmd to kill a job
        :param jobid: jobid to kill
        :return: cmd to kill a job.
        """
        cmd = ['qdel', jobid]  # qdel: command to delete a job

        cmd = self.connect_fn(cmd)
        return cmd

    def job_id(self, output: str) -> str:
        """
        Extract jobid from output of a qsub or similar command
        :param output: output from submission (or similar)
        :return: jobid as a string.
        """

        return output.split()[2].split('.')[0]

    def job_status(self, job_id: str, full_output: bool = False) -> str:
        """
        Return the status of a job. Tuple will contain strings. Needs to actually run.
        Could be turned into a gen command then parse the output.
        :param job_id: job id for status to be checked.
        :param full_output If True will return (raw) full output
        :return: One of 'Running','Held','Error','Suspended','Queuing',"Failed"
        """
        cmd = [f'qstat | grep {job_id}']
        cmd = self.connect_fn(cmd)  #
        cmd = [os.path.expandvars(c) for c in cmd]
        shell= False
        if len(cmd) == 1: # original output so need shell. ssh seems to run shell!
            shell=True
        result = subprocess.run(cmd, capture_output=True, text=True,shell=shell)
        if result.returncode == 1:
            return "notFound"
        result.check_returncode()
        status = result.stdout.split()[4]
        if status[0] == 'E':
            return 'Error'
        elif status[0] == 'r':
            return 'Running'
        elif status[0] in ['S', 's']:
            return "Suspended"
        elif status == 'hqw':
            return "Held"
        elif status == 'qw':
            return "Queuing"
        else:
            my_logger.warning(f"Got unknown status {status} from {result}")

        return f"Failed {status}"

    def my_job_id(self) -> str:
        """

        :return: the job id of the process. Uses $JOB_ID env var to get it.
        """
        try:
            id = os.environ["JOB_ID"]
        except KeyError:
            raise ValueError("No JOB_ID environment variable found. Are you running in SGE env? ")
        return id


class slurm_engine(abstractEngine):
    """
    Engine for SLURM
    """

    def submit_cmd(self, cmd: typing.List, name: str,
                   outdir: typing.Optional[pathlib.Path] = None,
                   rundir: typing.Optional[pathlib.Path] = None,
                   run_code: typing.Optional[str] = None,
                   hold: typing.List[str] | str | bool = False,
                   time: int = 1800,
                   mem: int = 4000,
                   n_cores: int = 1,
                   n_tasks: typing.Optional[int] = None):
        """
        Function to submit command to SLURM

        :param cmd: list of commands to run.
        :param name: name of job
        :param outdir: Directory where output will be put. If None then will be cwd/output
        :param rundir: Directory where job will be ran. If None will run in current working dir when command ran.
        :param run_code: If provided, code to use to run the job
        :param hold: If provided as a string, this jobid will need to successfully run before cmd is ran.
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
            my_logger.debug(f"Set outdir to {outdir}")
        submit_cmd = ['yhbatch', '-n',f'1',#f'--mem={mem}', f'--mincpus={n_cores}', f'--time={time}',
                      '--output', f'{outdir}/%x_%A_%a.out', '--error', f'{outdir}/%x_%A_%a.err',
                      '-J', name]  #liangwj
        # --mem={mem}: Request mem mbytes  of memory per job
        # --mincpus={n_cores}: Request at least n_cores CPU per job
        # --time={time}: Request a maximum run time of time  minutes per job
        # -J name: Name of the job
        if run_code is not None:
            submit_cmd += ['-A', run_code]
        if rundir is not None:  # by default Slrum runs in cwd.
            submit_cmd += ['-D', str(rundir)]  # should be abs path.
        if isinstance(hold, bool) and hold:  # got a book and its True. Just hold the job
            submit_cmd += ['-H']  # Hold it
        if isinstance(hold, str):  # String -- hold on one job
            submit_cmd += f'--dependency=afterok:{hold}'
        if isinstance(hold, list) and (len(hold) > 0):
            # hold on multiple jobs (empty list means nothing to be held)
            submit_cmd += [f"--dependency=afterok:" + ":".join(hold)]  #liangwj
        if n_tasks is not None:
            submit_cmd += ['-a', f'1-{n_tasks}']  # -a =  task array
        submit_cmd += cmd

        submit_cmd = self.connect_fn(submit_cmd, rundir=rundir)
        return submit_cmd

    def release_job(self, jobid: str) -> typing.List:
        """
        SLURM cmd to release_job a job
        :param jobid: The jobid of the job to be released
        :return: a list of things that can be ran!
        """
        cmd = ['yhcontrol', 'release', jobid]  # Command to release_job a job  #liangwj

        cmd = self.connect_fn(cmd)
        return cmd

    def kill_job(self, jobid: str) -> typing.List:
        """
        Slurm cmd to kill a job
        :param jobid: The jobid to kill
        :return: command to be ran (a list)
        """
        cmd = ['yhcancel', jobid]  #liangwj

        cmd = self.connect_fn(cmd)
        return cmd

    def job_id(self, output: str) -> str:
        """
        Extract jobid from output of sbatch or similar command
        :param output: string of output
        :return: jobid as a string.
        """

        return output.split()[3].split('.')[0]  #liangwj

    def job_status(self, job_id: str, full_output: bool = False) -> str:
        """
        Return the status of a job. Tuple will contain strings
        :param job_id: job id for status to be checked.
        :param full_output If True will return (raw) full output
        :return: One of 'Running','Held','Error','Suspended','Queuing',"Failed","NotFound"
        """

        cmd = ["squeue", f"--job_ids={job_id}", '--long']  # get the output in long form for specified job.
        if not full_output:
            cmd += ['--noheader']

        self.connect_fn(cmd)
        result = subprocess.check_output(cmd, text=True)
        if full_output:
            return result

        codes = dict(
            PENDING='Queueing', RUNNING='Running', SUSPENDED='Suspended', CANCELLED='Failed', COMPLETING='Running',
            COMPLETED='Finished', CONFIGURING='Running', FAILED='Failed', TIMEOUT='Failed', PREEMPTED='Queuing',
            NODE_FAIL='Failed', SPECIAL_EXIT='Failed')

        # check for job not present (either because it ran or was never there)
        # work out how to parse result.
        if len(result) == 0:  # nothing found
            return "NotFound"
        status = result.split()[4]  #liangwj
        if status.startswith("PENDING"):
            reason = result.split()[8].split("(")[1].replace(")","")    #status.split("(")[1].replace(")", "")  #liangwj
            if reason in ['JobHeldUser', 'JobHeldAdmin', "Dependency"]:
                return "Held"
            else:
                return "Queueing"
        else:
            return_code = codes[status]
            return return_code

    def my_job_id(self) -> str:
        """
        Return the process job id.
        :return: the job id of the process. Uses SLURM_JOB_ID env var to get it.
        """
        try:
            id = os.environ["SLURM_JOB_ID"]
        except KeyError:
            raise ValueError("No SLURM_JOB_ID environment variable found. Are you running in SLURM env? ")
        return id

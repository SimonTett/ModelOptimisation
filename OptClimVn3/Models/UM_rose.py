# Class to support Unified Model running in Rose.
# This version  has only been tested on archer2.
# For this configuration need to set various things in run_info block:
# 1) remote_machine: puma2
# 2) runEnvSetup : path to environment setup script on archer2
# 3) extra_args : any extra args to pass to optclim scripts.
# 4) Set up local_root_dir  to have sensible run names and archiving file choices.
# If running on other platforms then will need to change run_info and probably make some code changes.
# Things marked ARCHER2 are specific to ARCHER2. Generalise if on another platform.

# User will need to do some work to get UM model running in cylc working.
# 1) Reduce diagnostic output to minimum needed for optclim as data piles up on work dirs
# 2) Set model up to run for desired period for optimisation.
# 3) Turn of pruning as that removes data on archer2 that is needed for post-processing.
#  **Might** be fixed by changes to archive_root_path
# 4) Have archiving on -- that puts data in archive_root_path which gets set to model_dir/output
# 5) If archiving to JASMIN then set transfer_dir to something sensible.T
# 6) Test your case works.

# Possible further changes to creating  a UM_rose model
# Optional stuff
# 4) Modify Model.Model so can just read an existing model config which could be modified but no automatic changes.
#    In particular, it does not have model_dir and all the other stuff just the config.
#      Maybe a new super class ModelConfig which takes all the config related stuff then Model inherits from that. But will need to know what the configurations are...
# 5) Have a clean method which runs cylc clean. That could form part of the post_process step -- run after successfully running post process.
#     Perhaps a SubmitStudy method clean which runs clean on all models in the study. [Clean being model dependent]

"""
Current problems:
  transfer to jasmin is giving files something like: /gws/nopw/j04/terrafirma/tetts/optclim/test_run2/test_run2/tc000
  transfer_dir='/gws/nopw/j04/terrafirma/tetts/optclim/test_run2'

  My guess is that my evil hack of naming the run root_dir/tc000. Will try not changing transfer_dir to see if that helps.

  Data is being copied, for example, to output/test_run2/tc003/YYYYetc -- ideally would like data to go to output/YYYYetc
  Looks like run name is being used. To much of a faff to change now. Will deal with later if needed.

  jasmin transfer fails (because credentials  expired) then very hard to restart it.
    Auto restart does not work even when reauthorised..

  And then the post-processing job never gets released for reasons... Looks like housekeeping never got run...
  Soln: model.succeeded() -- which will release the job.

  Failures from make_ancil -- increase time in reference job.

"""



import fileinput
import functools
import logging
import os
import re
import tarfile
import typing
import shutil
import subprocess
import numpy as np

import cftime
import metomi.isodatetime.exceptions
import metomi.isodatetime.parsers as parse
import scipy.stats

import genericLib

from ModelBaseClass import register_param, type_param_fn  #
from Model import Model
import pathlib

from namelist_var import NamelistVar




my_logger = logging.getLogger(f"OPTCLIM.{__name__}")  # have this anywhere you want logging

type_create_script = typing.Literal["submit", "continue", "clean"] # type def for various flavours of _create_script.
class UM_rose(Model):
    """
    Class to support the Unified model running in ROSE. See specialised classes for cylc7 & cylc8 support.
    The complication is that this the UM uses cylc.
    Main changes come from including optclim.rc in suite.rc

    This class adds suite_name to the class attributes. This is where the suite name gets written.
    If not set

    Initialisation uses the following values from run_info which become variables -- see files in UM_rose_params :
    For the following variables values of None mean the suite is not modified.
        - runModelTime:str -- time as iso duration for model to run for.
        - runUser:str -- username to run the suite as.
        - runCode:str -- job code/account to run suite with.
        - prebuild:str -- path (on remote machine if remote_machine used) to prebuild. Means much faster compilation


        - transfer_dir :str -- path to archive directory on jasmin.


    - OPTCLIM_ARGS :str -- arguments to pass to optclim scripts which get passed to set_status_script. Default ''
    - OPTCLIM_SET_STATUS_SCRIPT -- path to script which sets the status. (Uses self.set_status_script)
    - runEnvSetup:str -- path to the environment setup script.
    - MODEL_CONFIG will be set to self.cache_path


    """
    # additional attributes to the model class.

    suite_name: typing.Optional[str]  # name of the suite


    # test if we are on Archer by calling hostname -A and that stdout contains archer2.ac.uk
    stat = subprocess.run(['hostname', '-A'], capture_output=True, text=True)
    if not ((stat.returncode == 0) and 'archer2.ac.uk' in stat.stdout):
        my_logger.warning('Not running on archer2. This code many need re-writing to work on other platforms')
        archer2 = False

    else:
        archer2 = True # running on archer2


    # Get the user ID
    user_id = os.environ.get('USER') or os.environ.get('USERNAME') # Needed?
    # some default values
    model_data_dir = pathlib.Path('output') # where model data gets put
    script_dir = pathlib.Path('scripts') # where scripts are stored.
    suite_dir = pathlib.Path('workflow') # where the suite configuration is stored.
    # default values per machine
    default_archer2_remote = dict(remote_machine='puma2')

    default_values = dict(archer2=default_archer2_remote)
    # values below to be set by sub-classes.
    suite_file_name = None  # name of the suite file
    include_file_name = None  # include file

    def __init__(self, *args, **kwargs):
        """
        Init the UM_rose instance. Calls the super-class init method.
         Then set ups suite_name
        :param args: positional args. Passed through to super-class
        :param kwargs: kwargs -- passed through to super-class.
           if kwargs contains suite_name that value will be used to set suite_name.

           If suite_name is not None then the configurations will point to that directory
           if kwargs contains model_data_dir that will be used to set model_data_dir
        Sets up the following no_key variables:
        runModelTime, runUser, runCode,OPTCLIM_ARGS,runEnvSetup,prebuild from run_info
        Sets up MODEL_CONFIG (path to the model config), OPTCLIM_SET_STATUS_SCRIPT (path to script that updates model status),
         and OPTCLIM_SIMULATED_OBS_PATH (path to where simulated obs are expected to be)
        and sets submit and continue scripts to be in config_dir
        """

        if 'config_dir' not in kwargs: # add config_dir if not present with a default value of workflow
            kwargs['config_dir'] =self.suite_dir
        suite_name = kwargs.pop('suite_name', None) # name of the suite. Popped as not used by super class.

        super().__init__(*args, **kwargs)  # call the super-class init method.
        self.set_suite_name(suite_name)


        # Set archer_archive_dir to be model_dir/model_data_dir as that is where we copy data to.
        archive_dir = (self.model_dir / self.model_data_dir).as_posix() # posix string for the UM
        self.parameters_no_key['archer_archive_dir'] = archive_dir # add to parameters_no_key
        my_logger.debug(f'Set archer_archive_dir to {archive_dir}')
        # modify parameters_no_key to include runModelTime, runUser, runCode, OPTCLIM_ARGS, runEnvSetup,prebuild,transfer_dir if set.
        # Those parameters do not contribute towards the unique key used to identify the model so go in parameters_no_key
        if self.run_info is not None:  # need to test for None as reloading of config gives us None.
            for key in ['runModelTime', 'runUser', 'runCode','prebuild',
                        'OPTCLIM_ARGS', 'runEnvSetup','transfer_dir']:
                if self.run_info.get(key) is not None:
                    # Get None if either null in the original  json config or not present
                    self.parameters_no_key[key] = self.run_info[key]
            local_root_dir = self.expand(self.run_info.get('local_root_dir'))
            # deal with transfer_dir -- for jasmin archiving
            if 'transfer_dir' in self.parameters_no_key:
                transfer_dir = self.expand(self.parameters_no_key['transfer_dir'],local=False) # remote path
                #transfer_dir = self.new_path(self.model_dir.parent,transfer_dir,root_dir=local_root_dir)
                # transfer_dir ends with model_dir.name and when the transfer to jasmin happens the model name is used so drop it.
                self.parameters_no_key['transfer_dir'] = transfer_dir.as_posix() # posix string for the UM
                my_logger.debug(f'Transfer dir: {self.parameters_no_key["transfer_dir"]}')

            # deal with prebuild
            if 'prebuild' in self.parameters_no_key:
                prebuild = self.parameters_no_key['prebuild']
                self.parameters_no_key['prebuild'] = self.expand(prebuild,local=False).as_posix()
                # expand any vars, make sure it looks like a purePath and convert to a posix format string.
                my_logger.debug(f'Prebuild dir: {self.parameters_no_key["prebuild"]}')


            # set RUNID to False and RUN_NAME to first 5 characters of the name.
            if self.name is not None:
                self.parameters_no_key['RUNID'] = False
                self.parameters_no_key['RUN_NAME'] = self.name[:5]  # first 5 characters of the name.

            # Deal with runEnvSetup
            run_env_setup = self.expand(self.parameters_no_key.get('runEnvSetup'))
            if run_env_setup is not None:
                # check run_env_setup actually exists.
                if not run_env_setup.is_file():
                    raise ValueError(f'run_env_setup {run_env_setup} does not exist.')
                self.parameters_no_key['runEnvSetup'] = run_env_setup.as_posix()  # need to convert to string.


            # deal with OPTCLIM_ARGS -- giving it a default value of ''
            self.parameters_no_key['OPTCLIM_ARGS'] = self.parameters_no_key.get('OPTCLIM_ARGS', '')
            # setup OPTCLIMTOP
            self.parameters_no_key['OPTCLIMTOP'] = str(genericLib.expand('$OPTCLIMTOP'))  # setup OPTCLIMTOP

        # set up MODEL_CONFIG to point to the configuration. TODO -- change to OPTCLIM_MODEL_CONFIG
        if self.config_path is not None:
            self.parameters_no_key['MODEL_CONFIG'] = self.config_path.as_posix()
            my_logger.debug(f"Set MODEL_CONFIG to {self.parameters_no_key['MODEL_CONFIG']}")

        if (self.model_dir is not None) and (self._post_process_output is not None): # where the simulated obs are expected to be.
            # some apps want this as the workflow makes the simulated obs and want to copy it into this directory.
            # this means a "null" post-processing code.
            self.parameters_no_key['OPTCLIM_SIMULATED_OBS_PATH'] = (self.model_dir/self._post_process_output).as_posix()
            my_logger.debug(f"Set OPTCLIM_SIMULATED_OBS_PATH to {self.parameters_no_key['OPTCLIM_SIMULATED_OBS_PATH']}")

        # set up OPTCLIM_SET_STATUS_SCRIPT
        if self.set_status_script is not None:
            self.parameters_no_key['OPTCLIM_SET_STATUS_SCRIPT'] = self.set_status_script.as_posix()
            my_logger.debug(f'Set OPTCLIM_SET_STATUS_SCRIPT to {str(self.set_status_script)}')
        # set up paths to the various scripts we want
        script_dir = self.model_dir/'scripts'
        self.submit_script = script_dir / 'submit_script.sh'  # script to run on puma2 to submit the job.
        self.continue_script = script_dir / 'continue_script.sh'  # probably don't need this for now. There if have an error.
        self.clean_script = script_dir / 'clean_script.sh'  # script to clean the suite.

    def set_suite_name(self, suite_name: typing.Optional[str]=None) -> None:
        """
        Set the suite name.
        :param suite_name: name of the suite.
         If not provided, it is derived from self.name (an X is prepended if the name starts with a character not allowed as start char by cylc).
          If local_root_dir is set in run_info and model_dir is relative to local_root_dir, the relative path from local_root_dir to the parent of model_dir is prepended to the suite name.
          otherwise self.model_dir.parent/name is appended to the modified form of self.name
        :return: None
        """
        if suite_name is None: # work out suite_name
            suite_name = self.name  # start with name
            if re.match(r'^[.,\-\d]', suite_name): # starts with . , - or digit. Make it start with X instead!
                my_logger.debug('Adding X to name')
                suite_name= 'X' + suite_name  # add X to start of name if starts with non-alpha char.

            local_root_dir = self.expand(self.run_info.get('local_root_dir'))

            if (local_root_dir is not None)  and (
                parent := self.model_dir.parent).is_relative_to(local_root_dir):
                part = parent.relative_to(local_root_dir)
                suite_name= (part/suite_name).as_posix()
                my_logger.debug(f'Added {part} from {local_root_dir} giving suite_name =  {suite_name}')
            elif local_root_dir is not None:
                raise ValueError(f"model_dir: {self.model_dir.parent} is not relative to run_info.local_root_dir: {local_root_dir}")
            else:
                suite_name = self.model_dir.parent.name + '/' + suite_name

        self.suite_name = suite_name

    @staticmethod
    def replace_file(file: pathlib.Path,
                     match: str,
                     replacement: str,
                     backup_ext: typing.Optional[str] = None
                     ) -> typing.Optional[tuple[pathlib.Path, int]]:
        """
        replace match in file with replacement. Note match & replacement  are regular expressions

        :param file: path to file
        :param match: RE to match
        :param replacement: what to replace match with. uses re.sub
        :param backup_ext: extension to use for backup file.
          If None or backup file already exists then no backup file is created.
        :return: file, and number of changes if any changes made. Or None if no changes made.
        """

        # step 1 check if match is in the file.
        found_match = False
        with file.open('rt') as f:
            while line := f.readline():
                if re.match(match, line):
                    found_match = True
                    my_logger.debug(f'Found match in {file} for {match} line:\n{line}')
                    break  # no need to read more lines.

        # if no match found then return None
        if not found_match:
            my_logger.debug(f'No match found in {file} for {match}')
            return None

        # got some matches so process this file

        count_match = 0
        with fileinput.input(files=file, inplace=True,
                             backup=backup_ext) as f:  # inplace fileinput uses temp backup file.
            for line in f:
                repl_line = re.sub(match, replacement, line)
                if repl_line != line:  # any changes?
                    count_match += 1
                print(repl_line, end='')
        return file, count_match




    def modify_model(self):
        """
        UM rose specific version of modify model,
        Does the following (after calling the superclass method
        Copies the optclim suite apps into the suite_dir and modifies the suite_file_name
        Also generates  any necessary scripts and sets archer_archive_dir to model_dir/model_data_dir

        """
        super().modify_model()  # call the base class method.
        # now do the specific stuff...
        self.copy_suite_apps()  # copy the optclim specific apps to the suite dir.
        self.update_suite_rc()  # update the suite.rc/flow.cylc file
        # need to generate scripts
        # done here as super class instantiate cleans directory before doing anything else and then checks.
        # so need to call this in modify_model
        args = self.parameters_no_key.get('OPTCLIM_ARGS')
        self._create_script('submit', args=args)
        self._create_script('continue', args=args)
        self._create_script('clean',args=args)  # create the clean script.





    def _create_script(self,
                       script_type: type_create_script,
                       args: typing.Optional[str] = None):
        """
        Create a script to run on puma2.
        :param script_type: type of script. 'submit' or 'continue'.
        :param args: any arguments to pass to the script.
        :return: nothing.
        """
        raise ValueError('Do not call this version. Call the cylc specific version not this one ')

    def update_suite_rc(self):
        """
        Update the cylc/rose suite.rc to include OptClim tasks.
        """
        suite_file = self.config_dir / self.suite_file_name
        genericLib.backup_file(suite_file, ext='.bak', create='copy')  # make a backup of the suite file.
        with suite_file.open('a') as suite_rc:
            if self.include_file_name is None:
                raise ValueError('Set self.include_file_name')
            inc = self.config_dir/self.include_file_name
            if not inc.exists():
                raise FileNotFoundError(f'file {inc} does not exist')
            
            suite_rc.write(f'\n\n%include {self.include_file_name}\n')

        if self.run_info.get('use_scratch', False):
            raise ValueError('use_scratch no longer supported. Modify your reference manually.')



    def copy_suite_apps(self):
        """Copy OptClim specific apps from the reference directory to the model directory"""
        optclim_tasks_root = pathlib.Path(__file__).parent / 'um_rose_files/optclim_tasks'
        shutil.copytree(optclim_tasks_root, self.config_dir, dirs_exist_ok=True)



    def check(self) -> bool:

        """
        Check the model. This is a UM_rose specific version of check.
        Calls the superclass method and then does the following checks:
        If START_TIME, RUN_TARGET and RESUB_TIME exist then
          1) Check they are parsable
          2) Check they are compatible.
             n*RUN_TARGET+START_TIME should be STATRT_TIME+RESUB_TIME.
        Will raise ValueError if any of these fail.
        2) Check that the submit and continue scripts are files.
           if not will raise FileNotFoundError.
        :return: True if the model is valid, False otherwise.
        """
        # handy helper function
        def parse_iso_value(value:str, name:str,
                            parser_obj:parse.DurationParser|parse.TimePointParser):
            """"
            Parse an iso formated data/time or duration string raising helpful error messages.
            :param value: The value to parse.
            :param name: The name of the parameter.
            :param parser_obj: The parser object. Should be a DurationParser for durations and a TimePointParser for datetimes.
            :param expected:
            """
            if isinstance(parser_obj, parse.DurationParser):
                expected = "Should be an ISO duration -- e.g. P1Y"
            elif isinstance(parser_obj, parse.TimePointParser):
                expected = "Should be an ISO time point -- e.g. 2022-01-12"
            else:
                raise ValueError("parser_obs should be a TimePointParser or a DurationParser")
            try:
                return parser_obj.parse(value)
            except metomi.isodatetime.exceptions.ISO8601SyntaxError as err:

                raise ValueError(
                    f"Invalid {name} value {value}. \n"
                    f"Expected {expected}. Original error: {err}"
                ) from err

        if not super().check(): # try super class check first and if that fails then return False.
            return False  # failed so return False.


        ## UM_rose specific checks.
        # 1) Check that START_TIME, RUN_TARGET and RESUB_TIME are compatible. Won't work perfectly for 360 day calendar
        # as then need to deal with 360 day calendar. And only works if have atmos model
        # By converting them to Time Points and Durations we are also checking that
        # strings are valid.
        atmos_time_params =  ['START_TIME','RUN_TARGET','RESUB_TIME'] # time variables
        parser = parse.TimePointParser()
        parser_durn = parse.DurationParser()
        parser_list = [parser,parser_durn,parser_durn]
        times = self.read_params(atmos_time_params,fail=False)
        times = {k:v for k,v in times.items() if v is not None} # remove any none values
        if False:
        if len(times) > 0 : # have atmos times  in workflow.
            # see what calendar is and if it is 360 day raise warning.
            cal = self.calendar()  # get the calendar.
            if cal != 'standard':
                my_logger.warning(f'''
                 model calendar is set to {cal}. This means that START_TIME, RUN_TARGET and RESUB_TIME 
                may not be compatible with it. Please check these values. Fix code to deal with {cal} if get error''')

            start_time, run_target, resub_time = (parse_iso_value(times[n],n,parse_obj) for n,parse_obj in zip(atmos_time_params,parser_list))
            # iterate from start_time  to start_time + run_target.
            # Doing this because months are not the same (second) duration throughout the year...
            end_time = start_time + run_target
            time = start_time
            while time < end_time:
                time += resub_time
            # Now time should be start_time + run_target
            if time != end_time:
                raise ValueError(f'RUN_TARGET {run_target} and RESUB_TIME {resub_time} are not compatible')

        # 2)  check the various scripts we want exist
        for script in [self.submit_script, self.continue_script,self.clean_script]:
            if not (script is None or script.is_file()):
                raise FileNotFoundError(f"{script} is not a file.")
        # 3) Check values of archive_pp and archive_netcdf and warn if they are not True
        for param in ['archive_pp', 'archive_netcdf']:
            if self.read_param(param,raise_error=False) is not True:
                my_logger.warning(f'{param} is not  True.'+\
                                  f' Your data will not be copied into {self.model_dir/self.model_data_dir} and post-processing may fail.')

        return True

    def running(self,jid:typing.Optional[str]='NOJID') -> typing.Optional[str]:
        """
        UM_model version of running. No jid possible for UM as cylc handling all of that.
        :param jid: job id -- if passed in overrides any existing job id.

        """
        my_jid = super().running(jid=jid)  # call the super-class method specifying jid
        return my_jid




    def submit_cmd(self) -> typing.List[str|pathlib.Path]:
        """"
        Generate the submission command. Overrides the super-class version.

        """
        if self.status in ['INSTANTIATED', 'PERTURBED']:  # start again.
            script = self.submit_script
        elif self.status == 'CONTINUE':
            script = self.continue_script
            raise NotImplementedError('Continue not implemented yet')
        else:
            raise ValueError(f"Status {self.status} not expected ")

        cmd = [self.model_dir/script]

        return cmd

    def copyConfig(self, direct: pathlib.Path,
             extra_files: typing.Optional[list[pathlib.Path]] = None,
             update_paths: bool = True) -> "UM_rose":

        """
        Copy method for UM_rose class. Calls the super-class method with extra files
            All files must in self.model_dir
        :param direct: directory where Model  is to be copied. Will be created if it does not exist
        :param extra_files: list of extra files (paths provided relative to self.model_dir) to be copied to new directory.
        :param update_paths -- If True then any path parameters will be updated to reflect new directory structure.
        :return: Copied Model. Will copy only Model config & post process unless extra_files provided.
        """

        files_to_add = [self.script_dir, self.config_dir.relative_to(self.model_dir)]
        # TODO_relative_paths -- eventually make all paths relative to model_dir.
        if extra_files is not None:
            files_to_add += extra_files
        cp_obj = super().copyConfig(direct=direct,extra_files=files_to_add,update_paths=update_paths)
        return cp_obj  # return the copied object.



    # TODO (when needed). Add a delete method to kill the suite and delete the suite_dir
    # Cmd for killing the suite is cylc stop --now SUITE_NAME but needs to run on puma2.
    # Can remove the suite dir on archer2/local machine.
    # Outline of delete method:
    #  Kill running suite: cylc stop --now self.suite_dir.name
    #  Remove the suite dir: shutil.rmtree(self.suite_dir)
    # run the super-class delete method to remove the model_dir.



## end of class definition.
class UM_rose_cylc7(UM_rose):
    """
    Class to support Unified Model running in ROSE cylc7.
    This is a subclass of UM_rose and adds the UM_rose specific parameters.
    """

    # cylc 7 specific methods.
    suite_file_name = 'suite.rc'  # name of the suite file for cylc7.
    include_file_name='optclim.rc' # include file for cylc7

    def _create_script(self,
                       script_type: type_create_script,
                       args: typing.Optional[str] = None) -> None:
        """
        Create a script to run on puma2.
        :param script_type: type of script. 'submit' or 'continue'.
        :param args: any arguments to pass to the script.
        :return: nothing.
        """

        if script_type == 'submit':
            script = self.submit_script
        elif script_type == 'continue':
            script = self.continue_script
        elif script_type == 'clean':
            script = self.clean_script
        else:
            raise ValueError(f'Unknown script type: {script_type}')
        script.parent.mkdir(parents=True, exist_ok=True)  # might need to create directory
        script.unlink(missing_ok=True)  # unlink it if it exists.
        with script.open('wt') as f:
            f.write('#!/bin/bash --login\n')
            cmd = ['rose', 'suite-run']
            if script_type == 'submit':
                cmd += ['--new','--no-gcontrol']
            elif script_type == 'continue':
                cmd+= ['--restart ','--no-gcontrol']
            elif script_type == 'clean':
                cmd = ['rose', 'suite-clean']
            else:
                raise ValueError(f'Unknown script_type: {script_type}')
            if args:
                cmd.append(args)
            remote_dir = self.remote.get('remote_directory')
            if remote_dir is not None:
                config_dir = remote_dir/self.config_dir # path on remote machine to config dir.
            else:
                config_dir = self.model_dir/self.config_dir # path on local machine to config dir.

            cmd.append(f'-C {config_dir.as_posix()}')  # path to the config dir. rose needs posix path.
            f.write(' '.join(cmd) + '\n')
        # set permissions to be executable
        script.chmod(0o755)



class UM_rose_cylc8(UM_rose):
    """
    Class to support Unified Model running in ROSE cylc8.
    This is a subclass of UM_rose and adds the UM_rose specific parameters.
    """
    suite_file_name = 'flow.cylc'  # name of the suite file for cylc8
    include_file_name = 'optclim_c8.rc'

    # cylc 8 specific methods.


    def _create_script(self,
                       script_type: type_create_script,
                       args: typing.Optional[str] = None) -> None:
        """
        Create a script to run on puma2. This is the cylc8 specific version.
        :param script_type: type of script. 'submit' or 'continue'.
        :param args: any arguments to pass to the script.
        :return: nothing.
        """


        if script_type == 'submit':
            script = self.submit_script
        elif script_type == 'continue':
            script = self.continue_script
        elif script_type == 'clean':
            script = self.clean_script
        else:
            raise ValueError(f'Unknown script type: {script_type}')
        script.parent.mkdir(parents=True, exist_ok=True)  # might need to create directory
        script.unlink(missing_ok=True)  # unlink it if it exists.
        remote_dir = self.remote.get('remote_model_dir')
        if remote_dir is not None:
            config_directory = self.new_path(self.config_dir,remote_dir)
            if not config_directory.is_absolute(): # need abs path
                config_directory = pathlib.PurePath('~')/config_directory
        else:
            # this code unlikely to be well tested as on archer2 run cylc from puma!
            config_directory = self.config_dir
            if not config_directory.is_absolute(): # need abs path
                config_directory = pathlib.Path.cwd()/config_directory

        with script.open('wt') as f:
            f.write('#!/bin/bash --login\n')
            f.write('export CYLC_VERSION=8\n') # make sure in cycl8
            if script_type == 'clean':
                f.write(f'cylc stop --now --now --max-polls=100 {self.suite_name}\n')
                f.write(f'cylc clean --yes {self.suite_name}\n')


            else:
                cmd = ['cylc']
                if script_type == 'submit':
                    cmd += ['vip','--no-run-name']
                elif script_type == 'continue':
                    cmd += ['play','--no-run-name'] # might need a release as well.
                else:
                    raise ValueError(f'Unknown script_type {script_type}')
                if args:
                    cmd.append(args)
                cmd.append(config_directory.as_posix())  # path to the suite dir.
                cmd.append(f'--workflow-name={self.suite_name}')
                f.write(' '.join(cmd) + '\n')
        # set permissions to be executable
        script.chmod(0o755)
        return





class UKESM1_params(Model):
    """
    Class to provide functions to set/get UKESM1 specific parameters.
    Inherit in the various UKESM_1_1 classes as does not depend on cylc version.
    """

    # e UKESM1_1 specific parameters which overwrite any parameters in UM_rose
    # The functions are *mainly* for handing JULES nameslists.
    # JULES in UKESM1_1 is quite different from JULES in HadGEM-GAX.
    # UKESM1.1 has 13 Plant Functional Types (PFTs)
    # 101, 102, 103, 201, 202, 3, 301, 302, 4, 401, 402, 501, 502
    # There are also non vegetated sfc types.
    #  6, 7, 8, 9, 901-910
    # Land surface types :
    #       1 - Broadleaf Tree
    #     101 - Broadleaf Tree deciduous
    #     102 - Broadleaf Tree evergreen tropical
    #     103 - Broadleaf Tree evergreen temperate
    #       2 - Needleleaf Tree
    #     201 - Needleleaf Tree deciduous
    #     202 - Needleleaf Tree evergreen
    #       3 - C3 Grass
    #     301 - C3 Crop
    #     302 - C3 Pasture
    #       4 - C4 Grass
    #     401 - C4 Crop
    #     402 - C4 Pasture
    #       5 - Shrub
    #     501 - Shrub deciduous
    #     502 - Shrub evergreen
    #       6 - Urban
    #     601 - Urban roof
    #     602 - Urban canyon
    #       7 - Water
    #       8 - Soil
    #       9 - Ice
    # 901-925 - Glacier/Icesheet model ice surface on elevation classes(1-25)
    # 926-950 - Glacier/Icesheet model rock surface on elevation classes(1-25)

    # variable containing PFT names
    PFT_names = ['BLT_d','BLT_eg_trop','BLT_eg_temp','NLT_d','NLT_eg','C3_grass','C3_crop', 'C3_pasture',
                 'C4_grass','C4_crop','C4_pasture','Shrub_d','Shrub_eg']


    # service function for parameters that don't do anything!
    def no_param(self,
                 value: typing.Optional[float] = None,
                 transform: bool = True,
                 param:str = 'UNKNOWN') -> typing.Optional[float]:
        """
        Function  for latent variable parameters. Those are used in the 'main' function
        :param value: value to set. If None then return the value from the parameters
        :param transform: For compatibility with other parameters. Doesn't do anything.

        :return: None or value from the parameters.
        """
        if value is None:
            # if value is None then read the value from the namelist.
            value = self.parameters.get(param)
            return value
        else:
            return None # fn does not do anything so return None.
        
    #  service function to scale parameters.
    def scale_param(self,
                    value: typing.Optional[float]=None,
                    default: typing.Optional[list[float]] = None,
                    transform: bool = True,
                    nl: typing.Optional[NamelistVar] = None) -> type_param_fn:
        """
        Scale a parameter by a scale factor.
        :param nl: namelist information to use.
        :param value: value to use. Scaling is value/default[0]. If value is None then read the value from the namelist.
        :param default: default values
        :param transform: if True then transform the readin value when value is None.
        :return: list of values default scaled by value/default[0] or just retrieve the value from the namelist.
        """
        if value is None:  # if value is None then read the value from the namelist.
            value = self.read_nl_value(nl)
            if transform:
                value = value[0]

            return value

        scale = value / default[0]
        result = [scale * v for v in default]  # scale the default values.
        return [(nl, result)]

    def fn_param(self,
                   value: typing.Optional[float] = None,
                   transform: bool = True,
                   transform_values: typing.Optional[dict[NamelistVar,typing.Optional[typing.Callable]]] = None
                    ) -> type_param_fn:
        """
        Base fn to handle parameters which can be transformed.
        :param value: value to set. If None then read the value from the namelist.
        :param transform: transform the readin value when value is None if transform is True. Otherwise, return raw values
        :param transform_values -- dict of namelist variables and functions to transform the values. None means no transformation.
        :return: nl and values to set or just the value if None.
        """
        first_nl = list(transform_values.keys())[0] # Must have at least one key.
        if value is None and transform:  # invert the value
            transform_fn = transform_values[first_nl]

            value:float = float(self.read_nl_value(first_nl))
            if transform_fn is not None:
                value = transform_fn(value,inverse=True) # invert the value.
            return value
        elif value is None and not transform:
            values =[]
            for nl, fn in transform_values.items():
                nl_value = self.read_nl_value(nl)
                if fn is not None:
                    nl_value = fn(nl_value, inverse=True)
                values += [float(nl_value)]
            return values
        else: # forward stuff...
            result = []
            for nl, fn in transform_values.items():
                if fn is not None:
                    value = fn(value)
                result += [(nl,value)]  # just return the value.
            return result  # list of tuples (NamelistVar, value) to set.


    um_namelist_file = pathlib.Path('app/um/rose-app.conf')  # namelist file for UKESM1 specific parameters.

    def calendar(self) -> str:
        """
        Function to get calendar string for model. Setting not implemented as not needed and likely has big ramifications.
        Do it via the user interface!
        : return: a string which should be compatible with cftime.datetime that gives the calendar used by the model.
        """
        nl_cal = NamelistVar('um_rose', self.um_namelist_file, 'namelist:nlstcall', 'lcal360')
        lcal360 = self.read_nl_value(nl_cal, raise_error=False)
        if lcal360 is None:
            my_logger.warning('lcal360 not set. Assuming standard calendar. If running on 360 day calendar then set lcal360 to True')
            cal = 'standard'
        elif lcal360:
            my_logger.debug('lcal360 set to True. Using 360 day calendar.')
            cal = '360_day'
        else:
            my_logger.debug('lcal360 set to False. Using standard calendar.')
            cal = 'standard'

        return cal  # return the calendar string.

    @register_param("ensembleMember")
    def ens_member(self,
                   ens_member: typing.Optional[int],
                   transform:bool = True) -> typing.Union[list[tuple[NamelistVar,int]],int]:
        """
        :param ens_member: ensemble member. The ensemble member to set. If None then read the value from the namelist.
        :param transform:  Does nothing and present for compatibility with other parameters.
        :return: [(nl,int)] or ensemble member value if None.
        """
        nl = NamelistVar('um_rose', self.um_namelist_file, 'env', 'ENS_MEMBER')
        inverse = (ens_member is None)
        if inverse:
            ens_member:int = self.read_nl_value(nl, raise_error=False)
            return ens_member
        # otherwise set the ensemble member.
        return [(nl, ens_member)]

    @register_param('START_TIME')
    def start_time(self, value: typing.Optional[str]=None,
                   transform: bool = True) -> type_param_fn:
        """
        Fn to get or set the start_time for the model. Complex as need to check 360 day calendar and if overriding basis
        :param value: value to set START_TIME to as an ISO string. No checking is done.
           If wrong model may likely crash...
           If None then will return value from model.
        :param transform: if True if read in data then will transform the value to ISO8601 string.
        :return: ISO8601 string  or list of namelist info & values to set it to.

        """

        import re

        def pad_iso_datetime(iso_str):
            # thanks tochat gpt
            # utility fn to pad a ISO datetime string to ensure it has the correct format to pass to cftime.datetime.strptime
            # Match year, optional month, day, time, and optional fractional seconds
            match = re.match(
                r'^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?(?:T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?)?$',
                iso_str
            )
            if not match or not match.group(1):
                raise ValueError("ISO date string must specify a 4-digit year and be properly formatted")
            year = match.group(1)
            month = match.group(2) or '01'
            day = match.group(3) or '01'
            hour = match.group(4) or '00'
            minute = match.group(5) or '00'
            second = match.group(6) or '00'
            frac = match.group(7)
            result = f"{year}{month}{day}T{hour}{minute}{second}"
            if frac:
                # Ensure fractional seconds are padded to 6 digits
                result += '.' + frac
            return result

        # Example usage:
        # pad_iso_datetime_strict("2000-01-01T12:30:45.123456")  # OK
        # pad_iso_datetime_strict("2000")  # OK
        # pad_iso_datetime_strict("01-01T12:30:45")  # Raises ValueError


        nl_override = NamelistVar('um_rose', self.um_namelist_file, 'namelist:headers','i_override_date_time')
        override = self.read_nl_value(nl_override, raise_error=False)  # read the override value.
        if override != 2:
            my_logger.warning(f"Override value {override} not 2. Reading or setting it will not work as expected. ")
        nl_start_time =  NamelistVar('um_rose', self.um_namelist_file, 'namelist:headers','new_date_time')
        cal = self.calendar()  # get the calendar string.


        if value is None:
            start_time = self.read_nl_value(nl_start_time)  # read the value from the namelist.
            if transform:
                start_time = cftime.datetime(*start_time, calendar=cal)  # convert to datetime
                start_time = start_time.isoformat()  # and format as ISO time string

            return start_time

        # convert value to a time then a list.
        fmt = pad_iso_datetime(value)
        time = cftime.datetime.strptime(value, format=fmt, calendar=cal)  # convert to datetime
        time = [time.year, time.month, time.day, time.hour, time.minute, time.second]  # what the model wants

        return [(nl_start_time, time),(nl_override,2)] # set the start time and override value to 2.

    @register_param('starticetkelvin') # using this name as this is the variable that will be passed in.
    def cloud_ice(self, start_icet_kelvin: typing.Optional[float] = None,
                  transform: bool = True) -> type_param_fn:
            """
            Set the cloud_ice parameters. This is a UKESM1 specific parameter.
            :param start_icet_kelvin: value to set. If None then read the values from the namelists.
            :param transform: transform the read in values when value is None if transform is True. Otherwise, return raw values
               returns two floats start_icet_kelvin and all_icet_degc_0to1/all_icet_degc when transform is True/False
            :return: nl and values to set or just the values if None.

            From: John Rostron -- 22/7/25:
            allicetdegc is more complicated, and David is probably best to explain that one. In summary my understanding of how it's calculated is:
             The CDF value for starticekelvin is determined (its PDF is a trapezium, with vertices at 234.15, 253.15, 267.15 and 273.15
             This is multiplied by the value for allicetdegc0to1
            The inverse CDF value of this is calculated, using a PDF for allicetdegc which is the same as that for starticekelvin, but with vertices shifted by -273.15.

            SFBT: I think the 234.15 is an error and should be 233.15 as that is the min value allowed.

            """
            nls = [NamelistVar('um_rose', filepath=self.um_namelist_file,
                               namelist='namelist:run_cloud', nl_var=nl_var) for nl_var in ['starticetkelvin', 'allicetdegc']]
            trap_points = [253.15, 267.15]  # points for trapezoidal distribution in K
            loc = 234.15  # location for trapezoidal distribution in K
            scale = 273.15-loc # scale for trapezoidal distribution in K
            dist_start_icet_kelvin = scipy.stats.trapezoid( *[(x-loc)/scale for x in trap_points],scale=scale,loc=loc)
            # trapezoidal distribution for start_icet_kelvin -- temperature in K at which cloud ice starts to form
            dist_all_icet_degc = scipy.stats.trapezoid( *[(x-loc)/scale for x in trap_points],scale=scale,loc=loc-273.15)
            # trapezoidal distribution for all_icet_degc --  temperature in C at which all cloud water is ice.
            # Difference between the two distributions is that the all_icet_degc is converted from K to C.
            # So just shift the distribution by -273.15 to get the all_icet_degc distribution.

            if start_icet_kelvin is None: # read values from the namelist.
                start_icet_kelvin, all_icet_degc = (float(self.read_nl_value(nl)) for nl in nls)

                if transform:
                    # work out all_icet_degc from start_icet_kelvin and all_icet_degc.
                    # see where values are in the distribution.
                    start_ice_x = dist_start_icet_kelvin.cdf(start_icet_kelvin)
                    all_ice_x = dist_all_icet_degc.cdf(all_icet_degc)  # convert to K for distribution.
                    if all_ice_x > start_ice_x:
                        raise ValueError(f'all_ice_x {all_ice_x} is greater than start_ice_x {start_ice_x}. '
                                         f'Check the values are correct.')


                    if np.abs(start_ice_x) < 1e-5:  # if very close to 0 then set allicetdegc to 0.
                        all_icet_degc_0to1 = 0.0
                    else:
                        all_icet_degc_0to1 = all_ice_x/start_ice_x
                    if not (0.0 <= all_icet_degc_0to1 <= 1.0):
                        raise ValueError(f'allicetdegc0to1 {all_icet_degc_0to1} not in range 0 to 1')
                    result = dict(starticetkelvin=start_icet_kelvin, allicetdegc0to1=all_icet_degc_0to1)  # return the values.
                else:
                    result = dict(starticetkelvin=start_icet_kelvin, allicetdegc=all_icet_degc)
                return result # return the values.

            # values to set. Need to get the latent parameter allicedegc0to1 from the parameters.
            # If we don't have it then need to run the inverse calculation to get it from ref config.
            all_icet_degc_0to1 = self.parameters.get('allicetdegc0to1', None)  # default is None.
            if all_icet_degc_0to1 is None:  # if not set then calculate it.
                all_icet_degc_0to1 = self.cloud_ice( transform=True)['allicetdegc0to1']  # call ourselves  to get the value.
            start_ice_x = dist_start_icet_kelvin.cdf(start_icet_kelvin)  # where in the dist are we?
            all_ice_x = start_ice_x * all_icet_degc_0to1  # where in the dist are we for all_ice?
            all_icet_degc = dist_all_icet_degc.ppf(all_ice_x)  # get the value from the distribution.
            return [(nl, v) for nl, v in zip(nls, [start_icet_kelvin, all_icet_degc])]  # return a list of tuples (NamelistVar, value) to set.


    @register_param('aparam')
    def aerosol_cld(self, aparam: typing.Optional[float], transform: bool = True) -> type_param_fn:
        """
        Set aparam and bparam. bparam depends on aparam and liu_latent.
        :param aparam: value of param to to set. If None then read the value of aparam from the namelist.
        :param transform: transform the readin value when value is None if transform is True. Otherwise, return raw values
        :return: nl and values to set or just the value if None.
        """
        nls = [NamelistVar('um_rose', filepath=self.um_namelist_file,
                         namelist='namelist:run_radiation', nl_var=nl_var) for nl_var in ['aparam', 'bparam']]



        if aparam is None:
            aparam,bparam = (float(self.read_nl_value(nl)) for nl in nls) # read the namelist values
            if transform:
                # Compute liu_latent from aparam & bparam.
                liu_latent = bparam+0.19-0.617*aparam  # inverse from linear regression -- see below.
                result = dict(aparam=aparam,liu_latent=liu_latent)
            else:
                result = dict(aparam=aparam,bparam=bparam)
            return result
        # if we have a value then set the bparam value.
        liu_latent = self.parameters.get('liu_latent') # latent so read from param. If None then just read it.
        if liu_latent is None:
            pvalues = self.aerosol_cld(None) # run ourselves to get value
            liu_latent = pvalues['liu_latent']

        bparam  = -0.19+0.617*aparam+liu_latent # worked out from linear regression on GA8 parameter data.
        return [(nl,v) for nl,v in zip(nls,[aparam, bparam])]  # return a list of tuples (NamelistVar, value) to set.

    @register_param('rho_snow_fresh')
    def rho_snow(self,rho_snow_fresh: typing.Optional[float],
                 transform: bool = True) -> type_param_fn:
        """
        Set rho_snow_fresh & rho_snow_et_crit parameters.  Uses rho_snow_fresh+rho_snow_et_crit_delta to set rho_snow_et_crit.
        :param rho_snow_fresh: value of rho_snow_fresh to set. If None then read the value from the namelist.
        :param transform: transform the readin value when value is None if transform is True. Otherwise, return raw values
        :return: nl and values to set or values of rho_snow_fresh and rho_snow_et_crit_delta if value is None.
        """
        nls = [NamelistVar('um_rose', filepath=self.um_namelist_file, namelist='namelist:jules_snow', nl_var=nl_var)
               for nl_var in ['rho_snow_fresh', 'rho_snow_et_crit']]

        if rho_snow_fresh is None:  # pass None to read_nl_value ie. invert the value.
            result:list[float] =  [float(self.read_nl_value(nl)) for nl in nls]  # read the namelist values
            if transform:
                rho_snow_fresh= result[0]  # want the 1st element of the values.
                rho_snow_et_crit_delta = result[1]-result[0]  # want the 2nd element of the values.
                result = dict(rho_snow_fresh=rho_snow_fresh,rho_snow_et_crit_delta=rho_snow_et_crit_delta)  # return the rho_snow_fresh and rho_snow_et_crit_delta values.
            else:
                result=dict(rho_snow_fresh=result[0], rho_snow_et_crit=result[1])
            return result
        # if we have a value then set the rho_snow_et_crit value.
        rho_snow_et_crit_delta = self.parameters.get('rho_snow_et_crit_delta', 41.0)
        # default value is 150.0 - 109.0 = 41.0
        rho_snow_et_crit = rho_snow_fresh - rho_snow_et_crit_delta  # set the rho_snow_et_crit value.
        return [(nl,v) for nl,v in zip(nls,[rho_snow_fresh, rho_snow_et_crit])]  # return a list of tuples (NamelistVar, value) to set.


    @register_param('fsmc_p0_io')
    def fsmc_p0_io(self,
                   value: typing.Optional[float],
                   transform:bool = True) -> type_param_fn:
        """
        Set the fsmc_p0_io parameter. This is a UKESM1 specific parameter.
        :param value: value to set. If None then read the value from the namelist.
        :param transform: transform the readin value when value is None if transform is True. Otherwise, return raw values
        :return: nl and values to set or just the value if None.
        """
        nl = NamelistVar('um_rose', filepath=self.um_namelist_file,
                         namelist='namelist:jules_pftparm', nl_var='fsmc_p0_io')
        if value is None:  # pass None to read_nl_value ie. invert the value.
            value:list[float] = self.read_nl_value(nl)
            if transform:
                value:float = float(value[0])  # want the 1st element of the tuple.
            else:
                value = dict(zip(self.PFT_names,value))
            return value
        result = 13*[value] # UKESM1.1 has 13 PFTs
        return [(nl,result)]

    @register_param('gs_nvg_io')
    def gs_nvg_io(self, value: typing.Optional[float],transform:bool = True) -> type_param_fn:
        """
        Set the gs_nvg_io parameter. This is a UKESM1 specific parameter.
        :param value: value to set. If None then read the value from the namelist.
        :param transform: transform the readin value when value is None if transform is True. Otherwise, return raw values
        :return: nl and values to set or just the value if None.
        """
        nl = NamelistVar('um_rose', filepath=self.um_namelist_file,
                         namelist='namelist:jules_nvegparm', nl_var='gs_nvg_io')
        if value is None:
            value:list[float] = self.read_nl_value(nl)
            if transform:
                value:float = float(value[2])
            else:
                value = dict(zip(self.PFT_names,value))
            return value
        result = [0.00000,0.00000,1.00000e-2]+11*[1.00000e+6]  # UKESM1.1 default value
        result[2] = value  # set the 2nd element (bare soil) to value.
        return [(nl, result)]

    @register_param('n_lai_exposed')
    def n_lai_exposed(self,
                      lai: typing.Optional[float],
                      transform:bool = True) -> type_param_fn:
        """
        Set the number of exposed LAI values. This is a UKESM1 specific parameter.
        :param lai: Exposed LAI values. If None then invert and get value.
        :param transform: if True then transform the readin value when lai is None.
        :return: tuple(NamelistVar, lai) or None if lai is None.
        """

        nl = NamelistVar('um_rose', filepath=self.um_namelist_file,
                         namelist='namelist:jules_snow', nl_var='n_lai_exposed')  # add in default value.
        if lai is None:  # pass None to read_nl_value ie. invert the value.
            value: list[float] = self.read_nl_value(nl)
            if transform:
                value:float = float(value[5])   # want the 6th element corresponding to PFT index = 3 = grass.
            else:
                value = dict(zip(self.PFT_names,value))
            return value

        # from vn13.8 have the following:
        result = [2.0, 2.0, 27.0, 1.0, 2.0]+ 6 * [27.0]+[ 6.0, 6.0] # UKESM1.1 has 13 PFTs
        for indx in range(5, 11):  # set the grass/crop/pasture values.
            result[indx] = lai # set the grass/crop/pasture values to lai.

        return [(nl, result)]

    @register_param('unload_rate_u')
    def unload_rate_u(self, value: typing.Optional[float], transform:bool = True) ->  type_param_fn:
        """
        Set unload_rate_u. This is a UKESM1 specific parameter.
        :param value: unload_rate_u values. If None then invert and get value.
        :param transform : if True then transform the readin value when value is None.
        :return: tuple(NamelistVar, list of values -- only elements 3 & 4 are changed -- Needleleaf trees) or value in config[3] if value is None.
        """

        nl = NamelistVar('um_rose', filepath=self.um_namelist_file,
                         namelist='namelist:jules_snow', nl_var='unload_rate_u')  # add in default value.
        if value is None:  # pass None to read_nl_value i.e. invert the value.
            value: list[float] = self.read_nl_value(nl)
            if transform:
                value:float = float(value[3])  # want the 3rd element of the list = needleleaf trees.
            else:
                value = dict(zip(self.PFT_names,value))
            return value


        result = 13*[0.0]
        result[3:5] = [value, value]  # set the 4th and 5th elements (Needleleaf trees] to value.

        my_logger.warning('unload_rate_u needs verification')
        return [(nl, result)]





# generated fns to add in
# First lot are JULES specific parameters which are all very similar.
# Basically scale based on ratio of parameter to first value in the default list.
nl_jules = functools.partial(NamelistVar, 'um_rose',
                                 filepath=UKESM1_params.um_namelist_file,
                                 namelist='namelist:jules_pftparm')  # jules pftparam namelist var.

defaults=dict( # variable names and default values. parameter names are the same as in the namelist.
    #Note values hare are different from GA7/8/9 defaults.
    # Will need to change the default  values for calibration.
    tupp_io = [43,43,43,26,32,32,32,32,45,45,45,40,36],
    f0_io = [0.875,0.875,0.892,0.875,0.875,0.931,0.931,0.931,0.8,0.8,0.8,0.875,0.875],
    nl0_io =  [0.046,0.046,0.046,0.033,0.033,0.073,0.073,0.073]+5*[0.06],
    rootd_ft_io =  [2,3,2,2,1.8]+8*[0.5],
    dz0v_dh_io = 5*[0.05]+8*[0.1]


)
for name,default in defaults.items():
    UKESM1_params.register_param_with_partial(
            name, # name to register the function as.
            UKESM1_params.scale_param, # using scale parameter function.
            default=default, # default value
            nl=nl_jules(nl_var=name))

# handle the latent parameters which do not directly set a parameter but are used in other functions.
for latent_param  in ['liu_latent', 'allicetdegc0to1', 'rho_snow_et_crit_delta']:
    UKESM1_params.register_param_with_partial(
        latent_param,  # name to register the function as.
        UKESM1_params.no_param,  # using no_param as the base function.
        param=latent_param,         # parameter name to use.

    )

# and on functions where there are dependencies on other parameters.

def cca_md2dp_knob(value: float, inverse: bool = False) -> float:
    """
    Convert cca_md_knob to cca_dp_knob. Just I!
    :param value: value to convert.
    :param inverse: if True then convert from cca_dp_knob to cca_md_knob.
    :return: converted value.
    """
    return value  # fn is identity function!

# function for parameter: cca_md_knob
namelists = [NamelistVar('um_rose', filepath=UKESM1_params.um_namelist_file, namelist='namelist:run_convection',
                        nl_var='cca_md_knob'),
             NamelistVar('um_rose', filepath=UKESM1_params.um_namelist_file, namelist='namelist:run_convection',
                         nl_var='cca_dp_knob')]
functions = [None, None]
transform_values = dict(zip(namelists, functions))

UKESM1_params.register_param_with_partial(
    'cca_md_knob',  # name to register the function as.
    UKESM1_params.fn_param,  # using fn_param as the base function.
    transform_values=transform_values  # transform values to apply.
)  # generate/register the function. Do  implement functions (see functions)






class UKESM1_1(UM_rose_cylc7, UKESM1_params):
    """
    Class to support UKESM1_1 model running in ROSE with cylc 7.
    This joins the UM_rose_cylc7 and UKESM1_params classes.
    """
    @register_param('runModelTime')
    def run_time(self,
                 runTime: typing.Union[str, int, float, None],
                 transform:bool = True) -> \
            typing.Union[list[tuple[NamelistVar, str]], str]:
        """
        Set the run time for the model. This is in seconds or as an ISO duration string.
        UM wants it as a ISO duration string. This function will convert to that if needed.
        :param runTime: The run time in seconds or as an iso duration.
        :param transform -- does nothing. For compatibility with other functions.
        :return: list((nl,value)) or just the value read in from the config.
        """
        nl = NamelistVar('um_rose', filepath=pathlib.Path('rose-suite.conf'),
                         namelist='jinja2:suite.rc', nl_var='MAIN_CLOCK', default=0)
        if runTime is None:
            return self.read_nl_value(nl)
        if isinstance(runTime, str):
            check = parse.DurationParser().parse(runTime)  # make sure it parses
            val = runTime
        else:
            val = genericLib.seconds_to_isoduration(
                runTime)  # can't see any way to use metomi.isodatetime to do this.
        return [(nl, val)]

class UKESM1_1_c8(UM_rose_cylc8, UKESM1_params):
    """
    Class to support UKESM1_1 model running in ROSE with cylc 8.
    This joins the UM_rose_cylc8 and UKESM1_params classes.
    """
    @register_param('runModelTime')
    def run_time(self,
                 runTime: typing.Union[str, int, float, None],
                 transform:bool = True) -> \
            typing.Union[list[tuple[NamelistVar, str]], str]:
        """
        Set the run time for the model. This is in seconds or as an ISO duration string.
        UM wants it as a ISO duration string. This function will convert to that if needed.
        :param runTime: The run time in seconds or as an iso duration.
        :param transform -- does nothing. For compatibility with other functions.
        :return: list((nl,value)) or just the value read in from the config.
        """
        nl = NamelistVar('um_rose', filepath=pathlib.Path('rose-suite.conf'),
                         namelist='template variables', nl_var='MAIN_CLOCK', default=0)
        if runTime is None:
            return self.read_nl_value(nl)
        if isinstance(runTime, str):
            check = parse.DurationParser().parse(runTime)  # make sure it parses
            val = runTime
        else:
            val = genericLib.seconds_to_isoduration(
                runTime)  # can't see any way to use metomi.isodatetime to do this.
        return [(nl, val)]


# add in core optclim variables.
config_dir = pathlib.Path(__file__).parent / 'parameter_config/UM_rose_params' # directory with the config files.

pth_all = config_dir/'UM_rose_Parameters.csv' # parameters for all UM_rose models.
UM_rose.update_from_file(pth_all, duplicate=True)

pth_c7 = config_dir/'UM_rose_Parameters_c7.csv'# cylc7 specific
UM_rose_cylc7.update_from_file(pth_c7, duplicate=True)

pth_c8 = config_dir/'UM_rose_Parameters_c8.csv' # cylc8 specific
UM_rose_cylc8.update_from_file(pth_c8, duplicate=True)
pth = config_dir/'UM_rose_UKESM1_1.csv'
UKESM1_params.update_from_file(pth)  # load up the UKESM1_1 specific parameters.

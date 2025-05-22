# Class to support Unified Model running in Rose.
#This version is rather specialised for archer2.
# If running on other platforms then will need to refactor/generalise this code.
# It has some fairly large difference from Model
# 1) It uses suite_dir which is where the config info gets written. On archer2 ths should be in
# /home/n02/n02-puma/<username>/ Default is /home/n02/n02-puma/<username>/rose_optclim/name_PID
# 2) model_work_dir should be where the model actualy puts its data.
#     if model_data_dir is not set then this is set to $ROSE_DATA/$DATAM in running() is ROSE_DATA is set in the env.
# Data will copied from this directory to model_dir/'History_Data'  in succeeded().
#
# 3) optclim.rc to be added to the suite.rc file. which done by copying file and including it in suite.rc
#
# 4) Various new variables are added:
#    - MODEL_CONFIG -- set to the config file. (Used in optclim.rc)
#    - PREBUILD -- set to the prebuild file if present.
#    - runEnvSetup -- set to the environment setup script. Default is $OPTCLIMTOP/OptClimVn3/setup_archer2
#    - runUser -- set to the user name to run the suite as.
#    - runCode -- set to the job code/account to run suite with.
#    - runModelTime -- set to the time as iso duration for model to run for.
#    - OPTCLIM_ARGS -- set to the arguments to pass to optclim scripts which get passed to set_status_script.



# TODO - figure out what to do if the model fails. Coz often might fix in
#   cylc gui. But then model status won't get updated.
# But point of continue option is to automatically fix and run...
# So continue should run the right rose command. There is a restart option in the rose submit stuff.
# perhaps add an option to manually fix the status??? But then user would need to do that manually anyhow as job will fail when it gets told to move to suceeded..  Or fix that with a warning..

import fileinput
import logging
import os
import re
import tarfile
import typing
import shutil
import platform

import subprocess

import metomi.isodatetime.exceptions
import metomi.isodatetime.parsers as parse

import genericLib

from ModelBaseClass import register_param # Used to allow functions. Currently none defined.
from Model import Model
import pathlib
from namelist_var import NamelistVar, GroupConfig


my_logger = logging.getLogger(f"OPTCLIM.{__name__}") # have this anywhere you want logging

class UM_rose(Model):
    """
    Class to support the Unified model running in ROSE.
    The Complication is that this the UM uses cylc which requires submitting a jon to puma2.
    This version is rather specialised for archer2. If want to run on another platform then
    will need to refactor/generalise this code.

    This class adds suite_dir to the class attributes. This is where the suite info gets written.
    If not set then will be generated from the name and process id as:
    self.puma_dir/f'{self.name}_{os.getpid()}
    Class also adds model_data_dir to the object attributes. This is where data should be written by model.
      If None then when running() ran self.model_dir will be set to $ROSE_SUITE_DIR/$DATAM

    The initialisation uses the following values from run_info. :
    For the following variables values of None mean the suite is not modified.
        - runModelTime:str -- time as iso duration for model to run for.
        - runUser:str -- user name to run the suite as.
        - runCode:str -- job code/account to run suite with.
        - prebuild:str -- path to prebuild. Means much faster compilation

    - use_scratch:bool -- If True then use scratch space for work and share.
                           Files here are deleted after 28 days,
    - OPTCLIM_ARGS :str -- arguments to pass to optclim scripts which get passed to set_status_script. Default ''
    - runEnvSetup:str -- path to the environment setup script.
       If None (or not set) will be set to  $OPTCLIMTOP/OptClimVn3/setup_archer2


    MODEL_CONFIG will be set to self.cache_path

    """
    suite_dir: typing.Optional[pathlib.Path] # path for suite dir.
    model_data_dir: typing.Optional[pathlib.Path] # path for model_data_dir.
    # test if we are on Archer by calling hostname -A and that stdout contains archer2.ac.uk
    stat = subprocess.run(['hostname','-A'],capture_output=True,text=True)
    if not ((stat.returncode == 0) and 'archer2.ac.uk' in stat.stdout):
        my_logger.warning('Not running on archer2. This code will need re-writing to work on other platforms')
    # Get the user ID
    user_id = os.environ.get('USER') or os.environ.get('USERNAME')

    puma_dir = pathlib.Path('/home/n02/n02-puma') / user_id/'rose_optclim'  # puma2 root path on archer2.
    def __init__(self, *args, **kwargs):
        """
        Init the UM_rose instance. Calls the super-class init method.
        :param args: positional args. Passed through to super-class
        :param kwargs: kwargs -- passed through to super-class.
           if kwargs contains suite_dir that value will be used to set suite_die
        """
        suite_dir = kwargs.pop("suite_dir",None)
        model_data_dir=kwargs.pop('model_data_dir',None)
        if model_data_dir is not None:
            self.model_data_dir=pathlib.Path(model_data_dir)
        else:
            self.model_data_dir=None
        super().__init__(*args, **kwargs)  #
        # deal with suite_dir -- where suite info gets written. Different from model_dir which is where
        # models are ran. This different from other models so far.

        if suite_dir is None:
            if self.name is not None:
                # Work out suite_dir which is where suite info gets written.
                # suite dir name is name_PID -- should be unique enough..
                # problem is that cylc uses a very flat space...
                suite_dir = self.puma_dir/f'{self.name}_{os.getpid()}'
            else:
                suite_dir = None
        else:
            suite_dir = pathlib.Path(suite_dir)

        self.suite_dir = suite_dir #
        self.configs = GroupConfig(root_dir=self.suite_dir)  # grouped configs for writing out generic namelists

        # modify parameters_no_key to include runModelTime, runUser, runCode, OPTCLIM_ARGS, runEnvSetup if set.
        # Those parameters do not contribute towards the unique key used to identify the model.
        if self.run_info is not None: # need to test for None as reloading of config gives us None.
            for key in ['runModelTime','runUser','runCode','OPTCLIM_ARGS','runEnvSetup','prebuild']:
                if self.run_info.get(key) is not None: # (Get None if either null in the original  json config or not present)
                    self.parameters_no_key[key] = self.run_info[key]
            # Deal with runEnvSetup
            runEnvSetup = genericLib.expand(self.parameters_no_key.get('runEnvSetup',
                                                                       '$OPTCLIMTOP/OptClimVn3/setup_archer2'))
            # check runEnvSetup actually exists.
            if not pathlib.Path(runEnvSetup).is_file():
                raise ValueError(f'runEnvSetup {runEnvSetup} does not exist.')
            self.parameters_no_key['runEnvSetup'] = str(runEnvSetup) # need to convert to string.
            # deal with OPTCLIM_ARGS
            self.parameters_no_key['OPTCLIM_ARGS'] = self.parameters_no_key.get('OPTCLIM_ARGS','')

        # set up MODEL_CONFIG to point to the configuration.
        if self.config_path is not None:
            self.parameters_no_key['MODEL_CONFIG'] =str(self.config_path)
            my_logger.debug(f"Set MODEL_CONFIG to {str(self.config_path)}")

        # switch off fixed scripts. Will change submit_cmd instead.
        # NO will stick to existing approach.
        # Something like
        # ssh -Y puma2 'export PATH=$PATH:/home/n02/n02/fcm/metomi/bin; rose suite-run  --new --no-gcontrol -v -v -C ~/rose_optclim/case002_47642'
        # should work. But fails when tries to submit to archer2.
        # probably need some help from helpdesk or mike.
        self.submit_script = None
        self.continue_script = None # probably don't need this for now. There if have an error.
        # I think ROSE handles that kind of stuff so just need to resubmit the config on puma.
        # Alternatively (if easier) modify the super class submit method




    def create_model(self,
                     direct:typing.Optional[pathlib.Path]=None,
                     copy_ref:bool=True
                     ) -> None:
        """
        Create the model. This is a rose specific version of create model. It will copy the reference config
         to **self.suite_dir** and create model_dir using super().create_model.
        :param direct: directory to create the model in.
          Provided for compatibility with other models but if not None an error will be raised.
        :param copy_ref: If True then copy the reference directory to the suite_dir.
        :return:nothing.
        """
        if direct is not None:
            raise ValueError(f'direct = {direct} is not None in UM_rose specific version of create_model')
        super().create_model(copy_ref=False) # create model dir
        super().create_model(direct=self.suite_dir,copy_ref=copy_ref) # create the suite




    @staticmethod
    def replace_file(file:pathlib.Path,
                     match:str,
                     replacement:str,
                     backup_ext:typing.Optional[str] = None
                     ) -> typing.Optional[tuple[pathlib.Path,int]]:
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
                    break # no need to read more lines.

        # if no match found then return None
        if not found_match:
            my_logger.debug(f'No match found in {file} for {match}')
            return None

        # got some matches so process this file

        count_match = 0
        with fileinput.input(files=file, inplace=True,backup=backup_ext) as f: # inplace fileinput uses temp backup file.
            for line in f:
                repl_line = re.sub(match, replacement, line)
                if repl_line != line: # any changes?
                    count_match += 1
                print(repl_line, end='')
        return file,count_match

    @staticmethod
    def change_rose_dir(direct:pathlib.Path) -> dict[pathlib.Path:int]:
        """
        Recursively change all values of [file:$ROSE_DATA/] to [file: in text files in input directory.
        :param direct: directory to change.
        :return dict indexed by file path and number of changes made.
        """
        #TODO -- might need to replace other ROSE_DATA references. Suchk it and see!
        raise NotImplementedError("Code no longer needed")
        match = r"(.*)\[file:\$ROSE_DATA/(.*?)\](.*)"
        replacement = r"\1[file:\2]\3"

        if not direct.is_dir(): # check direct is a directory!
            raise ValueError(f'path {direct} is not a directory.')

        modified_files = dict()
        for file in direct.iterdir(): # iterate over files in the directory
            if file.is_dir():
                my_logger.debug(f'{file} is a directory so calling change_rose_dir')
                modified_files.update(UM_rose.change_rose_dir(file))
            elif file.is_file() and genericLib.likely_text_file(file): # likely a text file
                res = UM_rose.replace_file(file,match,replacement,backup_ext='.bak') # do replacement
                if res is not None: # any change?
                    my_logger.debug(f'Modified {pth}')
                    modified_files[res[0]] = res[1]
            else:
                my_logger.debug(f'file {file} is not a directory nor likely a text file so skipping')
        my_logger.debug(f'Changed {modified_files}')
        return modified_files

    def modify_model(self):
        """
        UM rose specific version of modify model,
        Needs to:
        1) Modify directories so that running in a sensible place wih output going there.
           use self.model_dir to get directory.
        1a) It may be necessary to modify the rose config so that it no longer copies to archer2 the rose info.
         (as we already have it).
        1c Add information from self.run_info to the rose config. These will include, if present:
            runTime -- time (in seconds) for model, runCode -- job code/account to run suite with.
            There may be other specific information needed
        2) Put in the cylc config just before model starts running (which could be multiple times):
            self.set_status_script self.config_path RUNNING
        3) Add to the cylc config after the model has finished
           self.set_status_script self.config_path SUCCEEDED
        4) Optionally add to the cylc config where errors are detected
               self.set_status_script self.config_path FAILED

        """
        super().modify_model()  # call the base class method.
        # now do the specific stuff...
        self.copy_suite_apps() # copy the optclim specific apps to the suite dir.
        self.update_suite_rc() # update the suite.rc file

    def update_suite_rc(self):
        """
        Update the cylc/rose suite.rc to include OptClim tasks.
        If use_scratch set then update rose_suite.conf to use scratch space.
        """
        suite_file = self.suite_dir / 'suite.rc'
        genericLib.backup_file(suite_file,ext='.bak',create='copy')  # make a backup of the suite.rc file.
        with suite_file.open('a') as suite_rc:
            suite_rc.write('\n\n%include optclim.rc\n')

        if self.run_info.get('use_scratch',False):
            # can modify rose-suite.conf elsewhere so back this up with a different name.
            with fileinput.input(self.suite_dir/'rose-suite.conf',inplace=True,backup='.bak_update') as f:
                for line in f:
                    if f.isfirstline(): # first line
                        print('## Scratch space being used')
                        print(r'root-dir{share}=ln*=/mnt/lustre/a2fs-nvme/work/n02/n02/$USER')
                        print(r'root-dir{work}=ln*=/mnt/lustre/a2fs-nvme/work/n02/n02/$USER')
                        
                    print(line,end='')

                # New lines first. Could be generalised but 
            my_logger.debug('Using scratch space for share and work space')
        


    def copy_suite_apps(self):
        """Copy OptClim specific apps from the reference directory to the model directory"""
        optclim_tasks_root = pathlib.Path(__file__).parent/'um_rose_files/optclim_tasks'
        shutil.copytree(optclim_tasks_root, self.suite_dir, dirs_exist_ok=True)

    def instantiate(self,fake:bool = False) -> None:
        """
        Instantiate the model. This is a UM_rose specific version of instantiate.
        It will call the superclass method and then tar/gzip the suite_dir and copy it to the model_dir.
        :param fake: if True then don't run the model. Default is False.
        :return:nothing.
        """
        super().instantiate(fake=fake) # call super class method.
        # now do the UM_rose specific stuff.
        # For now that is tar up the suite_dir and copy to model_dir.
        tar_file = self.model_dir/'rose_suite.tar.gz'
        tar_file.unlink(missing_ok=True) # remove any old tar file.
        with tarfile.open(tar_file, "w:gz") as tar:
            tar.add(self.suite_dir, arcname=self.suite_dir.name)
        my_logger.debug(f'Created tar file {tar_file} from {self.suite_dir}')

    def check(self) -> bool:

        """
        Check the model. This is a UM_rose specific version of check.
        Calls the superclass method and then do the following checks:
          1) Check that START_TIME, RUN_TARGET and RESUB_TIME are compatible. 
             RUN_TARGET is an integer multiple of RESUB_TIME. Complication is if RESUB_TIME is in months...
        Will raise ValueError if any of the checks fail.
        :return: True if the model is valid, False otherwise.
        """
        if not super().check():
            return False # failed so return False.
        ## UM_rose specific checks.
        # 1) Check that START_TIME, RUN_TARGET and RESUB_TIME are compatible.
        # TODO -- have read method for UM_ROSE configs that handles iso times and durations
        # By converting them to Time Points and Durations we are also checking that
        # strings are valid.
        try:
            start_time = parse.TimePointParser().parse(self.read_param('START_TIME'))
            run_target = parse.DurationParser().parse(self.read_param('RUN_TARGET'))
            resub_time = parse.DurationParser().parse(self.read_param('RESUB_TIME'))
        except metomi.isodatetime.exceptions.ISO8601SyntaxError as err: # catch any parsing errors.
            raise ValueError(f'Problem parsing one of START_TIME, RUN_TARGET or RESUB_TIME. {err}')
        # iterate from start_time  to start_time + run_target.
        # Doing this because months are not the same (second) duration throughout the year...
        end_time = start_time + run_target
        time= start_time
        while time < end_time:
            time += resub_time
        # Now time should be start_time + run_target
        if time != end_time:
            raise ValueError(f'RUN_TARGET {run_target} and RESUB_TIME {resub_time} are not compatible')

        return True

    def submit_cmd(self) -> typing.List[str]:
        """"
        Generate the submission command. Over rides the super-class version.
        """
        # check status sensible for submitting run,
        if self.status not in ['INSTANTIATED', 'PERTURBED','CONTINUE']:
            raise ValueError(f"Status {self.status} not expected ")
        remote_machine = self.run_info.get('remote_machine','')
        # remote machine where rose runs.
        cmd = self.expand('$OPTCLIMTOP/OptClimVn3/scripts/UM_rose/SUBMIT_to_puma.sh')
        cmd = [cmd,self.model_dir,remote_machine]
        return cmd

    @register_param('runModelTime')
    def run_time(self,runTime:typing.Union[str,int,float,None]) -> \
            typing.Union[list[tuple[NamelistVar,str]],str]:
        """
        Set the run time for the model. This is in seconds or as an ISO duration string.
        UM wants it as a ISO duration string. This function will convert to that if needed.
        :param runTime: The run time in seconds or as an iso duration.
        :return: list((nl,value)) or just the value read in from the config.
        """
        nl = NamelistVar('um_rose',filepath=pathlib.Path('rose-suite.conf'),
                         namelist='jinja2:suite.rc',nl_var='MAIN_CLOCK',default=0)
        if runTime is None:
            return self.read_nl_value(nl)
        if isinstance(runTime,str):
            check = genericLib.parse_isoduration(runTime) # make sure it parses
            val = runTime
        else:
            val = genericLib.seconds_to_isoduration(runTime)
        return [(nl,val)]

    def running(self) -> typing.Optional[str]:
        """
        UM_model version of running. No jid possible for UM as cylc handling all of that.
        Will try and set up model_data_dir if not set.
    
        """

        if self.model_data_dir is None:
            base_dir = os.environ.get('ROSE_DATA')
            if base_dir is not None:
                base_dir=pathlib.Path(base_dir)/os.environ['DATAM']
                self.model_data_dir=base_dir
                my_logger.debug(f'Set model_data_dir to {self.model_data_dir}')
        
        self.set_status('RUNNING') # update status and save to disk

        return 'NOJOBID'

    def succeeded(self):
        """
        UM_ROSE specific version of succeeded. 
         If self.model_data_dir is defined
             Copies pp, netcdf and last dump files  in this dir to model_dir/'History_Data'
        then calls superclass succeeded.
        Copying as files might be on other file systems and hard links across file systems do not work.
        """

        data_dir = self.model_dir/'History_Data' # where model data will be copied too
        if self.model_data_dir is None: # not set trigger an error
            raise ValueError('model_data_dir not set. Something probably went wrong in running() method')

        if not (self.model_data_dir.exists() and self.model_data_dir.is_dir()):
            raise FileNotFoundError(f'self.model_data_dir {self.model_data_dir} does not exist or is not a dir')
        my_logger.debug(f"model_data_dir set and is {self.model_data_dir}")
        # create data dir if it does not exist.
        data_dir.mkdir(parents=True, exist_ok=True)

        # ready to go now.
        file_patterns=['*.pp','*.nc'] # file patterns to copy.
        # iterate over patterns to find list of files to copy.
        files_to_copy= []
        for fpattern in file_patterns:
            files = [file for file in self.model_data_dir.glob(fpattern) if file.is_file()]
            files_to_copy += files
        # deal with dumps
        # UM file names do sort alphanumerically so last file is most recent..
        # note if this method gets run more than once you will have more than dump file...
        dump_files=[f for f in self.model_data_dir.glob('*.d*_00') if f.is_file()]
        files_to_copy.append(dump_files[-1]) # last dump file.

       # now copy the files to the data_dir.
        for file in files_to_copy:
            new_file = data_dir/file.name
            my_logger.debug(f'Copying {file} to {new_file}')
            shutil.copy2(file,new_file)

        super().succeeded() # call the superclass method.
        # Must occur after copying files otherwise data may not be accessible for the post-processing job.
            
            
            
                    
                    
        
        

pth = pathlib.Path(__file__).parent /'parameter_config/UM_rose_Parameters.csv'
UM_rose.update_from_file(pth, duplicate=True)

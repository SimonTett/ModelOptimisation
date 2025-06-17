# Class to support Unified Model running in Rose.
#This version is rather specialised for archer2.
# If running on other platforms then will need to refactor/generalise this code.
# It has some fairly large difference from Model. See class doc
# Things marker ARCHER2 are specific to ARCHER2. Generalise if on another platform.

# TODO - figure out what to do if the model fails. Coz often might fix in
#   cylc gui. But then model status won't get updated.
# But point of continue option is to automatically fix and run...

import fileinput
import logging
import os
import re
import tarfile
import typing
import shutil
import subprocess
import metomi.isodatetime.exceptions
import metomi.isodatetime.parsers as parse
import genericLib

from ModelBaseClass import register_param #
from Model import Model
import pathlib
from namelist_var import NamelistVar, GroupConfig


my_logger = logging.getLogger(f"OPTCLIM.{__name__}") # have this anywhere you want logging

class UM_rose(Model):
    """
    Class to support the Unified model running in ROSE.
    The complication is that this the UM uses cylc which requires submitting a jon to puma2.
    This version is rather specialised for archer2. If want to run on another platform then
    will need to refactor/generalise this code. Main changes come from including optclim.rc in suite.rc

    This class adds suite_dir to the class attributes. This is where the suite info gets written.
    If not set then will be generated from the name and process id as:
    self.puma_dir/f'{self.name}_{os.getpid()}
    Class also adds model_data_dir to the object attributes. This is where data should be written by model.
      If None then when running() ran self.model_dir will be set to $ROSE_DATA/$DATAM if $ROSE_DATA defined
      Data is copied from this directory (if not None) to model_dir/'share/data/History_Data'  in succeeded().

    Initialisation uses the following values from run_info which become variables -- see UM_rose_Parameters.csv. :
    For the following variables values of None mean the suite is not modified.
        - runModelTime:str -- time as iso duration for model to run for.
        - runUser:str -- username to run the suite as.
        - runCode:str -- job code/account to run suite with.
        - prebuild:str|bool -- path to prebuild. Means much faster compilation
                              if bool and True will guess path from reference.
                              This assumes on archer2 and user_id is the same as reference

    - use_scratch:bool -- If True then use scratch space for work and share.
                           Files here are deleted after 28 days,
    - OPTCLIM_ARGS :str -- arguments to pass to optclim scripts which get passed to set_status_script. Default ''
    - OPTCLIM_SET_STATUS_SCRIPT -- path to script which sets the status. (Uses self.set_status_script)
    - runEnvSetup:str -- path to the environment setup script.
       If None (or not set) will be set to  $OPTCLIMTOP/OptClimVn3/setup_archer2
    - MODEL_CONFIG will be set to self.cache_path

    """
    # additional attributes to the model class.
    suite_dir: typing.Optional[pathlib.Path] # path for suite dir.
    model_data_dir: typing.Optional[pathlib.Path] # path for model_data_dir.

    # test if we are on Archer by calling hostname -A and that stdout contains archer2.ac.uk
    archer2 = False
    stat = subprocess.run(['hostname','-A'],capture_output=True,text=True)
    if not ((stat.returncode == 0) and 'archer2.ac.uk' in stat.stdout):
        my_logger.warning('Not running on archer2. This code will need re-writing to work on other platforms')
        archer2 = True
        # probably overkill. Only for instantiate
    # Get the user ID
    user_id = os.environ.get('USER') or os.environ.get('USERNAME')
    base_path = f'{user_id}/rose_optclim' # which gives us path where files are stored.
    # ARCHER2
    puma_dir = pathlib.Path('/home/n02/n02-puma') /base_path  # puma2 root path on archer2.
    def __init__(self, *args, **kwargs):
        """
        Init the UM_rose instance. Calls the super-class init method.
        :param args: positional args. Passed through to super-class
        :param kwargs: kwargs -- passed through to super-class.
           if kwargs contains suite_dir that value will be used to set suite_dir.
           If suite_dir is not None then the configurations will point to that directory
           if kwargs contains model_data_dir that will be used to set model_data_dir
        Sets up the following no_key variables:
        runModelTime, runUser, runCode,OPTCLIM_ARGS,runEnvSetup,prebuild from run_info
        Sets up MODEL_CONFIG and OPTCLIM_SET_STATUS_SCRIPT
        and sets submit and continue scripts to be in suite_dir.
        """
        # remove suite_dir and model_data_dir from kwargs
        suite_dir:typing.Optional[str] = kwargs.pop("suite_dir",None)
        model_data_dir:typing.Optional[str]=kwargs.pop('model_data_dir',None)
        if model_data_dir is not None:
            self.model_data_dir=pathlib.Path(model_data_dir)
        else:
            self.model_data_dir=None
        super().__init__(*args, **kwargs)  # call the super-class init method.

        # deal with suite_dir -- where suite info gets written. Different from model_dir which is where
        # models are ran. This different from other models so far.
        if suite_dir is None:
            if self.name is not None:
                # Work out suite_dir which is where suite info gets written.
                # suite dir name is name_PID -- should be unique enough..
                # problem is that cylc uses a very flat space...
                self.suite_dir = self.puma_dir/f'{self.name}_{os.getpid()}'
            else:
                self.suite_dir = None
        else:
            self.suite_dir = pathlib.Path(suite_dir)
        self.configs = GroupConfig(root_dir=self.suite_dir)  # grouped configs for writing out generic namelists

        # modify parameters_no_key to include runModelTime, runUser, runCode, OPTCLIM_ARGS, runEnvSetup,prebuild if set.
        # Those parameters do not contribute towards the unique key used to identify the model.
        if self.run_info is not None: # need to test for None as reloading of config gives us None.
            for key in ['runModelTime','runUser','runCode','OPTCLIM_ARGS','runEnvSetup','prebuild']:
                if self.run_info.get(key) is not None: # (Get None if either null in the original  json config or not present)
                    self.parameters_no_key[key] = self.run_info[key]
            # deal with prebuild set to True -- where we guess the path.
            if self.parameters_no_key.get('prebuild') is True:
                prebuild_path = self._guess_prebuild()
                if prebuild_path is not None:
                    self.parameters_no_key['prebuild'] = str(prebuild_path)
                else:
                    self.parameters_no_key.pop('prebuild') # remove it as True won't work!
            # set RUNID to False and RUN_NAME to first 5 characters of the name.
            if self.name is not None:
                self.parameters_no_key['RUNID'] = False
                self.parameters_no_key['RUN_NAME'] = self.name[:5]  # first 5 characters of the name.

            # Deal with runEnvSetup
            runEnvSetup = genericLib.expand(self.parameters_no_key.get('runEnvSetup',
                                                                       '$OPTCLIMTOP/OptClimVn3/setup_archer2'))
            # check runEnvSetup actually exists.
            if not pathlib.Path(runEnvSetup).is_file():
                raise ValueError(f'runEnvSetup {runEnvSetup} does not exist.')
            self.parameters_no_key['runEnvSetup'] = str(runEnvSetup) # need to convert to string.

            # deal with OPTCLIM_ARGS -- giving it a default value of ''
            self.parameters_no_key['OPTCLIM_ARGS'] = self.parameters_no_key.get('OPTCLIM_ARGS','')
            # setup OPTCLIMTOP
            self.parameters_no_key['OPTCLIMTOP']=str(genericLib.expand('$OPTCLIMTOP') )# setup OPTCLIMTOP


        # set up MODEL_CONFIG to point to the configuration.
        if self.config_path is not None:
            self.parameters_no_key['MODEL_CONFIG'] =str(self.config_path)
            my_logger.debug(f"Set MODEL_CONFIG to {str(self.config_path)}")

        # set up OPTCLIM_SET_STATUS_SCRIPT
        if self.set_status_script is not None:
            self.parameters_no_key['OPTCLIM_SET_STATUS_SCRIPT'] = str(self.set_status_script)
            my_logger.debug(f'Set OPTCLIM_SET_STATUS_SCRIPT to {str(self.set_status_script)}')
       # set up submit and continue script. These need to be on puma2 so putting them in the suite_dir
        if self.suite_dir is not None:
            self.submit_script = self.suite_dir/'submit_script.sh' # script to run on puma2 to submit the job.
            self.continue_script = self.suite_dir/'continue_script.sh' # probably don't need this for now. There if have an error.


    def create_model(self,
                     direct:typing.Optional[pathlib.Path]=None,
                     copy_ref:bool=True
                     ) -> None:
        """
        Create the model. This is a rose specific version of create model. It will copy the reference config
         to **self.suite_dir** and create model_dir using super().create_model.
        :param direct: directory to create the model in.
        :param copy_ref: If True then copy the reference directory to the suite_dir.
        :return:nothing.
        """

        super().create_model(direct,copy_ref=False) # create model dir
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

    # utility fns. Private for now
    def _guess_prebuild(self) -> typing.Optional[pathlib.PurePath]:
        """
        Guess the prebuild dict. Will only work on archer2/puma
        :param self:
        :return: PurePath -- successfully guessed prebuild dct. (path is a dir) 
                 None -- guess did not work.
        """
        # ARCHER2

        # get the name from the reference and assume userid the same as this
        ref_suite_name = self.reference.name
        # work out user and id.
        # work out user id in reference if it is an abs path.
        user_id = self.user_id
        if self.reference.is_absolute():
            # work out user-id from  path
            user_id = self.reference.parts[4]
        prebuild = pathlib.Path('/home/n02/n02-puma/') / f'{user_id}/cylc-run/{ref_suite_name}/share/fcm_make_um'
        if (prebuild/'extract').is_dir():  # dct  a dir which exists and had an extract.
            my_logger.warning(f'Guessed prebuild on Archer2 to be {prebuild}')
        else:
            my_logger.warning(f'Prebuild: {prebuild/"extract"} is not a directory')
            return None
        # but actually need path on puma2. Sigh!
        prebuild = self._puma_path(prebuild) # get the puma path -- which is what is needed!
        return prebuild


    def _create_script(self,
                       script_type: typing.Literal['submit', 'continue'] ,
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
        else:
            raise ValueError(f'Unknown script type: {script_type}')
        script.parent.mkdir(parents=True, exist_ok=True)   # might need to create directory
        script.unlink(missing_ok=True)  # unlink it if it exists.
        with script.open('wt') as f:
            f.write('#!/bin/bash --login\n')
            cmd = ['rose', 'suite-run']
            if script_type == 'submit':
                cmd.append('--new')
            elif script_type == 'continue':
                cmd.append('--restart ')
            else:
                raise ValueError(f'Unknown script_type {script_type}')
            cmd.append('--no-gcontrol')
            if args:
                cmd.append(args)
            cmd.append(f'-C {self._puma_path(self.suite_dir)}')  # path to the suite dir.
            f.write(' '.join(cmd) + '\n')

    def modify_model(self):
        """
        UM rose specific version of modify model,
        Does the following (after calling the superclass method
        Copies the optclim suite apps into the suite_dir and modifies the suite.rc file
        Also generates the submit and continue scripts,

        """
        super().modify_model()  # call the base class method.
        # now do the specific stuff...
        self.copy_suite_apps() # copy the optclim specific apps to the suite dir.
        self.update_suite_rc() # update the suite.rc file
        # need to generate submit_cmd and continue_cmd
        # done here as super class instantiate cleans directory before doing anything else and then checks.
        # so need to call this in modify_model
        args = self.parameters_no_key.get('OPTCLIM_ARGS')
        self._create_script('submit',args=args)
        self._create_script('continue',args=args)


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
            my_logger.debug('Using scratch space for share and work space')

    def copy_suite_apps(self):
        """Copy OptClim specific apps from the reference directory to the model directory"""
        optclim_tasks_root = pathlib.Path(__file__).parent/'um_rose_files/optclim_tasks'
        shutil.copytree(optclim_tasks_root, self.suite_dir, dirs_exist_ok=True)

    def instantiate(self,fake:bool = False) -> None:
        """
        Instantiate the model. This is a UM_rose specific version of instantiate.
        Calls  superclass method and then tar/gzip the suite_dir and copy it to the model_dir.
        :param fake: passed to the super class method.
        :return:nothing.
        """

        super().instantiate(fake=fake) # call super class method.
        # tar up the suite_dir and copy to model_dir.
        tar_file = self.model_dir/'rose_suite.tar.gz'
        try:
            tar_file.unlink(missing_ok=True)  # remove any old tar file.
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(self.suite_dir, arcname=self.suite_dir.name)
            my_logger.debug(f'Created tar file {tar_file} from {self.suite_dir}')
        except Exception as e:
            my_logger.error(f"Failed to create tar file: {e}")
            raise
        
    def check(self) -> bool:

        """
        Check the model. This is a UM_rose specific version of check.
        Calls the superclass method and then does the following checks:
          1) Check that START_TIME, RUN_TARGET and RESUB_TIME are compatible. 
             RUN_TARGET is an integer multiple of RESUB_TIME. Complication is if RESUB_TIME is in months...
        Will raise ValueError if any of these fail.
        2) Check that the submit and continue scripts are files.
           if not will raise FileNotFoundError.
        :return: True if the model is valid, False otherwise.
        """
        if not super().check():
            return False # failed so return False.
        ## UM_rose specific checks.
        # 1) Check that START_TIME, RUN_TARGET and RESUB_TIME are compatible.
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
        # 2)  check the submit and continue scripts exist
        for script in [self.submit_script, self.continue_script]:
            if not (script is None or script.is_file()):
                raise FileNotFoundError(f"{script} is not a file.")

        return True
    @staticmethod
    def _puma_path(path:pathlib.Path) -> pathlib.PurePath:
        """
        Convert an archer2 path to a path on puma2 -- very specific to archer2/puma2.
        :param path: path to convert. Must begin with /home/n02/n02-puma which will be converted to /home/n02/n02
        If not then the path will be unmodified and retuned as a purePath.
        :return: puma path as a purePath.
        """
        if not isinstance(path, pathlib.Path):
            raise ValueError(f'path {path} is not a pathlib.Path')
        cpts = path.parts
        # ARCHER2
        if cpts[0:4] != ('/','home','n02','n02-puma'):
            my_logger.warning(f'path {path} does not start with /home/n02/n02-puma')
            return pathlib.PurePath(path)
        result = cpts[0:3] +tuple(['n02']) + cpts[4:] # replace n02-puma with n02
        result = pathlib.PurePath(*result) # convert to a pure path.
        return result

    def running(self) -> typing.Optional[str]:
        """
        UM_model version of running. No jid possible for UM as cylc handling all of that.
        Will try and set up model_data_dir if not set.
        Note does not call superclass method...
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
             Copies pp, netcdf and last dump files  in this dir to model_dir/'share/data/History_Data'
        If not defined raises ValueError.
        if not a dir raises FileNotFoundError.
        then calls superclass succeeded.
        Copying as files might be on other file systems and hard links across file systems do not work.
        """

        data_dir = self.model_dir/'share/data/History_Data' # where model data will be copied too
        if self.model_data_dir is None: # not set trigger an error
            raise ValueError('model_data_dir not set. Something probably went wrong in running() method')

        if not self.model_data_dir.is_dir():
            raise FileNotFoundError(f'self.model_data_dir {self.model_data_dir} is not a dir')
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
        # and sort them!
        dump_files = sorted(dump_files)
        files_to_copy.append(dump_files[-1]) # last dump file.

       # now copy the files to the data_dir.
        for file in files_to_copy:
            new_file = data_dir/file.name
            my_logger.debug(f'Copying {file} to {new_file}')
            shutil.copy2(file,new_file)

        super().succeeded() # call the superclass method.
        # Must occur after copying files otherwise data may not be accessible for the post-processing job.

    def submit_cmd(self) -> typing.List[str]:
        """"
        Generate the submission command. Overrides the super-class version.
        Will return ssh cmd that submits suite.
        """
        if self.status in ['INSTANTIATED', 'PERTURBED']: # start again.
            script = self.submit_script
        elif self.status == 'CONTINUE':
            script = self.continue_script
            raise NotImplementedError('Continue not implemented yet')
        else:
            raise ValueError(f"Status {self.status} not expected ")
        # ARCHER2
        cmd = ['ssh','puma2',self._puma_path(script)] # script to submit
        # Could make puma2 more generic -- so specified through run_info.
        # But no point at the moment as only running on archer2/puma2.
        # and I suspect that changes will need to be different on other platforms.
        # Where hopefully one can just run rose/cylc on the local machine.
        return cmd

    ## methods that handle 'complex' model parameters.
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
            check = parse.DurationParser().parse(runTime) # make sure it parses
            val = runTime
        else:
            val = genericLib.seconds_to_isoduration(runTime) # can't see any way to use metomi.isodatetime to do this.
        return [(nl,val)]
## of class definition.
# add in core optclim variables.
pth = pathlib.Path(__file__).parent /'parameter_config/UM_rose_Parameters.csv'
UM_rose.update_from_file(pth, duplicate=True)
# and then all the simple parameters
pth = pathlib.Path(__file__).parent /'parameter_config/UM_rose_UKESM1_1.csv'
UM_rose.update_from_file(pth,duplicate=False)

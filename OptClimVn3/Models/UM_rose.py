# Class to support Unified Model running in Rose.
#This version is rather specialised for archer2.
# If running on other platforms then will need to refactor/generalise this code.
# It has some fairly large difference from Model. See class doc
# Things marker ARCHER2 are specific to ARCHER2. Generalise if on another platform.

# TODO - figure out what to do if the model fails. Coz often might fix in
#   cylc gui. But then model status won't get updated.
# But point of continue option is to automatically fix and run...

import fileinput
import functools
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

from ModelBaseClass import register_param, type_param_fn  #
from Model import Model
import pathlib
from namelist_var import NamelistVar, GroupConfig

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")  # have this anywhere you want logging


class UM_rose(Model):
    """
    Class to support the Unified model running in ROSE. This is for cylc7. See UM_rose_cylc8 for cylc8 version.
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
    suite_dir: typing.Optional[pathlib.Path]  # path for suite dir.
    model_data_dir: typing.Optional[pathlib.Path]  # path for model_data_dir.

    # test if we are on Archer by calling hostname -A and that stdout contains archer2.ac.uk
    archer2 = False
    stat = subprocess.run(['hostname', '-A'], capture_output=True, text=True)
    if not ((stat.returncode == 0) and 'archer2.ac.uk' in stat.stdout):
        my_logger.warning('Not running on archer2. This code will need re-writing to work on other platforms')
        archer2 = True
        # probably overkill. Only for instantiate
    # Get the user ID
    user_id = os.environ.get('USER') or os.environ.get('USERNAME')
    base_path = f'{user_id}/rose_optclim'  # which gives us path where files are stored.
    # ARCHER2
    puma_dir = pathlib.Path('/home/n02/n02-puma') / base_path  # puma2 root path on archer2.
    suite_file_name = None

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
        suite_dir: typing.Optional[str] = kwargs.pop("suite_dir", None)
        model_data_dir: typing.Optional[str] = kwargs.pop('model_data_dir', None)
        if model_data_dir is not None:
            self.model_data_dir = pathlib.Path(model_data_dir)
        else:
            self.model_data_dir = None
        super().__init__(*args, **kwargs)  # call the super-class init method.

        # deal with suite_dir -- where suite info gets written. Different from model_dir which is where
        # models are ran. This different from other models so far.
        if suite_dir is None:
            if self.name is not None:
                # Work out suite_dir which is where suite info gets written.
                # suite dir name is name_PID -- should be unique enough..
                # problem is that cylc uses a very flat space...
                self.suite_dir = self.puma_dir / f'{self.name}_{os.getpid()}'
            else:
                self.suite_dir = None
        else:
            self.suite_dir = pathlib.Path(suite_dir)
        self.configs = GroupConfig(root_dir=self.suite_dir)  # grouped configs for writing out generic namelists

        # modify parameters_no_key to include runModelTime, runUser, runCode, OPTCLIM_ARGS, runEnvSetup,prebuild if set.
        # Those parameters do not contribute towards the unique key used to identify the model.
        if self.run_info is not None:  # need to test for None as reloading of config gives us None.
            for key in ['runModelTime', 'runUser', 'runCode', 'OPTCLIM_ARGS', 'runEnvSetup', 'prebuild']:
                if self.run_info.get(
                        key) is not None:  # (Get None if either null in the original  json config or not present)
                    self.parameters_no_key[key] = self.run_info[key]
            # deal with prebuild set to True -- where we guess the path.
            if self.parameters_no_key.get('prebuild') is True:
                prebuild_path = self._guess_prebuild()
                if prebuild_path is not None:
                    self.parameters_no_key['prebuild'] = str(prebuild_path)
                else:
                    self.parameters_no_key.pop('prebuild')  # remove it as True won't work!
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
            self.parameters_no_key['runEnvSetup'] = str(runEnvSetup)  # need to convert to string.

            # deal with OPTCLIM_ARGS -- giving it a default value of ''
            self.parameters_no_key['OPTCLIM_ARGS'] = self.parameters_no_key.get('OPTCLIM_ARGS', '')
            # setup OPTCLIMTOP
            self.parameters_no_key['OPTCLIMTOP'] = str(genericLib.expand('$OPTCLIMTOP'))  # setup OPTCLIMTOP

        # set up MODEL_CONFIG to point to the configuration.
        if self.config_path is not None:
            self.parameters_no_key['MODEL_CONFIG'] = str(self.config_path)
            my_logger.debug(f"Set MODEL_CONFIG to {str(self.config_path)}")

        # set up OPTCLIM_SET_STATUS_SCRIPT
        if self.set_status_script is not None:
            self.parameters_no_key['OPTCLIM_SET_STATUS_SCRIPT'] = str(self.set_status_script)
            my_logger.debug(f'Set OPTCLIM_SET_STATUS_SCRIPT to {str(self.set_status_script)}')
        # set up submit and continue script. These need to be on puma2 so putting them in the suite_dir
        if self.suite_dir is not None:
            self.submit_script = self.suite_dir / 'submit_script.sh'  # script to run on puma2 to submit the job.
            self.continue_script = self.suite_dir / 'continue_script.sh'  # probably don't need this for now. There if have an error.


    def create_model(self,
                     direct: typing.Optional[pathlib.Path] = None,
                     copy_ref: bool = True
                     ) -> None:
        """
        Create the model. This is a rose specific version of create model. It will copy the reference config
         to **self.suite_dir** and create model_dir using super().create_model.
        :param direct: directory to create the model in.
        :param copy_ref: If True then copy the reference directory to the suite_dir.
        :return:nothing.
        """

        super().create_model(direct, copy_ref=False)  # create model dir
        super().create_model(direct=self.suite_dir, copy_ref=copy_ref)  # create the suite

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

    @staticmethod
    def change_rose_dir(direct: pathlib.Path) -> dict[pathlib.Path:int]:
        """
        Recursively change all values of [file:$ROSE_DATA/] to [file: in text files in input directory.
        :param direct: directory to change.
        :return dict indexed by file path and number of changes made.
        """
        #TODO -- might need to replace other ROSE_DATA references. Suchk it and see!
        raise NotImplementedError("Code no longer needed")
        match = r"(.*)\[file:\$ROSE_DATA/(.*?)\](.*)"
        replacement = r"\1[file:\2]\3"

        if not direct.is_dir():  # check direct is a directory!
            raise ValueError(f'path {direct} is not a directory.')

        modified_files = dict()
        for file in direct.iterdir():  # iterate over files in the directory
            if file.is_dir():
                my_logger.debug(f'{file} is a directory so calling change_rose_dir')
                modified_files.update(UM_rose.change_rose_dir(file))
            elif file.is_file() and genericLib.likely_text_file(file):  # likely a text file
                res = UM_rose.replace_file(file, match, replacement, backup_ext='.bak')  # do replacement
                if res is not None:  # any change?
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
        if (prebuild / 'extract').is_dir():  # dct  a dir which exists and had an extract.
            my_logger.warning(f'Guessed prebuild on Archer2 to be {prebuild}')
        else:
            my_logger.warning(f'Prebuild: {prebuild / "extract"} is not a directory')
            return None
        # but actually need path on puma2. Sigh!
        prebuild = self._puma_path(prebuild)  # get the puma path -- which is what is needed!
        return prebuild

    def modify_model(self):
        """
        UM rose specific version of modify model,
        Does the following (after calling the superclass method
        Copies the optclim suite apps into the suite_dir and modifies the suite_file_name
        Also generates the submit and continue scripts,

        """
        super().modify_model()  # call the base class method.
        # now do the specific stuff...
        self.copy_suite_apps()  # copy the optclim specific apps to the suite dir.
        self.update_suite_rc()  # update the suite.rc file
        # need to generate submit_cmd and continue_cmd
        # done here as super class instantiate cleans directory before doing anything else and then checks.
        # so need to call this in modify_model
        args = self.parameters_no_key.get('OPTCLIM_ARGS')
        self._create_script('submit', args=args)
        self._create_script('continue', args=args)

    def _create_script(self,
                       script_type: typing.Literal['submit', 'continue'],
                       args: typing.Optional[str] = None) -> None:
        """
        Create a script to run on puma2.
        :param script_type: type of script. 'submit' or 'continue'.
        :param args: any arguments to pass to the script.
        :return: nothing.
        """
        raise ValueError('TDo not call this version. Call the cylc7 version not this one ')

    def update_suite_rc(self):
        """
        Update the cylc/rose suite.rc to include OptClim tasks.
        If use_scratch set then update rose_suite.conf to use scratch space.
        """
        suite_file = self.suite_dir / self.suite_file_name
        genericLib.backup_file(suite_file, ext='.bak', create='copy')  # make a backup of the suite file.
        with suite_file.open('a') as suite_rc:
            suite_rc.write('\n\n%include optclim.rc\n')

        if self.run_info.get('use_scratch', False):
            self.set_scratch()  # cylc specific method to set scratch space.
            # can modify rose-suite.conf elsewhere so back this up with a different name.

    def set_scratch(self):
        """
        This version will raise notImplementedError as how done depends on the cylc version.
        :return:
        """
        raise NotImplementedError('This method is not implemented. It is cylc version specific. ')

    def copy_suite_apps(self):
        """Copy OptClim specific apps from the reference directory to the model directory"""
        optclim_tasks_root = pathlib.Path(__file__).parent / 'um_rose_files/optclim_tasks'
        shutil.copytree(optclim_tasks_root, self.suite_dir, dirs_exist_ok=True)

    def instantiate(self, fake: bool = False) -> None:
        """
        Instantiate the model. This is a UM_rose specific version of instantiate.
        Calls  superclass method and then tar/gzip the suite_dir and copy it to the model_dir.
        :param fake: passed to the super class method.
        :return:nothing.
        """

        super().instantiate(fake=fake)  # call super class method.
        # tar up the suite_dir and copy to model_dir.
        tar_file = self.model_dir / 'rose_suite.tar.gz'
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
            return False  # failed so return False.
        ## UM_rose specific checks.
        # 1) Check that START_TIME, RUN_TARGET and RESUB_TIME are compatible.
        # By converting them to Time Points and Durations we are also checking that
        # strings are valid.
        try:
            start_time = parse.TimePointParser().parse(self.read_param('START_TIME'))
            run_target = parse.DurationParser().parse(self.read_param('RUN_TARGET'))
            resub_time = parse.DurationParser().parse(self.read_param('RESUB_TIME'))
        except metomi.isodatetime.exceptions.ISO8601SyntaxError as err:  # catch any parsing errors.
            raise ValueError(f'Problem parsing one of START_TIME, RUN_TARGET or RESUB_TIME. {err}')
        # iterate from start_time  to start_time + run_target.
        # Doing this because months are not the same (second) duration throughout the year...
        end_time = start_time + run_target
        time = start_time
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
    def _puma_path(path: pathlib.Path) -> pathlib.PurePath:
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
        if cpts[1:4] != ('home', 'n02', 'n02-puma'):  # ignoring root so works on windows...
            my_logger.warning(f'path {path} does not start with /home/n02/n02-puma')
            return pathlib.PurePath(path)
        result = cpts[0:3] + tuple(['n02']) + cpts[4:]  # replace n02-puma with n02
        result = pathlib.PurePath(*result)  # convert to a pure path.
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
                base_dir = pathlib.Path(base_dir) / os.environ['DATAM']
                self.model_data_dir = base_dir
                my_logger.debug(f'Set model_data_dir to {self.model_data_dir}')

        self.set_status('RUNNING')  # update status and save to disk

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

        data_dir = self.model_dir / 'share/data/History_Data'  # where model data will be copied too
        if self.model_data_dir is None:  # not set trigger an error
            raise ValueError('model_data_dir not set. Something probably went wrong in running() method')

        if not self.model_data_dir.is_dir():
            raise FileNotFoundError(f'self.model_data_dir {self.model_data_dir} is not a dir')
        my_logger.debug(f"model_data_dir set and is {self.model_data_dir}")
        # create data dir if it does not exist.
        data_dir.mkdir(parents=True, exist_ok=True)

        # ready to go now.
        file_patterns = ['*.pp', '*.nc']  # file patterns to copy.
        # iterate over patterns to find list of files to copy.
        files_to_copy = []
        for fpattern in file_patterns:
            files = [file for file in self.model_data_dir.glob(fpattern) if file.is_file()]
            files_to_copy += files
        # deal with dumps
        # UM file names do sort alphanumerically so last file is most recent..
        # note if this method gets run more than once you will have more than dump file...
        dump_files = [f for f in self.model_data_dir.glob('*.d*_00') if f.is_file()]
        # and sort them!
        dump_files = sorted(dump_files)
        files_to_copy.append(dump_files[-1])  # last dump file.

        # now copy the files to the data_dir.
        for file in files_to_copy:
            new_file = data_dir / file.name
            my_logger.debug(f'Copying {file} to {new_file}')
            shutil.copy2(file, new_file)

        super().succeeded()  # call the superclass method.
        # Must occur after copying files otherwise data may not be accessible for the post-processing job.

    def submit_cmd(self) -> typing.List[str]:
        """"
        Generate the submission command. Overrides the super-class version.
        Will return ssh cmd that submits suite.
        """
        if self.status in ['INSTANTIATED', 'PERTURBED']:  # start again.
            script = self.submit_script
        elif self.status == 'CONTINUE':
            script = self.continue_script
            raise NotImplementedError('Continue not implemented yet')
        else:
            raise ValueError(f"Status {self.status} not expected ")
        # ARCHER2
        cmd = ['ssh', 'puma2', self._puma_path(script)]  # script to submit
        # Could make puma2 more generic -- so specified through run_info.
        # But no point at the moment as only running on archer2/puma2.
        # and I suspect that changes will need to be different on other platforms.
        # Where hopefully one can just run rose/cylc on the local machine.
        return cmd

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

    def _create_script(self,
                       script_type: typing.Literal['submit', 'continue'],
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
        script.parent.mkdir(parents=True, exist_ok=True)  # might need to create directory
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

    def set_scratch(self):
        """
        Set the scratch space for the UM_rose cylc7 model.
        This will update the rose-suite.conf file to use scratch space.
        :return: nothing.
        """
        # ARCHER2
        with fileinput.input(self.suite_dir / 'rose-suite.conf', inplace=True, backup='.bak_update') as f:
            for line in f:
                if f.isfirstline():  # first line
                    print('## Scratch space being used')
                    print(r'root-dir{share}=ln*=/mnt/lustre/a2fs-nvme/work/n02/n02/$USER')
                    print(r'root-dir{work}=ln*=/mnt/lustre/a2fs-nvme/work/n02/n02/$USER')

                print(line, end='')
        my_logger.debug('Using scratch space for share and work space')


class UM_rose_cylc8(UM_rose):
    """
    Class to support Unified Model running in ROSE cylc8.
    This is a subclass of UM_rose and adds the UM_rose specific parameters.
    """
    suite_file_name = 'flow.cylc'  # name of the suite file for cylc8

    # cylc 8 specific methods.
    # to use scratch space need to change site/archer2.rc in the [[HPC]] section change platform from archer2 to archer2-nvme
    # Can't be done by changing a variable as this is a jinja2 file :-(
    # also need to change archer2-bg to archer2-nvme-bg
    def set_scratch(self):
        # need to change site/archer2.rc in the [[HPC]] section change platform from archer2 to archer2-nvme
        # Can't be done by changing a variable as this is a jinja2 file :-(
        # also need to change archer2-bg to archer2-nvme-bg
        self.replace_file(self.suite_dir / 'rose-suite.conf',
                          r'^\s*platform\s*=\s*archer2\s*$',
                          'platform = archer2-nvme')
        self.replace_file(self.suite_dir / 'rose-suite.conf',
                          r'^\s*platform_bg\s*=\s*archer2-bg\s*$',
                          'platform_bg = archer2-nvme-bg')

    def _create_script(self,
                       script_type: typing.Literal['submit', 'continue'],
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
        else:
            raise ValueError(f'Unknown script type: {script_type}')
        script.parent.mkdir(parents=True, exist_ok=True)  # might need to create directory
        script.unlink(missing_ok=True)  # unlink it if it exists.
        with script.open('wt') as f:
            f.write('#!/bin/bash --login\n')
            cmd = ['cylc']
            if script_type == 'submit':
                cmd.append('vip')
            elif script_type == 'continue':
                cmd.append('play')
            else:
                raise ValueError(f'Unknown script_type {script_type}')
            if args:
                cmd.append(args)
            cmd.append(f'{self._puma_path(self.suite_dir)}')  # path to the suite dir.
            f.write(' '.join(cmd) + '\n')


# add in core optclim variables.
pth_c7 = pathlib.Path(__file__).parent / 'parameter_config/UM_rose_Parameters.csv'
UM_rose_cylc7.update_from_file(pth_c7, duplicate=True)

pth_c8 = pathlib.Path(__file__).parent / 'parameter_config/UM_rose_Parameters_cylc8.csv'
UM_rose_cylc8.update_from_file(pth_c8, duplicate=True)


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
            return value


        result = 13*[0.0]
        result[3:5] = [value, value]  # set the 4th and 5th elements (Needleleaf trees] to value.

        my_logger.warning('unload_rate_u needs verification')
        return [(nl, result)]



    @register_param('runModelTime')
    def run_time(self,
                 runTime: typing.Union[str, int, float, None],
                 transform:bool = True) -> \
            typing.Union[list[tuple[NamelistVar, str]], str]:
        """
        Set the run time for the model. This is in seconds or as an ISO duration string.
        UM wants it as a ISO duration string. This function will convert to that if needed.
        :param runTime: The run time in seconds or as an iso duration.
        :param transform -- does nothing. For compatability with other functions.
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
functions = [None, cca_md2dp_knob]
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


class UKESM1_1_c8(UM_rose_cylc8, UKESM1_params):
    """
    Class to support UKESM1_1 model running in ROSE with cylc 8.
    This joins the UM_rose_cylc8 and UKESM1_params classes.
    """

pth = pathlib.Path(__file__).parent / 'parameter_config/UM_rose_UKESM1_1.csv'
UKESM1_params.update_from_file(pth)  # load up the UKESM1_1 specific parameters.
UKESM1_1.update_from_file(pth, duplicate=False)
UKESM1_1_c8.update_from_file(pth, duplicate=False)

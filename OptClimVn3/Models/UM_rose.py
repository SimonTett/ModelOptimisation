# Class to support Unified Model running in Rose.
# Is going to drive a fairly extensive refactoring of param_info.
import logging
import os
import typing
import shutil

from scipy.constants import value

import genericLib

from ModelBaseClass import register_param # Used to allow functions. Currently none defined.
from Model import Model
import pathlib
from namelist_var import NamelistVar


my_logger = logging.getLogger(f"OPTCLIM.{__name__}") # have this anywhere you want logging

class UM_rose(Model):
    """
    Class to support the Unified model running in ROSE.
    The Complication is that this class will need to run a rose job on another super-computer


    """

    def __init__(self, *args, **kwargs):
        """
        Init the UM_rose instance. Calls the super-class init method.
        if run_info['runUser'] is not set, it will be set to the current user id on the current machine.
        :param args: positional args. Passed through to super-class
        :param kwargs: kwargs -- passed through to super-class
        """
        super().__init__(*args, **kwargs)  #
        # modify parameters_no_key to include runModelTime, runUser and runCode if set.
        # Those parameters do not contribute towards the unique key used to identify the model.

        for key in ['runModelTime','runUser','runCode']:
            if self.run_info.get(key) is not None: # (Get None if either null in the original  json config or not present)
                self.parameters_no_key[key] = self.run_info[key]
        # switch off fixed scripts. Will change submit_cmd instead.
        self.submit_script = None
        self.continue_script = None # probably don't need this for now. There if have an error.
        # I think ROSE handles that kind of stuff so just need to resubmit the config on puma.
        # Alternatively (if easier) modify the super class submit method



    def create_model(self):
        """
        Create the model. This is a rose specific version of create model.

        """
        super().create_model()
        # do any rose specific stuff here. Though most of it will be in modify_model which gets called by create_model.

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
        self.update_suite_rc()
        self.copy_suite_apps()

    def update_suite_rc(self):
        """Update the cylc/rose suite.rc to include OptClim tasks"""
        with open(self.model_dir / 'suite.rc', 'a') as suite_rc:
            suite_rc.write('\n\n%include optclim.rc\n')

    def copy_suite_apps(self):
        """Copy OptClim specific apps from the reference directory to the model directory"""
        optclim_tasks_root = self.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/optclim_tasks')
        shutil.copytree(optclim_tasks_root, self.model_dir, dirs_exist_ok=True)

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



pth = pathlib.Path(__file__).parent /'parameter_config/UM_rose_Parameters.csv'
UM_rose.update_from_file(pth, duplicate=True)

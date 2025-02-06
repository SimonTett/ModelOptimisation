# Class to support Unified Model running in Rose.
# Is going to drive a fairly extensive refactoring of param_info.
import logging
import typing


from ModelBaseClass import register_param # Used to allow functions. Currently none defined.
from Model import Model
import pathlib


my_logger = logging.getLogger(f"OPTCLIM.{__name__}") # have this anywhere you want logging

class UM_rose(Model):
    """
    Class to support the Unified model running in ROSE.
    Complication is that this class will need to run a rose job on another super-computer


    """

    def __init__(self, *args, **kwargs):
        """
        Init the UM_rose instance. Calls the super-class. May need to set some variables.
        :param args: positional args. Passed through to super-class
        :param kwargs: kwargs -- passed through to super-class
        """
        super().__init__(*args, **kwargs)  #
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



pth = pathlib.Path(__file__).parent /'parameter_config/UM_rose_Parameters.csv'
UM_rose.update_from_file(pth, duplicate=True)

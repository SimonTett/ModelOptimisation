"""
simple model for testing. With parameters in json file.  Model does very little!

"""
import functools
import sys
import typing
import logging
import fileinput
import json
import re
import platform

import numpy as np

from Model import Model
from ModelBaseClass import register_param
import pathlib
import copy
from namelist_var import NamelistVar, type_allowed_fortran,GroupConfig
from Model import Model


namelist_var = functools.partial(NamelistVar, type_name='json_nl')  # make it easier to create NamelistVar
my_logger = logging.getLogger(f"OPTCLIM.{__name__}")
class simple_model_pars_json(Model):
    #StudyconfigPath:pathlib.Path
    # simple model.. Need to have its own  version of submit_cmd, modify_model, perturb
    # all other methods are as Model.

    def __init__(self, *args, study= None, **kwargs): # study should be a study but study imports model.
        """
        simple_model init -- calls super class -- see Model.__init__ for documentation on these.
        :param args:  arguments
        :param study: a Study.
        :param kwargs: keyword arguments
        This makes use of study and stores the path to the configuration in self.StudyConfig_path.
        If study is None then self.StudyConfig_path is None.
        """

        super().__init__(*args, study=study, **kwargs)  # call the super class init.
        self.StudyConfig_path = None  # setup StudyConfig_path attribute.
        if study is not None:
            self.StudyConfig_path = study.config.fileName()  # store the path to the config.
            if not self.StudyConfig_path.is_absolute(): # not absolute so make it so
                self.StudyConfig_path = pathlib.Path.cwd()/self.StudyConfig_path
        self.submit_script = pathlib.PurePath('run_simple_model_pars_json.py')
        self.continue_script = self.submit_script # continue is just submit


    # model generation fns.

    @register_param("RHCRIT")
    def cloud_RH_crit(self,
                      rhcrit: typing.Optional[type_allowed_fortran]
                      ) -> typing.Union[type_allowed_fortran,tuple[NamelistVar, type_allowed_fortran]]:
        """
        Set rhcrit to value.
        :param value: value to set rhcrit to.
        :return: list of tuples (NamelistVar,value)
        """
        # Code to read number of vertical levels from the config
        # not used as this model is really for testing
        #n_level_nl = namelist_var(nl_var='N_VERT_LEVELS', namelist='model_params',
        #                          filepath=pathlib.Path('parameters.json'), default=3)
        #nvert:int = self.configs.read_value(n_level_nl)
        nvert = 3 # hard coded for now.
        rhcrit_nl = namelist_var(nl_var='RHCRIT', namelist='model_params',
                                 filepath=pathlib.Path('parameters.json'), default=0.7)

        if rhcrit is None:
            cloud_rh_crit:type_allowed_fortran = self.configs.read_value(rhcrit_nl)
            rhcrit:float = cloud_rh_crit[0]
            expected = nvert * [rhcrit]
            if expected != cloud_rh_crit:
                raise ValueError(f"Expected rhcrit {np.array(expected)} but got {np.array(cloud_rh_crit)}")
            return rhcrit
        else:
            cloud_rh_crit:list[float] = nvert * [rhcrit]
        return rhcrit_nl, cloud_rh_crit


    def create_cmd(self, status:str, modifystr:str,indent:int=0) -> typing.List[str]:
        """
        Create the string to run a cmd to set status.
        :param status: status to set.
        :param indent: How much to indent lines
        :return: list of strings to print to model script file
        """
        #TODO -- add way of setting debug status for self.set_status_script

        cmd =[f'"{self.set_status_script}"', f'"{self.config_path}"', status]
        cmd = [f'{self.set_status_script}', '-v',f'{self.config_path}', status] # verbose
        if platform.system() == 'Windows':
            cmd = [sys.executable]+cmd



        # run command and get some diagnostics
        lst=[f'cmd = {cmd} {modifystr}',
             f'result = subprocess.run(cmd) {modifystr}',
             f'if result.returncode != 0: {modifystr}',
             f'    print(f"Command {cmd} failed") {modifystr}',
             f'    result.check_returncode() {modifystr}'
         ]
        space = " "*indent
        lst =[space+l for  l in lst ]
        return lst

    def modify_model(self):
        """
        Make changes to run_simple_model_pars_json.py
        Adds in cmds to set status.
        :return: nada
        """
        super().modify_model()
        pth = self.model_dir / 'run_simple_model_pars_json.py'
        modifystr = '## modified'
        with fileinput.input(pth, inplace=True, backup='.bak') as f:
            for line in f:
                if re.search(modifystr, line):
                    raise Exception("Already modified Script")
                elif f.isfirstline():  # first line
                    # we are running so set status to RUNNING.
                    print(line[0:-1])  # print line out.
                    print(f"import subprocess {modifystr}") # import subprocess.
                    print("\n".join(self.create_cmd('RUNNING',modifystr)))
                elif re.match(r'^\s+sys.exit\([1-9]\)', line):  # Failed
                    # 50/50 chance it sets FAILED or just crashed!
                    cmd = " \n".join(self.create_cmd('FAILED',modifystr,indent=8))
                    print(f"""
    if np.random.uniform(0,1.0) < 0.5: {modifystr}
        pass # nothing done {modifystr}
    else: {modifystr}
{cmd}""") # cmd already has indention included.
                    print(line[0:-1])  # print out the original line.
                elif re.match(r'^\s*sys.exit\(0\)', line):  # Succeeded
                    print("\n".join(self.create_cmd('SUCCEEDED', modifystr)))  # finished  so have a succeeded status.
                    print(line[0:-1])  # print out the original line.
                else:
                    print(line[0:-1]) # print out the original line.



    # over write submit_cmd
    def submit_cmd(self) -> typing.List[str]:
        """"
        Generate the submission command.
        If status is INSTANTIATED or PERTURBED then submit to the Q run_simple_model.py
        """
        runTime = self.run_info.get('runTime', 30)  # default is 30 seconds.
        runCode = self.run_info.get('runCode')
        if self.status in ['INSTANTIATED', 'PERTURBED']:
            script = self.submit_script
        elif self.status == 'CONTINUE':
            script = self.continue_script
        else:
            raise ValueError(f"Status {self.status} not expected ")
        # just use the submit.
        outdir = self.model_dir / 'model_output'
        outdir.mkdir(parents=True, exist_ok=True)
        my_logger.debug(f"Created {outdir}")
        cmd = self.engine.submit_cmd([self.model_dir/script, str(self.StudyConfig_path)],
                                     f"{self.name}{len(self.model_jids):05d}",
                                     outdir,
                                     run_code=runCode, time=runTime,
                                     rundir=self.model_dir)

        return cmd

    def perturb(self, parameters: typing.Optional[dict] = None):
        """
        Sets fail_probability to 0.0 so stopping and random failure.
        :return: Nada!
        """
        return super().perturb(parameters=dict(fail_probability=0.0))


    def archive(self,
                archive: "tarfile.TarFile",
                rootDir: pathlib.Path,
                extra_files: typing.Optional[typing.List[typing.Union[pathlib.Path,str]]] = None):
        
        if extra_files is None:
            extra_files = []
            

        # generate list of extra files that are to be archived.
        files_to_archive = extra_files + ["parameters.json",
                                          "run_simple_model_pars_json.py",
                                          "model_output.json",
                                          "input.json"]

        return super().archive(archive,rootDir, 
                               extra_files=files_to_archive)




pth = pathlib.Path(__file__).parent /'parameter_config/simple_model_Parameters.csv'
simple_model_pars_json.update_from_file(pth, duplicate=True)


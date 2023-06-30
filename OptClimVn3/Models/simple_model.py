"""
simple model for testing. Model does very little!
"""
import typing
import logging

import json

from Model import Model
import engine # need the engine!


class simple_model(Model):
    # simple model.. Need to have personal version of submit_cmd and set_params
    # all other methods are as Model.

    def set_params(self, parameters: typing.Optional[dict] = None):
        """
        For simple_model just dump stuff to a json file
        :param parameters: parameters to be written out
        :return: Nada
        """

        out_file = self.model_dir / "params.json"
        with open(out_file, 'w') as fp:
            json.dump(parameters, fp, allow_nan=False)

        logging.debug(f"Dumped parameters to {out_file}")

    # over write submit_cmd
    def submit_cmd(self,run_info,
                   engine:engine) -> typing.List[str]:
        """"
        Generate the submission command.
        :param run_info # run_info. only runTime and runCode are used
        :param engine # engine for submission.
        If status is INSTANTIATED or PERTURBED then submit to the Q simple_model.py
        """
        runTime = run_info.get('runTime',30) # default is 30 seconds.
        runCode = run_info.get('runCode')
        if self.status in ['INSTANTIATED', 'PERTURBED']:
            script = "simple_model.py"
        elif self.status == 'CONTINUE':
            raise ValueError("No CONTINUE allowed for simple_model.py")
        else:
            raise ValueError(f"Status {self.status} not expected ")
        # just use the submit.
        outdir = self.model_dir/'model_output'
        outdir.mkdir(parents=True,exist_ok=True)
        cmd = engine.submit_cmd([script], f"{self.name}{self.run_count:05d}", outdir,
                                runCode=runCode, runTime=runTime)

        return cmd

    def perturb(self, parameters:typing.Optional[dict]=None):
        """
        Does nothing so will raise an error
        :param parameters: dict of parameters to set.
        :return: Nada!
        """
        raise NotImplementedError("Implement perturb")


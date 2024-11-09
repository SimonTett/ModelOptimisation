# generalisation of Model to allow testing
import pathlib
from Model import Model,register_param
from namelist_var import namelist_var,type_allowed_fortran
import logging
import typing
my_logger = logging.getLogger(f"OPTCLIM.{__name__}")
class myModel(Model):
    @register_param('RHCRIT')
    def cloudRHcrit(self, rhcrit:typing.Optional[type_allowed_fortran]):
        """
        Compute rhcrit on multiple model levels
        :param rhcrit: meta parameter for rhcrit. If None relationship will be inverted.
        :return: (value of meta parameter if inverse set otherwise
           a tuple with namelist_var infor and  a list of rh_crit on model levels

        """
        # Check have 19 levels.
        rhcrit_nl = namelist_var(filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT', namelist='RUNCNST')
        curr_rhcrit = rhcrit_nl.read_value(dirpath=self.model_dir)
        if len(curr_rhcrit) != 19:
            raise ValueError("Expect 19 levels")
        inverse = rhcrit is None
        if inverse:
            return rhcrit_nl.read_value(dirpath=self.model_dir)[3]
        else:
            cloud_rh_crit = 19 * [rhcrit]
            cloud_rh_crit[0] = max(0.95, rhcrit)
            cloud_rh_crit[1] = max(0.9, rhcrit)
            cloud_rh_crit[2] = max(0.85, rhcrit)
            return rhcrit_nl, cloud_rh_crit
    @register_param('bad') # bad function
    def bad(self,value:typing.Optional[type_allowed_fortran]):
        """
        A bad function
        """
        return value
    @register_param('multi_var')
    def multi_var(self,
                  value:typing.Optional[float]) -> typing.Union[list[float,float],list[tuple[namelist_var,float]]]:
        """
        A function with multiple namelist_var
        """
        multi_var = [
            namelist_var(filepath=pathlib.Path('CNTLATM'), nl_var='LATITUDE_BAND', namelist='RUNCNST',default=0.0),
            namelist_var(filepath=pathlib.Path('CNTLATM'), nl_var='TOWER_FACTOR', namelist='RUNCNST',default=0.0)
        ]
        if value is None:
            return [multi_var[0].read_value(dirpath=self.model_dir),multi_var[1].read_value(dirpath=self.model_dir)]
        else:
            result = [(a,value) for a in multi_var]
            return result

myModel.add_param_info({'ANVIL_FACTOR': namelist_var(filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT', namelist='RUNCNST',default=0.0)})
myModel.add_param_info({'RHCRIT': namelist_var(filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT2', namelist='RUNCNST',default=0.8)})
myModel.add_param_info({'fred':2}) # should trigger an error.
pth = pathlib.Path(__file__).parent.parent /'parameter_config/HadCM3_Parameters.csv'
myModel.update_from_file(pth)
# generalisation of Model to allow testing
import pathlib
from Model import Model,register_param
from namelist_var import NamelistVar,type_allowed_fortran
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
        rhcrit_nl = NamelistVar(type_name='namelist_var',filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT', namelist='RUNCNST')
        curr_rhcrit = self.read_nl_value(rhcrit_nl)
        if len(curr_rhcrit) != 19:
            raise ValueError("Expect 19 levels")
        inverse = rhcrit is None
        if inverse:
            return self.read_nl_value(rhcrit_nl)[3]
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
                  value:typing.Optional[float]) -> typing.Union[list[float,float],list[tuple[NamelistVar,float]]]:
        """
        A function with multiple namelist_var
        """
        multi_var = [
            NamelistVar(type_name='namelist_var', filepath=pathlib.Path('CNTLATM'), nl_var='LATITUDE_BAND', namelist='RUNCNST',default=0.0),
            NamelistVar(type_name='namelist_var',filepath=pathlib.Path('CNTLATM'), nl_var='TOWER_FACTOR', namelist='RUNCNST',default=0.0)
        ]
        if value is None:
            result:list[float,float] = [self.read_nl_value(var) for var in multi_var]
            return result

        else:
            result = [(a,value) for a in multi_var]
            return result

myModel.add_param_info({'ANVIL_FACTOR': NamelistVar(type_name='namelist_var',filepath=pathlib.Path('CNTLATM'), nl_var='ANVIL_FACTOR', namelist='RUNCNST',default=0.0,name='ANVIL_FACTOR')})
myModel.add_param_info({'RHCRIT': NamelistVar(type_name='namelist_var',filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT2', namelist='RUNCNST',default=0.8,name='RHCRIT2')})
myModel.add_param_info({'fred':2}) # should trigger an error.
pth = pathlib.Path(__file__).parent.parent /'parameter_config/HadCM3_Parameters.csv'
myModel.update_from_file(pth)
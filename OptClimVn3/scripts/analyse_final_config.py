# Code to read in and analyse the final config.
import StudyConfig
import pathlib
filepath=pathlib.Path(r"C:\Users\stett2\Downloads\gamil_14prts_final.json")

config = StudyConfig.readConfig(filepath)
DFOLS=config.DFOLS_config()
trans_jac = config.transJacobian()
mat = config.transMatrix()
jac= mat@trans_jac@(mat.T) # jacobian in "real space"
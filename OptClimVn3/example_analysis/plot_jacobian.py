# Code to read in final config and plot the jacobian.
# jacobian is normalised by the error and the range of the parameters.
# So values are error change for parameter change from min to max.
import pandas as pd

import StudyConfig
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
filepath=pathlib.Path(r"wenjun_fix/gamil_14prts_final.json")

config = StudyConfig.readConfig(filepath)
DFOLS=config.DFOLS_config()
jac = config.jacobian()
# now to scale by error and range.
cov = config.Covariances(scale=True)['CovTotal'] # Need scaling. Really should get that from config.
error = pd.Series(np.sqrt(np.diag(cov)),index=cov.index)
scale = config.paramRanges()
p_scale = scale.loc['maxParam']-scale.loc['minParam']
norm_jac = (jac.T/error).T*p_scale
# noramlised Jacobian -- parameter scaling is for change from min to max.
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,11),clear=True,num='norm_jacobian')
sns.heatmap(norm_jac,ax=ax,cmap='coolwarm',center=True,robust=True,annot=True,vmin=-20,vmax=20,fmt='3.0f')
ax.tick_params(labelsize='x-small',labelrotation=45)
ax.set_title('Normalised Jacobian')
fig.show()
# Get the Hessian, normalised by param range **2 and plot it
hessian = config.hessian()
norm_hess =((hessian*p_scale).T*p_scale).T
scale = 10**np.floor(np.log10(norm_hess.max().max()))
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,11),clear=True,num='norm_hessian')
sns.heatmap(norm_hess/scale,ax=ax,cmap='coolwarm',center=True,vmin=0,robust=True,annot=True)#,fmt='3.0f')
ax.tick_params(labelsize='x-small',labelrotation=45)
ax.set_title(f'Normalised Hessian/{scale:.0g}')
fig.show()

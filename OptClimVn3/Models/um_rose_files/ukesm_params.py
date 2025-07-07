# bit of scratch code to work out parameter values given parameter names.
import json
import pathlib
import typing
import datetime

import metomi.rose.config
import pandas as pd
import genericLib
from  namelist_var import UMroseNamelistConfig as UMroseNamelistConfig,type_allowed_fortran


import hashlib
import requests

from test_data.gamil_test.generic_json_wenjun import my_logger


def read_cached_csv(url, cache_dir:typing.Optional[pathlib.Path]=None):
    """
    Download a CSV file from a URL and cache it locally in cache_dir.
    If the file is already cached, read from the cache.
    """
    if cache_dir is None:
        cache_dir = pathlib.Path.cwd()/'cache'  # Default cache directory is current working directory

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use a hash of the URL as the filename to avoid collisions
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    local_file = cache_dir / f"{url_hash}.csv"
    if not local_file.exists():
        # Download the file
        raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/') # deal with github blob URLs
        r = requests.get(raw_url)
        r.raise_for_status()
        with open(local_file, 'wb') as f:
            f.write(r.content)
    return pd.read_csv(local_file)

def find_param(config:metomi.rose.config,param:str) -> typing.Optional[tuple[str,type_allowed_fortran]]:
    """
Find the parameter in the configuration and return its group name and default value. None if not found
    :param config:  UM configuration object.
    :param param: parameter name to find in the configuration.
    :return: group name and default value of the parameter if found, otherwise None.
    """
    for group, sub_node in config.walk(no_ignore=True):
        value = sub_node.get([param], None)  # get the value of the parameter if it exists
        if value is not None :
            default = UMroseNamelistConfig.parse_value(value.value)
            group_name = group
            return group_name, default
    return None  # if the parameter is not found, return None

genericLib.setup_env()
my_logger = genericLib.setup_logging(1)
# Get QUMP HadGEM3 parameters from a CSV file on GitHub. Using GA9Parameters.csv for UKESM1.1.
# Note defaults are from the UKESM1.1 configuration and warning will be printed if the QUMP value is different.
source_url = 'https://github.com/qump-project/qump-hadgem3/blob/master/data/GA7Parameters_fakedata.csv'
output_param = genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/parameters_UKESM1_1.ijson')
simple_param_csv = genericLib.expand('$OPTCLIMTOP/OptClimVn3/Models/parameter_config/UM_rose_UKESM1_1.csv')
#source_url = 'https://github.com/qump-project/qump-hadgem3/blob/master/data/GA7Parameters_fakedata.csv'
df = read_cached_csv(source_url)  # read in the spreadsheet with the names of the parameters we are interested in.
df= df.iloc[0:24,:] # take the first 24 rows of the dataframe. This is the Meta data.
# get rid of unwanted columns and make an index
df = df.drop(['ROW TYPE','Realisation','Initialisation','Parameters','delta_toa','co2_pathway_file'],
             errors='ignore',axis=1)
df = df.set_index('ATTRIBUTE_NAME').T
df.index.name = 'parameter'  # set the index name to 'parameter'

# convert things that should be strings to strings
str_cols = ['longName','units','namelistFile','namelist','transformFunc']
df[str_cols] = df[str_cols].astype(str)  # convert to string
flt_values = ['standardValue','min','max']  # columns that should be floats
df[flt_values] = df[flt_values].apply(pd.to_numeric, errors='coerce')  # convert to float, coerce errors to NaN

## get in the reference. Used to see if variables set.
ref_model_vars = pathlib.Path(genericLib.expand(
    '$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898/app/um/rose-app.conf')) # ROSE config
with ref_model_vars.open('rt') as fp:
    ref_config = metomi.rose.config.load(fp)
# extract the parameter names from the spreadsheet
# find fns.
# find values we have by looking in the config
ukesm_values = dict()
missing_params = []
for name,series in df.iterrows():
    if series.namelist.lower == 'nan':
        my_logger.error(f'No namelist specified for {name}')
        continue
    nl = ['namelist:'+series.namelist,name]
    value = ref_config.get(nl)
    if value is not None:
        value = UMroseNamelistConfig.parse_value(value.value)
        ukesm_values[name] =value
        if isinstance(value, list) and (series.transformFunc == 'nan'):  # need a fn to make a list
            df.loc[name,'transformFunc'] = f'fn_list_{name}'
            my_logger.info(f' {name} needs fn to make list')

    else:
        my_logger.warning(f'No value found for {name} using {nl} in reference config')
        missing_params.append(name)

L = df.index.isin(ukesm_values.keys())
df_present = df[L]
# sort by namelist and then variable name
df_present = df_present.sort_values(by=['namelist',df_present.index.name])
fns_mask = df_present.transformFunc != 'nan'
params = df_present[~fns_mask].index.tolist()  # get the column names as a list
params_fn = df_present[fns_mask].index.tolist()
L=df_present.transformFunc.str.startswith('fn_list_')
fn_list = df_present.index[L].tolist()

print('Missing parameters: ')
for param in missing_params:
    print(param,df.loc[param,'namelist'])
print('+++++++\nFunctions: ')
for param in set(params_fn)-set(fn_list):
    print(param,df.loc[param,['namelist','transformFunc']].values)
print('+++++++\nFunctions to list: ')
for param in  set(fn_list):
    print(param,df.loc[param,['namelist','transformFunc']].values)

print('\n================================================================')



## Make up config file. Best way is to generate a pandas dataframe
df_params = pd.DataFrame(dict(parameter=params,
                              type='um_rose',
                              filepath='app/um/rose-app.conf',
                              namelist=['namelist:'+nl for nl in df_present.loc[params].namelist],
                              nl_var=params,
                              default=[ukesm_values[p] for p in params],
                              name=None,
                              function_name=None)).set_index('parameter')
# and make the parameter config file.
df_params.to_csv(simple_param_csv)

param_dict=dict(
    _comment = f'File auto-generated from HadGEM3 parameters spreadsheet. Version 0.1 of {datetime.date.today().isoformat()}',
    number_params_comment=f'This file has {len(params)} parameters.',
    sourceurl_comment = f'Source URL for HadGEM3 parameters: {source_url}',
    defaultParams=dict(
        comment = "Default parameters for UKESM1.1. These are the parameters that are used if not specified in the configuration file.",
    ),
    minmax=dict(
        comment= "Defines the minimum and maximum ranges for UKESM parameters. Must be defined for ALL parameters",
    )
)
# this only works for simple parameters where there is a 1-1 matching between the parameter name and the value.
#TODO -- add functions to deal with the parameters that need function calls to get the value.
# some of which convert a single value to a list of values.
for param in params:
    qump_value = df.loc[param,'standardValue']
    ukesm_value = ukesm_values[param]
    if qump_value != ukesm_value:
        my_logger.warning(f'QUMP value for {param} is {qump_value} but UKESM1.1 value is {ukesm_value}. Using UKESM1.1 value.')

    param_dict['defaultParams'][param] = ukesm_value
    if df.loc[param,'longName'] != 'nan':  # if the long name is not NaN, use it as the comment
        param_dict['defaultParams'][param+'_comment'] = df.loc[param,'longName']
    param_dict['defaultParams'][param+'namelist_comment'] = df.loc[param,'namelist']  # provide the namelist as a comment

    param_dict['minmax'][param] = df.loc[param,['min','max']].to_list() # set min and max

# add in fn_params.
param_dict['defaultParams']['fn_params_comment'] = f'{len(params_fn)} function params of which {len(fn_list)} produce lists'
for param in params_fn:
    qump_value = df.loc[param,'standardValue'] # no UKESM1.1 value for now. TODO implement methods and then invert them,
    param_dict['defaultParams'][param] = qump_value
    param_dict['defaultParams'][param+'_fn_comment'] = df.loc[param,'transformFunc']
    if df.loc[param,'longName'] != 'nan':
     param_dict['defaultParams'][param+'_comment'] = df.loc[param,'longName']
    param_dict['minmax'][param] = df.loc[param,['min','max']].to_list() # set min and max

with output_param.open( 'wt') as f:
    json.dump(param_dict,f,indent=4,sort_keys=False)  # write the dictionary to a JSON file with indentation and sorted

print(f'Have {len(params)} simple parameters and {len(params_fn)} function parameters.')





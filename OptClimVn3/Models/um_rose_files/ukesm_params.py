# Code to convert MO QUMP data to UKESM1.1 parameters suitable for use in OptClim.
# TODO: Deal with latent parameters that are not in the namelist but are in the QUMP data.
#   These parameters do not change namelists but do impact variables which are in the namelist.
# Cases are liu_latent (impacting bparam), rho_snow_et_crit_delta (impacting rho_snow_et_crit)
# and allicetdegc0to1 (impacting allicetdegc). Best approach is to have a function that sets the params they impact
# and things they depend on and reads the latent param from the parameters attribute.
# These should not go in the simple parameters file.
# but need default values and ranges for the ijson file.
# They also need functions generated which use them and dummy functions for the latent parameters which do nothing.
# I wonder if simple parameters should have some null that means do nothing??
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
from UM_rose import UKESM1_1_c8 # import UKESM1_1 to get the functions which we then invert to get default values.
import tempfile
import numpy as np



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

def read_qump_data(url:str,
                   cache_dir:typing.Optional[pathlib.Path]=None,
                   extract_row:str = 'Parameter') -> pd.DataFrame:
    """
    Read the QUMP HadGEM3 parameters from a CSV file on GitHub.
    :param url: URL of the CSV file.
    :param cache_dir: Directory to cache the CSV file. Passed to read_cached_csv.
    :param extract_row: The row type to extract from the CSV file. Default is 'Parameter'.
    :return: DataFrame containing the parameters.
    """
    df = read_cached_csv(url,cache_dir=cache_dir)  # read in the spreadsheet with the names of the parameters we are interested in.
    df = df[df.loc[:,'ROW TYPE']==extract_row] # want the meta-data for the parameters, not the data.
    # get rid of unwanted columns and make an index
    df = df.drop(['ROW TYPE', 'Realisation', 'Initialisation', 'Parameters', 'delta_toa', 'co2_pathway_file'],
                 errors='ignore', axis=1)
    df = df.set_index('ATTRIBUTE_NAME').T
    df.index.name = 'parameter'  # set the index name to 'parameter'

    # convert things that should be strings to strings
    str_cols = ['longName', 'units', 'namelistFile', 'namelist', 'transformFunc','dependency','printInNamelist']
    if 'printInNamelist' not in df.columns:  # if the column is not present, add it.
        df['printInNamelist'] = 'TRUE'
    df[str_cols] = df[str_cols].astype(str)  # convert to string
    flt_values = ['standardValue', 'min', 'max']  # columns that should be floats
    df[flt_values] = df[flt_values].apply(pd.to_numeric, errors='coerce')  # convert to float, coerce errors to NaN

    return df

def read_qump_experiments(url:str,
                          cache_dir:typing.Optional[pathlib.Path]=None,
                          extract_row:str = 'Experiment',
                          start_count:int = 0) -> pd.DataFrame:
    """
    Read the QUMP HadGEM3 experiments from a CSV file on GitHub.
    :param url: URL of the CSV file.
    :param cache_dir: where to cache the CSV file. Passed to read_cached_csv.
    :param extract_row: Name of the row type to extract from the CSV file. Default is 'Experiment'.
    :return: dataframe containing the experiments.
    """
    df = read_cached_csv(url,cache_dir=cache_dir)  # read in the spreadsheet with the names of the parameters we are interested in.
    df = df[df.loc[:,'ROW TYPE']==extract_row] # want the data for the experiments, not the meta-data.
    # get rid of unwanted columns and make an index
    df = df.drop(['ROW TYPE', 'Realisation', 'Initialisation', 'Parameters', 'delta_toa', 'co2_pathway_file'],
                 errors='ignore', axis=1)
    df = df.set_index('ATTRIBUTE_NAME')
    df.columns.name = 'parameter'  # set the index name to 'parameter'
    df.index = [f'Case#{v+start_count:d}' for v in range(0, len(df.index))]
    df.index.name = 'case'  # set the index name to 'case'
    df = df.apply(pd.to_numeric) # convert all values to numeric
    return df

def lin_reg(df:pd.DataFrame,
            yparam:str='None',
            xparams:typing.Optional[list[str]]=None) -> dict[str,float]:
    """
Perform a linear regression on the given parameters.
    :param df: DataFrame containing the parameters.
    :param yparam: y parameter to use in the regression.
    :param xparams: x parameters to use in the regression.
    :return: dict of parameter names and their coefficients.
    """
    if xparams is None:
        xparams = ['None']
    from sklearn.linear_model import LinearRegression


    # Fit the model
    model = LinearRegression()
    # Extract the X and y values from the DataFrame
    data = df.loc[:, xparams + [yparam]].dropna()  # get the x and y parameters from the dataframe
    X = data.loc[:,xparams].values  # get the x parameters as a numpy array
    y = data[yparam].values.reshape(-1, 1)  # get the y parameter as a 2D numpy array
    model.fit(X, y)

    # Get the coefficients and intercept
    coeffs:list[float] = model.coef_[0]  # get the coefficients as a 1D array
    intercept:float = model.intercept_[0]  # get the intercept as a float
    result = {f'{xparam}_coef': coeff for xparam, coeff in zip(xparams, coeffs)}
    result['intercept'] = intercept  # add the intercept to the result
    return result

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


def generate_method(method_name:str,
                    functions:list[typing.Optional[str]]=[],
                    namelists:list[str]=[],
                    nl_vars:list[str]=[]):
    """
    Generate a method skeleton for the given functions, namelists and nl_vars.

    :param functions: List of functions
    :param namelists: list of namelists to set.
    :param nl_vars:  list of variables
    :return:  String containing the function skeleton.
    Uses     def fn_param(self,
                   value: typing.Optional[float],
                   transform: bool = True,
                   transform_values: typing.Optional[dict[NamelistVar,typing.Optional[typing.Callable]]] = None
                    ) -> type_param_fn:
    """

    nl_str= [ f"NamelistVar('um_rose', filepath=self.um_namelist_file, namelist='namelist:{namelist}', nl_var='{nl_var}')"
              for namelist,nl_var in zip(namelists,nl_vars)]
    nl_str = f'namelists=[{",\n    ".join(nl_str)}]'  # make a list of namelist variables to set.
    fn_str = [f"{fn}" for fn in functions]  # stringy functions
    fn_str = f'functions=[{", ".join(fn_str)}]'  # make a list of functions to apply.
    transform_str = 'transform_values=dict(zip(namelists,functions))'  # make a list of transform functions to apply.
    fn_def = "    "+nl_str+"\n    "+fn_str+"\n    "+transform_str+"\n    "  # combine the strings into a single string.
    fn_def += f"""
    UKESM1_params.register_param_with_partial(
            '{method_name}', # name to register the function as.
            UKESM1_params.fn_param, # using fn_param as the base function.
            transform_values=transform_values # transform values to apply.
            )  # generate/register the function. Do  impliment functions (see functions)
    """
    return fn_def

import math

def round_sig(x, sig=3):
    # thanks chat gpt
    if x == 0:
        return 0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)



genericLib.setup_env()
my_logger = genericLib.setup_logging('INFO') # want info level logging for this script.

# Get QUMP HadGEM3 parameters from a CSV file on GitHub. Using GA9Parameters.csv for UKESM1.1.
# Note defaults are from the UKESM1.1 configuration and warning will be printed if the QUMP value is different.
source_urls = ['https://github.com/qump-project/qump-hadgem3/blob/master/data/GA7Parameters_fakedata.csv',
                'https://github.com/qump-project/qump-hadgem3/blob/master/data/GA8Parameters.csv',
               'https://github.com/qump-project/qump-hadgem3/blob/master/data/GA9Parameters.csv'
               ]
output_param = genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/parameters_UKESM1_1.ijson')
simple_param_csv = genericLib.expand('$OPTCLIMTOP/OptClimVn3/Models/parameter_config/UM_rose_UKESM1_1.csv')
#source_url = 'https://github.com/qump-project/qump-hadgem3/blob/master/data/GA7Parameters_fakedata.csv'

# remove cases where variable has dependency -- those get handled by functions.
# rename some index members.
df = pd.concat([read_qump_data(url) for url in source_urls]).groupby(level=0).last()
# check for uniqueness
if not df.index.is_unique:
    raise ValueError('QUMP HadGEM3 parameters are not unique. ')

# get in all the dataframes just keeping the last one for each parameter.

exper_data = []
count = 0
for url in source_urls:
    exp_df=read_qump_experiments(url,start_count=count)  # read in the experiments data.
    count += len(exp_df)  # increment the count by the number of experiments in the last dataframe.
    exper_data.append(exp_df)  # append the experiments data to the list.
df_exper = pd.concat(exper_data, axis=0)  # concatenate the experiments data into a single dataframe.

# various hacks to transform the data to make it compatible with the UKESM1.1 model. (at 13.8)
# hack 1 -- remove parameters that get set through dependency or are internal sets.
df = df.drop(['dec_thres_cu','bparam','rho_snow_et_crit','allicetdegc'], errors='ignore')
if not df.index.is_unique:
    raise ValueError('QUMP HadGEM3 parameters are not unique. ')
#
# then rename some parameters to match the UKESM1.1 names
rename_dict={
    'ps_anth_so2_emiss':'anth_so2_ems_scaling',
    'ps_natl_dms_emiss': 'seadms_ems_scaling',
    'ps_natl_ss_emiss': 'sea_salt_ems_scaling',
    'ps_prim_moc': 'marine_pom_ems_scaling',
    'ps_dry_depvel_so2':'dry_depvel_so2_scaling',
    'ps_sigma_updraught':'sigma_updraught_scaling',
    'ps_acc_cor_scav': 'acc_cor_scav_scaling',
    'ps_bc_refrac_im':'bc_refrac_im_scaling',
    'ps_dry_depvel_acc':'dry_depvel_acc_scaling',


}

df = df.rename(index=rename_dict)  # rename the parameters to match the UKESM1.1 names.
df_exper = df_exper.rename(columns=rename_dict)
if not df.index.is_unique:
    my_logger.warning('QUMP HadGEM3 parameters are not unique after renaming. ')
    df = df.groupby(level=0).last()  # group by the index and take the last value for each parameter.
if not df_exper.index.is_unique:
    my_logger.warning('QUMP HadGEM3 experiments are not unique after renaming. ')
    df_exper = df_exper.groupby(level=0).last()  # group by the index and take the last value for each experiment.

# hack 3. Deal with functions that set multiple parameters. Could be worked out from the spreadsheet but for now will just specify them.
fns_special = {'aparam':['aparam','liu_latent'],#
               'rho_snow_fresh':['rho_snow_fresh','rho_snow_et_crit_delta'],
               'starticetkelvin':['starticetkelvin', 'allicetdegc0to1']} # these are special functions that set multiple parameters and use latent variables.
# This dict is keyed of the param name and the value is a list of parameters that are returned by the function.
# we drop these parameters when doing the simple parameters and run the functions to get the UKESM1.1 values.
# also modify teh function name in df to match the UKESM1.1 function name.
for fn,params in fns_special.items():
    try:
        # rename the function to match the UKESM1.1 name.
        df.loc[fn,'transformFunc'] = fn  # set the transform function to the function name.
        for param in params:
            df.loc[param,'transformFunc'] = fn  # set the transform function to the function name.
    except KeyError:
        my_logger.warning(f'Function {fn} not found in dataframe. Skipping renaming.')
        continue

# remove
## get in the reference. Used to see if variables set.
ref_dir = genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-dr157')  # ROSE config # UM 13.8 configuration
ref_model_vars = ref_dir/'app/um/rose-app.conf' # ROSE config
tmpdir = tempfile.TemporaryDirectory()  # create a temporary directory to store the reference config
mdir = pathlib.Path(tmpdir.name)  # make a pathlib Path object for the temporary directory
model = UKESM1_1_c8(name='test',reference=ref_dir,model_dir=mdir,suite_dir=mdir/'suite')  # create a UKESM1.1 model object to get the reference configuration
model.instantiate() # really need a way of reading in a ref config without instantiating the model.
with ref_model_vars.open('rt') as fp:
    ref_config = metomi.rose.config.load(fp)
# extract the parameter names from the spreadsheet
# find fns.
# find values we have by looking in the config
ukesm_values = dict()
missing_params = []
need_fns=dict()  # keep track of parameters that need functions
fn_params = [] # list of params that are functions. They should be dropped from simple variables.
list_fns=dict()
bad_fns=dict() # list of function that are missing some information.

# deal with special functions that set multiple parameters.
for fn,params in fns_special.items():
    default = model.read_param(fn)  # read the default value from the model
    ukesm_values.update(dict(zip(params,default)))  # set the default value for the parameters.
    fn_params += params  # add the parameters to the list of function parameters.


# 1) Handle dependencies
for depend in df.dependency[df.dependency != 'nan'].unique():
    if ',' in depend:  # for now can't handle such dependencies.
        # which params?
        params = df[df.dependency == depend].index.tolist()  # get the parameters that have this dependency.
        my_logger.warning(f'Parameters {params} have dependency {depend} with a comma in it. Skipping.')
        # print out try.
        depend_params_str = ["'"+d+"'" for d in depend.split(',')]  # split the dependency into its components.
        s = 'to est values try lin_reg(df_exper,xparams=['+ ','.join(depend_params_str)+f'],yparam="{params[0]}")'
        missing_params += [d.strip() for d in depend.split(',')]  # add the parameters to the missing parameters list.
        missing_params += params  # add the parameters to the missing parameters list.
        my_logger.warning(s)
        continue
    param = rename_dict.get(depend, depend)  # rename the parameter if it is in the rename_dict, otherwise use the original name.
    series = df.loc[param]  # get the series for the parameter
    fn = series.transformFunc   # get the function name
    if fn.lower() == 'nan':  # if the function is nan, we don't need a function.
        fn=None
    if param in fns_special:  # if the parameter is in the special fns dict, we need to handle it differently.
        fn = param  # get the function name from the existing entry.
    need_fns[depend] = dict(
        functions=[fn],
        namelists=[series.namelist],
        nl_vars=[depend]
    )
    fn_params+=[depend]  # add the parameter to the list of function parameters.
if not df.index.is_unique:
    raise ValueError('QUMP HadGEM3 parameters are not unique. ')

# 2) Handle dependency
L = (( df.dependency != 'nan') & ~(df.dependency.str.contains(',')))  # find rows where dependency  is not  nan and there are not multiple dependencies.
for param, series in df[L].iterrows():
    fn = series.transformFunc  # get the function name
    if fn.lower() == 'nan':  # if the function is nan, we don't need a function.
        fn = None
    need_fns[series.dependency]['functions'] += [fn]  # add the function to the list_fns dictionary
    need_fns[series.dependency]['namelists'] += [series.namelist]  # add the namelist to the list_fns dictionary
    need_fns[series.dependency]['nl_vars'] += [param]  # add the nl_var to the list_fns dictionary
    fn_params+=[param]  # add the parameter to the list of function parameters.
# 3) Check if namelist var exists in reference config
for param, info in need_fns.items():
    for idx,(fn,namelist,nl_var) in enumerate(zip(info['functions'],info['namelists'], info['nl_vars'])):
        if df.loc[param,'printInNamelist'].lower() == 'false':  # latent variable, so skip it.
            my_logger.info(f'Parameter {param} is a latent variable. Skipping.')
            continue
        nl_path = ['namelist:' + namelist, nl_var]
        value = ref_config.get(nl_path)

        if value is None or value.state == '!!':
            my_logger.warning(f'Missing nl_var {nl_var} in namelist namelist:{namelist} for param {param}')
            bad_fns[param] = info
        else:
            v = UMroseNamelistConfig.parse_value(value.value)
            if isinstance(v, list):  # if the value is a list, we need to handle it differently.

                if fn is None:  # if the function is None, we need to generate a function to handle the list.
                    my_logger.info(f'Parameter {nl_var} needs a function to handle the list value {v}.')
                    info['functions'][idx] = f'list_{nl_var}'
                else:
                    my_logger.info(f'Parameter {nl_path} has a list value {v} in namelist {namelist}.')
            ukesm_values[param] = v

# 4) remove the bad_fns from the need_fns dictionary.
for param in bad_fns.keys():
    need_fns.pop(param)  # remove the parameter from the need_fns dictionary if it is missing.

#5) get the UKESM1.1 values.Generating function skeletons for those that need functions.
for param, info in need_fns.items():
    try:
        value = model.read_param(param)  # read the parameter from the model
        ukesm_values[param] = value  # store the value
    except KeyError:
        my_logger.warning(f'Parameter {param} not found in {type(model)}. Implement.')
        print('# function for parameter:',param)
        print(generate_method(param,**info)) # print out skeleton function to implement.
#6 ) Remove the fn parameters from the dataframe.
df_present = df.drop(fn_params + missing_params, errors='ignore')  # remove the parameters that are functions and that are missing from the dataframe.


for nl_var,series in df_present.iterrows():
    if series.namelist.lower == 'nan' or series.printInNamelist.lower() == 'false':  # if the namelist is nan or false, we don't need to do anything.
        my_logger.warning(f'No namelist specified for {nl_var}')
        continue
    nl = ['namelist:'+series.namelist,nl_var]
    value = ref_config.get(nl)

    if value is not None and value.state != '!!':  # if the value is not None and not commented out, we can use it.
        value = UMroseNamelistConfig.parse_value(value.value)
        if isinstance(value, list):  # need a fn to make a list

            if series.transformFunc.lower() != 'nan':  # if the transform function is not nan, we can use it.
                raise ValueError('Have a list value but transformFunc is not nan. This should not happen.')
            df.loc[nl_var,'transformFunc'] = f'{nl_var}'  # set the transform function to the list function.
            # see if we have it already.
            try:
                ukesm_values[nl_var] = model.read_param(nl_var)
            except KeyError: # if the parameter is not found, we need to generate a function to handle it.
                my_logger.warning(f'Parameter {nl_var} needs a function to handle the list value {value}.')
                bad_fns[nl_var] = dict(
                    functions=[f'{nl_var}'],  # use the function name from the spreadsheet .
                    namelists=[series.namelist],  # use the namelist from the spreadsheet
                    nl_vars=[nl_var])
        else:
            ukesm_values[nl_var] = value # simple parameter so just store the value.
    else: # value not present in the reference config.`
        my_logger.warning(f'No value found for {nl_var} using {nl} in reference config')
        missing_params.append(nl_var)

# drop missing params from ukesm_values
ukesm_values = {k: v for k, v in ukesm_values.items() if k not in missing_params}
L = df.index.isin(ukesm_values.keys())
df_present = df[L]  # keep only the parameters that are in the ukesm_values dictionary.
if not df.index.is_unique:
    raise ValueError('QUMP HadGEM3 parameters are not unique. ')
## If min > default or max < default, then set min and max values to default
for param,value  in ukesm_values.items():  # iterate over the ukesm_values dictionary
    if value < df_present.loc[param,'min']:
        my_logger.warning(f'Parameter {param} value {value} is less than minimum {df_present.loc[param,"min"]}. Setting minimum to value.')
        df_present.loc[param,'min'] = value
    if value > df_present.loc[param,'max']:
        my_logger.warning(f'Parameter {param} value {value} is greater than maximum {df_present.loc[param,"max"]}. Setting maximum to value.')
        df_present.loc[param,'max'] = value


# sort by namelist and then variable name
df_present = df_present.sort_values(by=['namelist',df_present.index.name])


print('Missing parameters: ')
for param in missing_params:
    print(param,df.loc[param,'longName'])
print('+++++++\nMissing Functions: ')
for param in bad_fns:
    print(param,df.loc[param,'longName'])


print('\n================================================================')
# round all UKESM defaults to 5 sig fig.
ukesm_values = {param: round_sig(value, 5) for param, value in ukesm_values.items()}  # round the values to 4 significant figures.
fn_params = [p for p in fn_params if p in df_present.index]  # keep only the function parameters that are in the dataframe.
params_simple = list(set(df_present.index.tolist())-set(fn_params))  # list of simple parameters
# and remove the ones that do not have a namelist.
## Make up config file. Best way is to generate a pandas dataframe
df_params = pd.DataFrame(dict(parameter=params_simple,
                              type='um_rose',
                              filepath='app/um/rose-app.conf',
                              namelist=['namelist:'+nl for nl in df_present.loc[params_simple].namelist],
                              nl_var=params_simple,
                              default=[ukesm_values[p] for p in params_simple],
                              name=None,
                              function_name=None)).set_index('parameter')
# and need to sort this!
df_params = df_params.sort_values(by=['namelist','parameter'])  # sort by namelist and then parameter name.

# and make the parameter config file.
df_params.to_csv(simple_param_csv)
params = df_present.index.tolist()  # list of all parameters
param_dict=dict(
    _comment = f'File auto-generated from HadGEM3 parameters spreadsheet. Version 0.9 of {datetime.date.today().isoformat()}',
    number_params_comment=f'This file has {len(params)} parameters.',
    sourceurl_comment = f'Source URLs for HadGEM3 parameters: {", ".join(source_urls)}',
    defaultParams=dict(
        _comment = "Default parameters for UKESM1.1. These are the parameters that are used if not specified in the configuration file.",
    ),
    minmax=dict(
        comment= "Defines the minimum and maximum ranges for UKESM parameters. Must be defined for ALL parameters",
    )
)


for param,series in df_present.iterrows():
    qump_value = series.loc['standardValue']
    ukesm_value = ukesm_values[param]
    if qump_value != ukesm_value:
        my_logger.warning(f' QUMP value for {param} is {qump_value} but UKESM1.1 value is {ukesm_value}. Using UKESM1.1 value.')
        param_dict['defaultParams'][param+'_qump_value_comment'] = qump_value
    param_dict['defaultParams'][param] = ukesm_value
    if df.loc[param,'longName'] != 'nan':  # if the long name is not NaN, use it as the comment
        param_dict['defaultParams'][param+'_comment'] = series.loc['longName']
    if df.loc[param,'transformFunc'] != 'nan':
        param_dict['defaultParams'][param+'_function_comment'] = df.loc[param,'transformFunc']  # add the function name as a comment
    param_dict['defaultParams'][param+'_namelist_comment'] = series.loc['namelist']  # provide the namelist as a comment
    param_dict['minmax'][param] = series.loc[['min','max']].to_list() # set min and max


with output_param.open( 'wt') as f:
    json.dump(param_dict,f,indent=4,sort_keys=False)  # write the dictionary to a JSON file with indentation and sorted

print(f'Have {len(params_simple)} simple parameters and {len(fn_params)} function parameters.')





# Planned changes to StudyConfig.transMatrix

## Problem
transMatrix takes two arguments that impact its behaviour but are not in the configuration file:
- minEvalue:float=1e-6,
- warn_scale:float = 1e-1

## Proposed solution
- Add these two parameters to the configuration in the covariance section via self.getv('study', {}).get('covariance', {})
- change minEvalue and warn_scale to both be None by default.
- if value is None then get value from config with default values as now. 
- Do by adding a new function get_covariance_param which takes the parameter name and default value as arguments and returns the value from config or default if not provided.

## Proposed tests for get_covariance_param
- Name not in covariance section of config: should return default value.
- Name in covariance section of config: should return value from config.
- Name in covariance section of config but value is None: should return default value.

## Proposed tests for transMatrix
- change minEvalue and transMatrix should change. 

## Potential future changes
Generalise get_covariance_param to be get_config_param which takes a path (. separated?) and default value as arguments. 
Then, returns the value from config or default if not provided or value is None. 
This would allow use of this function for other parameters in the future if needed.


# Plan to check for non-deterministic algorithm. 

## Problem 

- Will detect non-deterministic algorithm because it will not reuse existing model cases. 
- Currently, do that by checking logical_info (which gets reset every iteration) against model_index to verify no missing models in logical_info
- However, this approach will fail if load models from files or initialise run_submit with models.

## Requirements

Check for non-determinism only for models asked for in earlier iterations.
Means that models loaded in or used in initialization do not trigger an error 

## Solution

### Changes to runSubmit

    - Add  to _init_ a model_status -- this is a *per* model tracker.
    - Modify _init_ to set model_status[key] for all models passed in to be 'initial' 
    - Modify from_dict to set all keys in model_status (if it does not exist)  to 'unknown' [Can't think of other behaviour to handle legacy]
    - Add own version of create_model (overriding the SubmitStudy version). This sets status to 'called' for  models
    - Add own version of read_model_configs  (overriding Study version). This sets status to 'read' for all models read in
    - modify make_model to set model_status  to 'called' for any not None model created/reused. 
    - modify reset_logical_info() to set all 'called' or 'unknown'  models to 'not_called'
    - modify check_deterministic to check that nothing in model_status is 'not_called'. If any trigger error and report on unused models.
    - modify check_deterministic to verify that all models in model_index are in model_status and vice versa. set difference is empty.


                                                         

### Tests

#### LogicalInfo
  -   __init__  could check that model_status exists? 
  - check that from_dict for legacy cases sets status to 'called' for all models
#### runSubmit 
- create model. model_status size should increase and last value be 'called'
- __init__.  
  - Passing in models model_status should be 'initial' for all models. 
  - Check that model_status exists if Models is None
- from_dict. For legacy cases model_status should be entirely 'unknown'
- read_model_config. All models read in should have status 'read'
- reset_logical_info. all model_status should be 'not_called' or 'initial' or 'read'
- check_non_deterministic
  - Run a cycle twice with a non-deterministic algorithm and faking. Should get an error.
  - Run a cycle twice with a deterministic algorithm but some inital/readin data that won't be used. 
    - Should not get an error. 
    - Some of the 'initial' & 'read' cases should be 'called'
  - Add a fake key to model_status. Should trigger an error. 

    


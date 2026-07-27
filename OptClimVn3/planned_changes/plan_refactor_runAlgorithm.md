# Plan to  refactor runAlgorithm 

## Goals
1) Split runAlgorithm functionality into two scripts:
- A setup and a run script. Setup sets up the SubmitStudy config and run runs the SubmitStudy config.
- Some functionality in runAlgorithm is now  in OptClim_control.py which will not be included in setup/run scripts.
- Functionality around failure handling will go to OptClim_control.py.
- In doing this make use of changes to paths in Study etc.
2) Modify runAlgorithm. If called, will fail and say to call setup/run scripts.


## Why split runAlgorithm.py 
runAlgorithm is a large script that both sets up a study config AND runs it.
Splitting it up will make it more modular and have clear separation between initial setup and running the study. 
Code is very long and verbose. So a refactor will make it more readable and easier to maintain.



### Detailed setup_config  specification.

setup_config.py path_to_study_config.  Does setup of a SubmitStudy config.
- [path_to_SubmitStudy.sfcg] Path to SubmitStudy config which will get created. A sensible default will be used if not given.
- [path_to_oldSubmitStudy] Path to old SubmitStudy config. If provided, model **configs** will be copied from the old SubmitStudy to the new study_dir.
- update_params update the params in the model cases being copied.
- update_obs update the obs in the model cases being copied.
- Both options should fail if path_to_oldSubmitStudy is not given.
 
- delete & purge Those could probably be combined into one option which deletes script_dir and everything in it. 
 Will continue to force user to confirm they want to do this.
- confirm_delete/no_confirm_deltete. If no_confirm_delete is given no confirmation will be asked for. 
- log_level [One of debug, info, warning, error, critical]  Default is warning. This will set the logging level for the script. 
  config can override this for OptClim library logging but should not override the script logging. 

Should fail if the path_to_SubmitStudy.sfcg **or** its parent exists 

### Detailed run_config.py specification.

#### Purpose
Executes a single iteration of an  **existing** SubmitStudy config. This will run the algorithm, 
instantiate any new models needed  and submit them, and itself,  to the queue. 
Optionally, it archives and produces a monitoring plot and decides what to do with failed models.

#### Inputs
A path to a SubmitStudy config. This will be read in and used.

#### Outputs 
Modified SubmitStudy config and any new model configurations created.
#### Error Conditions
- Fail if the path_to_SubmitStudy.sfcg does not exist. (or anything else requested).
- Fail if any models are not in an expected state.
- Fail if algorithm_name is not recognized. 

#### Command Line Options

run_config.py path_to_SubmitStudy.sfcg  

- path_to_SubmitStudy.sfcg. Path to SubmitStudy config.

- log_level [One of debug, info, warning, error, critical]  Default is warning. This will set the logging level for the script and everything else. 
 Note that config in SubmitStudy.sfcg overrides this for the OptClim library. 

- mode [One of run,fake-run, dry-run,readonly]  Default is run. 
  * run run the study.
  * fake-run fake the runs (and don't submit anything). Useful for testing the config.
  * dry-run instantiate the runs but do not submit anything.
  * readonly read the config (and run algorithm) but do not instantiate or submit anything. This is useful for testing the config and algorithm without actually running anything.

- timeout -- timeout in seconds for grabbing lock.

- monitor/archive. For both options if no relative path is given a default path will be used.
- monitor -- produce monitoring plot at the end of iteration.
- archive -- archive the study at the end of the iteration.  
- 

## Addtions to OptClim_control.py
Add the following options to OptClim_control.py. These come from runAlgorithm.py. 
- guess_fail -- guess  cases that are running that have failed. [Note this is tricky and probably needs updating from current behaviour].
- fail continue|perturb|perturbc|delete  -- what to do with failed models. (models in status FAILED)
    * continue Continue Set status of failed model to CONTINUE. Forcing it to start from last saved model state.
    * perturb Perturb Pertub failed model and set status to INSTANTIATED. Forcing it to start again.
    * perturbc Perturb failed model & set status to CONTINUE. Forcing it to start from last saved model state. 
    * delete Delete failed model. 



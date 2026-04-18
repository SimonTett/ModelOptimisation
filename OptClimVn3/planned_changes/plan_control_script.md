# Plan to add script to control OptClim. 

OptClim needs some way of controlling it while running. Options needed are:

- **kill** Kill current optimisation, all running models and post-processing tasks. 
- **stop** Mark optimisation to be stopped next time it runs.  This will run any instantiated cases.
- **continue** Clear stopping. 
- **plot** Plot current logical evolution. (calls config.plot()) 
- **update** Update parameters from current config file. 
- **clear** Clear any existing locks
- **archive** Archive current config and results.
- 

The script doing all this could also do other things -- like run the config etc but that is for later.





## Plan
1) Add file locking on runSubmit configuration file. 
  Requires new method runSubmit.lock_config which will create a lock file for the config file. This will be used in runAlgorithm and the control script.
  Will put detailed locking in genericLib to allow for easy use in other places and to use different locking modules.

2) Add to SubmitStudy a new attribute: next_command which is either None or 
  **stop** No generation of new models but will run all  existing instantiated models.

3) Add control script with options to call the various functions. Will use file locking with no timeout. 


## Locking Plan

- Use py-lockfile to create a lock file for runAlgorithm. From either read in or initial creation to final write.
- Locks will not block failing immediately if attempt to get lock does not succeed for control case. 
- For runAlgorithm have a 30 sec timeout. [For now make this hardwired]
- Only use locking if planning to modify config -- so not for plotting. But other cases need it. 

## Stopping plan

- No more cases are created 
  - modify SubmitStudy.create_model to return None if stopping. 
  - modify runSubmit.create_model to raise appropriate error if model created is None. 
- Any cases that are instantiated are submitted (as now)
- runAlgorithm terminates if no new cases are submitted (as now).  



## OptClim script. 
Has the following syntax:

OptClim_control.py [config_file] [--log_level debug|info|warning ] command [option_args_per_command]

Implements the following options:
1) stop - set next_command to stop. 
2) continue - set next_command to None. This will allow optimisation to run on. 
3) plot -- call  plot method. Need to implement plot for runSubmit. 
   *Optional* argument is [--output_file filepath] for plot.  Default is f"monitor_{self.name}.png" in current directory.
4) kill -- call StudyConfig kill - kills everything.
5) update -- call StudyConfig update_params - updates parameters from config file and updates evaluation database info.
   *Optional* argument is [--config_file filepath]. Default is current config file in self.config.fileName()
6) ~~ archive -- call StudyConfig archive - archives current config and results.~~ [Not now as need to decide what to archive beyond minimal list]

All but plot will require file locking. locking should fail immediately if lock cannot be obtained.
Command line arguments will be the config file and the option. 


## runAlgorithm plan

Remove the following options as these go to the control script:
- update_config
- kill
- stop
- monitor
~~- archive~~

## Possible cylc plan.
Use cylc ext-triggers (https://cylc.github.io/cylc-doc/stable/html/user-guide/writing-workflows/external-triggers.html#ext-triggers-push) 
 to build dependencies between "runAlgorithm" and model/post-processing tasks. 
This then gives a cylc workflow which runs the algorithm, models & post-processing. 
Problem don't know how many models will be run and if no models to run then we are done. 
Two possible approaches:
1) "runAlgorithm" is a task which runs the algorithm, generates the new workflow and, if any models to run, triggers the workflow. 
   If no models to run then it is done.
2) Have a workflow which is dynamic and generates new tasks as needed. This is more complex but would allow for better monitoring of the workflow.

Case 1 does not obviously have any advantages over using slurm though might be a bit simpler as don't have models held and dependencies coded in.
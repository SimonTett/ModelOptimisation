# Problem
run_info growing topsy-turvy like and has become a dumping ground for anything to do with running the model.
Consequently, it mixes up system/supercomputer stuff (initial aim) and model-specific stuff. 
Also have complications with running post-processing on same super-computer as model run  when data is moved onto remote machine

Will move to four  blocks:
1) Information for update_status script. This is to control resources for update_status script (if needed)
2) Information for running the post-processing. Need to specify machine etc. 
3) Information for running the model. Broadly this would inherit stuff from current run_info
4) Information for (remote) archiving. Basically where the remote data goes. 

1 requires small changes to model_create. 2) is tricky and likely involves extensive changes to current system.
3 & 4 are updates to existing systems. 

Need a machine block?? Need to specify machine, userid, root path, setup script(?) on that machine. 
Structure something like: [cylc will ignore engine and just add to cylc job]
remote running is not supported except with cylc.  If have multiple machine but not cylc then throw error.
Should all machines have host_name?  Broadly want similar behaviour to current unless explicitly ask for something else.
But already have a postProcess block. Should reuse that?

Am I overdoing this? Broadly machines only sensible for cylc and so only for um_rose. Good to identify cylc stuff.  I think my actual use case is post-processing within cylc which still needs a dummy job and hacks to transfer -- where data goes. 
So, could add a machine, root_path, run_time to postProcess and that is all I need. Potentially need run_code? 

Rule "Only do what is needed now." Thus, will add to postProcess extra information.  But actually I don't think solves very much. 
Better solution, for now, is to modify um_rose.py so that it overrides the std model stuff to 
1) Modify the workflow, copy the pp script & data into app. See https://metomi.github.io/rose/doc/html/tutorial/rose/applications.html 
{
"max_runs":null,
"max_model_simulations":null,
"model_name":"UM_ROSE",
"machines":{
"_comment":"Machine properties.If host_name provided then things will be copied across and remote submitted (if needed)"
 "jasmin":{"user_id":xxx,"setup_script":"$OPTCLIMTOP/setup_jasmin.sh","run_code":"no2-xxxx",
            submit_engine="SLURM",host_name='jasmin.ac.uk',environment={"OPTCLIMTOP":"xxxx"},
            environment_comment:"dict of Environment variables"},
"jasmin_comment":"Information for things running on jasmin",
"cirrus":{"user_id":xxx,"setup_script":"$OPTCLIMTOP/setup_cirrus.sh","run_code":"no2-xxxx",submit_engine='SLURM',
          "environment":{"OPTCLIMTOP":"xxxx"},"environment_comment","Dict of environment variables"}
"cirrus_comment","Information for things running on cirrus"
},
"post_process":{"root_path":"/gws/xx/etc",machine='jasmin',"run_time":"1h"},
"post_process_comment":"post processing options",
"install_workflow":{"machine":"cirrus",root_path="xxxx"},
"install_workflow_comment":"Options for workflow install",
"simulate":{"machine":"cirrus","run_time":"12h"},
"simulate_comment":"Options for simulation",
"set_status":{machine="cirrus","run_time":"30m",script_path='$OPTCLIMTOP/OptClimVN3/scripts/set_status.py'}
"set_status_comment":"Options for setting status",
"run_iteration":{machine="cirrus","run_time":"45m"},
"run_iteration_comment":"Options for running the next iteration of the algorithm"
}

run_info now  contains two sub-dictionaries: system_run_info and model_run_info:

- system_run_info contains information about the system, supercomputer, and environment in which the model is run. 
- model_run_info contains information specific to the model being run. 

run_info grew partly because of the need to do a remote installation of the model on a supercomputer.  
For UM the workflow needs to be on puma, model ran on a supercomputer, and data archived somewhere else.  
So need a way to specify information about where the installation occurs (generalized to user_IID@machine:path) & data archive place (same generalization).
Want to specify root directories for model install and data archive with the actual path depending on where OptClim is being run. So need a local root dir to define that. 
broadly local_root_dir -> install_dir, and archive_dir. Paths relative to local_root_dir then get mapped relative to install_dir and archive_dir.
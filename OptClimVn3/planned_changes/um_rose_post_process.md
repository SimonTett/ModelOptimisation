# UM-ROSE post-process.

Issue: UM transfers data to remote machine -- jasmin. Transfers happen as model runs so do not ever have full dataset on super-computer. Thus, real post-processing needs to run on remote machine. 

# Context 

Climate models run on a super-computer using rose/cylc to run in three-month cycles. After each cycle data is transferred to jasmin, using globus, and deleted from the super-computer. The optimization calculations need a set of simulated "observables" available on the super-computer. The simulated "observations" are an extremely low-dimensional summary of the data but could require processing of the entire model dataset.

# Problem
Current system submits slurm scripts, in held status, to do post-processing at the end of the simulation.  These are released when the model completes. The current work-around, to stop house keeping deleting needed data, is to copy data  to a known place on the super-computer. This means data management is a problem and the system is not robust to failure. There have been some recent instances of this problem and so fixing it is worth doing.

# Solution

Follow Estanislao's approach and have a reference workflow that does post-processing at the end of the workflow. This requires definition of how to set this up and, for now, will not be automatically done. Using principle of "Only do what is needed now." means minimalist changes needed to make this work. A **possible** next step would be to automate the process of generating the post-processing within cylc.

The reference configuration will therefore need:

- A post-processing script.
- Configuration json file for script. Placed in standard place.
- Name of output file. 
- Documentation on  how to set the script up.  
- Setup all as application following guidance at:  https://metomi.github.io/rose/doc/html/tutorial/rose/applications.html
- Will need a set of "namelist variables" which can be overwritten. Additional variables are:
  - *post_process_output_file*: name relative to *transfer_dir*

When the post-processing application is run then 


- Need a dummy post-process job that copies summary data back from jasmin to super-computer. Need to provide name of file here [$pp_transfer/workflow_name_time.nc] -- use netcdf to preserve precision and gets put in as a variable somewhere for subsequent use.  Continue to use existing modifications on transfer.  If post-process information contains script name then throw an error. 

Post-processing script will be remote_copy.sh which runs on archer2 on the serial Q and uses globus to do the transfer. The pp_transfer variable will be used to get the directory for the Look at transfer script to get framework. Hopefully, there are variables! See postproc\rose_app.conf and pp_transfer (which points to a directory). 

## Custom Aggregation Script (2025-11)

Add custom aggregation script to postproc app. Runs **on Jasmin** (where data lives) once at workflow end (R1 task), finds all cycle outputs, time-sorts and aggregates.

### Changes

1. **`app/postproc/rose-app.conf`** - Add to `[env]`: `TRANSFER_DIR` (Jasmin path), `MODEL_NAME=${CYLC_WORKFLOW_NAME}`, `OUTPUT_FILE=${TRANSFER_DIR}/${MODEL_NAME}/aggregated.nc`, `AGGREGATE_CONFIG=$ROSE_DATA/postproc/aggregate_config.json`. Add `[command] custom_aggregate=bin/aggregate.sh`.

2. **`app/postproc/bin/aggregate.sh`** - Bash wrapper: sources **Jasmin's** Python env (user modifies: `. /path/to/jasmin/venv/bin/activate` or `module load` or `uv run`), calls Python script with `--config $AGGREGATE_CONFIG --output $OUTPUT_FILE`.

3. **`app/postproc/file/postproc/aggregate_config.json`** - User-defined processing options (variables, time ranges, etc.).

4. **`flow.cylc`** - Add R1 task: `R1//^+{{EXPT_RUNLEN}}-{{EXPT_RESUB}} = postproc => postproc_custom` with `script = rose task-run --command=custom_aggregate` and **`platform = jasmin`**.

5. **User provides** `$OPTCLIMTOP/scripts/aggregate.py` - Uses `rglob($TRANSFER_DIR/$MODEL_NAME/**/*.nc)`, aggregates with xarray/iris, writes to `$OUTPUT_FILE` (on Jasmin).

### Key Design
- Single source of truth: `[env]` section, namelists reference via `${VAR}`
- Script runs on Jasmin where data lives (no cross-mount issues)
- Script finds its own input files (no cycle paths needed)
- User controls Python env and processing logic

### Issues to Resolve

- [ ] **Cylc platform config**: Does `global.cylc` on Puma2 have Jasmin platform defined? What SSH/auth setup is needed?
- [ ] **Jasmin Python environment**: What modules/venv available on Jasmin? Document setup in `aggregate.sh`.
- [ ] **Authentication**: How does Cylc submit jobs to Jasmin? (SSH keys, Globus Compute, etc.)
- [ ] **Estanislao's workflow**: Review his Jasmin post-processing setup for patterns/lessons.
- [ ] **Testing**: Need test workflow to validate Jasmin task execution before full integration.



# Change log

- 2025-12-28: Added update_params.py script to copy and update parameters. 
- 2025-12-28:  runSubmit, SubmitStudy & Model now have update_params methods which update parameters from on file config.
                    For Model this will likely need work for derived model types.
- 2025-12-25: Model, SubmitStudy &  runSubmit have copyConfig methhods to allow copying of a configuration 
- 2025-12-28: archive now makes use of copy method which makes it easier to generalise. 

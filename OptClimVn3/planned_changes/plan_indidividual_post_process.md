# Plan for individual post-processing in OptClimVn3

This document outlines plan for individual post-processing tasks. 

Post processing will now, like model generation, allow post-processing to differ between models. 
Will use the functionality provided by multi-model support in OptClimVn3 to give a reference_name to each model.
This reference_name will be used the post-processing step to determine which post-processing to apply.

Plan
1) Set reference_name. Needs to be another (optional) argument to __init__ in Model class (and anything inheriting from it).
  Tricky part -- it has to come from the user defined multi-model configuration though the name could also be in run_info too.
  reference_name comes from the name of the reference directory if not otherwise specified.   

Test Cases: 
  Create a multi-model configuration with two models, each with different reference_name. 
    Check that the reference_name is set correctly in each model after creation.  
  Check reference_name is set to self.reference.name if not otherwise specified.


2) Allow a standard post-processing which gets overridden by reference_name specific post-processing. 
   Add to config post_process_for_reference which is a dict of reference_name: dict of post-processing options which override standard post-processing. 
   Can also be null/None and if so is ignored.  
   And if post_process_for_reference exists then it will be searched (failing if the name does not exist) and used to overwrite the standard post-processing options.
   This is done in Model.set_post_process

Test Cases:
    Create a multi-model configuration with two models, each with different reference_name and different post_process_for_reference.
      Check that the post-processing options are set correctly in each model after creation.
    Check that if an invalid reference_name is provided in post_process_for_reference then an error is raised.

3) Modify example multi-model code to name the models, and modify runSubmit.make_model to have an optional argument reference_name
     to pass to SubmitStudy.create_model which actually does the work.

Test Cases:
    Modify existing multi-model test cases to include reference_name in model creation.
    Check that models are created with the correct reference_name when not provided (i.e., defaults to reference.name).

4) Modify from_dict to work out reference_name if it is not provided (just use reference.name). Allows compatability with existing code.

Test Cases:
    Create a model from a dict without reference_name and check that it defaults to reference.name.
    Create a model from a dict with reference_name and check that it is set correctly.



Added config_name to Model methods which returns a combination of reference_name & ensembleMember. 
The means that logical_info.models changes from dict of dicts
  where inner dict keys are config names and values are models to dict of lists with list being a list of models.

This generates some changes in runSubmit.LogicalInfo including the to/from_dict methods.
Also noticed code was buggy as name order not same as model generation. So this approach is a bit more robust.
This directory/folder provides modules & a package to support different Models.

# Modules
Model.py -- root class for all models. 
Provides generic functions that should be overloaded. 
If you do want to add your own method then within that method it is recommended to first call the super-class method.

model_base.py -- provides two classes 
   * journal -- provides support for recording history and output of commands
   * model_base -- registers model with generic_json and provides minimum methods needed by that.


namelist_var.py -- provides class that supports namelists. You might want to use this as an example or extend this if your model sets parameters in other ways.

param_info.py -- provides class that supports parameter sets including a register function.

simple_model.py -- a simple model that does little but is intended as an example and for testing. 

HadCM3 -- a class for HadCM3.

# Adding a new model 

To add a new model you need do the following:

* Define the parameters that directly impact your namelists (or whatever else you use) in a .csv file.
   By default, this should contain the following columns:
   * parameter -- the name of the parameter
   * type --  the tye (namelist_var or function)
   * filepath -- the path, relative to model_dir, for the namelist file
   * namelist -- the name of the namelist 
   * nl_var -- the name of the  namelist variable
   * default -- if provided the default value to use when reading if namelist or nl_var not provided
   * name -- If provided the name of the namelist_var
   * function_name -- generally not provided but if provided the name of the function.
  See parameter_config/HadCM3_Parameters.csv for an example. 

* Write a class that inherits from Model. See HadCM3.py for a comprehensive example.
   
    * The class should define all the functions you need for the class. All functions need to be registered using the @register_param decorator defined in Model. These functions should either: 
      * generate a list of namelist_var's each of which set a single variable (which can be a list) in the models namelists.
         (If your model makes changes in a different way than namelists you will have a bit more work to do. Suggest defining something similar to namelist_var
         which has similar behaviour.
      * Or make some changes directly and return None.
  
      * Modify __init__ though make sure you call the super class. 
       You will want to do this to setup submit_script and continue_script. You may also want to define specific behaviour or attributes for your model.

      * Define modify_model method -- though do call the super cass method first. This should make any changes necessary to your model script and continue script. 
      In particular, you will want to run:
        * path_to_scripts/set_model_status self.config_path RUNNING when your model starts.
        * path_to_scripts/set_model_status self.config_path FAILED when it fails
        * path_to_scripts/set_model_status self.config_path SUCEEDED when it has successfully ran. 

      * Define submit_cmd method -- this returns a command to run your model 
  
      * Load up the default single parameter values ideally from parameter_config/MODELNAME_Parameters.csv

      * Generate pertrub method which calls Model.perturb with the parameters/values to perturb.
      
* Write some tests ideally in test_MODELNAME.py in the test_models directory. See test_HadCM3.py for a comprehensive example. 
   
* Add the name of your class to the __any__ variable in the __init__.py file. That way reading and writing configurations will work.
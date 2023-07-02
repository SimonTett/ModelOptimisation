# Modues

model_base.py -- provides two classes 
   * journal -- provides support for recording history and output of commands
   * model_base -- registers model with generic_json and provides minimum methods needed by that.

ModelBaseClass -- provides registration function and class for registering functions, setting parameters and registering models. 

namelist_var.py -- provides class that supports namelists. You might want to use this as an example or extend this if your model sets parameters in other ways.

param_info.py -- provides class that supports parameter sets including a register function.

generic_json.py -- support for reading and writing objects to disc as json files.

genericLib.py -- provide general functions.

Tests are all in support_tests directory.
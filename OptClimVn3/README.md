# Modules that support runnings studies.
Modules are:


* Study.py -- read only support for Studies (collections of model simulations)

* SubmitStudy.py -- generation of new model simulations, caching of aready run simulations and submission of simulations

* runSubmit.py -- algorithms for model submission etc

* StudyConfig.py -- reads in and decodes study configuration files. 

* optclim_exceptions.py -- provides exceptions needed by SubmitStudy -- an exception to be raised if model does not exist.

Also see support for modules that provide general support and Models for model classes.
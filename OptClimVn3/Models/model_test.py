""""
Test model class. This designed to test whole algorithm stage without actually running a model.
Possibly needs a fake function. Idea is to overload some methods.
If you want to see output of logging in test output see:
https://docs.pytest.org/en/latest/how-to/logging.html
Default is to show warning and above messages
"""
import logging
import typing

from Model import Model


class model_test(Model):
    """
    Model class for testing system stuff
    """


    status_info = dict(CREATED=None,
                       INSTANTIATED=["CREATED"],  # Instantiate a model requires it to have been created
                       PROCESSED=['INSTANTIATED'])  # Fake model -- go straight from INSTANTIATED to PROCESSED
    allowed_status = set(status_info.keys())

    # How to get fake fn passed in? Needs to be available at submission time.
    # and be available at the class level.
    # fn needs to only act on "changeable" parameters. Which is not something that sits in the model's domain...
    # post processing block could be modified to include a list of parameters to consider.
    # which would come from the algorithm level. Provide a default fake fn and so fake script??
    """
    Alternatively -- a post processing script should have  a "fake" mode. That's a better design. post-processing knows
     what it should be dealing with. It just get a post process block with fake in it which tells it which parameters 
       are variable.  When it does it then generates the fake obs however it likes. 
       Post process block should be filled in with lots of useful info from studyConfig, 
    if fake:
      model='model_test'
      variable_params = StudyConfig.variable_params() # extract from start params. Also have fixed_params from config.
      post_process_block = StudyConfig.postProcess()
      # modify post-process block
      post_process_block['fake']=True. Thisnshould be generalised back to Model. All "models" need a fake thing
      and that only depends on the post-process. So basically when submit called just fake it! 

    """

    def submit(self, submit_cmd: typing.Callable, run_post_process_cmd: str) -> str:
        """
        override default
        :param submit_cmd: submit cmd
        :param run_post_process_cmd:  cmd to run post processing cmd
        :return: something!
        """
        # this is a fake fn and does nothing except call process.
        logging.info("No submission being done. Just calling process to fake obs")
        self.process()


"""
Test code for config.


"""

import os
import tempfile
import unittest

import numpy as np

import HadCM3
import StudyConfig
import config
import optClimLib


class testModelSimulation(unittest.TestCase):
    """
    Test cases for config. There should be one for every function in config.py
    Sadly code is horrible and complex. And generally needs a Submit object to act on.

    """

    def setUp(self):

        """

        :arg self

        Setup -- generate a model  object.

        """

        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = self.tmpDir.name
        refDir = 'test_in'
        jsonFile = os.path.join('Configurations', 'example.json')
        configData = StudyConfig.readConfig(filename=jsonFile, ordered=True)  # parse the jsonFile.
        self.config = configData
        params = configData.beginParam().to_dict()
        model = HadCM3.HadCM3(testDir,obsNames=configData.obsNames(),create=True,
                             refDirPath= configData.referenceConfig(),ppOutputFile='obs.json',name='test1',parameters=params)
        self.model = model
        # generate case with modelEnsemble = 2
        params.update(ensembleMember=2)
        model2= HadCM3.HadCM3(testDir,obsNames=configData.obsNames(),create=True,
                             refDirPath= configData.referenceConfig(),ppOutputFile='obs.json',name='test2',parameters=params)

        self.model2 = model2

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        optClimLib.delDirContents(self.tmpDir.name)

    def test_easyFake(self):
        """

        test easyFake

        cases are:
        1) Standard case
        2) Case with ensembleMember


        """
        obs1 = config.easyFake(self.model, self.config)
        obs2 = config.easyFake(self.model2, self.config)
        self.assertTrue(np.abs(obs1.loc['netflux_global']-obs2.loc['netflux_global']) < 0.5)






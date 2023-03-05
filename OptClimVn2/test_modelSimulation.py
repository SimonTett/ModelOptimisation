"""
test_modelSimulation: test cases for modelSimulation methods.
This test routine needs OPTCLIMTOP specified sensibly. 
"""

import collections
import json
import math
import os
import shutil
import tempfile
import unittest
import logging
import f90nml
import numpy as np
import pandas as pd
from OptClimVn2 import optClimLib, ModelSimulation
from HadCM3 import iceAlbedo


__author__ = 'stett2'


class testModelSimulation(unittest.TestCase):
    """
    Test cases for modelSimulation. There should be one for every method in ModelSimulation.

    """

    def setUp(self):
        """
        Standard setup for all test cases
        :return:
        """
        logging.basicConfig(level=logging.INFO)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.verbose = False  # verbose if True
        parameters =dict()
        parameters['one'] = 1.0
        parameters['two'] = 2.0
        self.parameters = parameters
        self.testDir = tempfile.TemporaryDirectory().name
        refDir = 'test_in'
        self.dirPath = self.testDir

        self.obsNames = ['temp@500_nhx', 'temp@500_tropics', 'temp@500_shx']

        refDirPath = os.path.expandvars(os.path.expanduser(refDir))
        refDirPath = os.path.join(refDirPath, 'start')
        self.exampleObsPath = os.path.join(refDir, '01_GN', 'h0101', 'observables.nc')
        self.refDirPath = refDirPath
        self.model = ModelSimulation.ModelSimulation(self.testDir,
                                                     name='test', create=True,
                                                     refDirPath=self.refDirPath,
                                                     ppExePath='postProcess.sh',
                                                     ppOutputFile='obs.nc',
                                                     parameters=self.parameters,
                                                     obsNames=self.obsNames,
                                                     verbose=self.verbose)
        shutil.copy(self.exampleObsPath, os.path.join(self.testDir, 'obs.nc'))
        # copy over a netcdf file of observations.

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        optClimLib.delDirContents(self.testDir)

    def test_Init(self):
        """
        Test init methods works.
        Probably over-kill and could be a pain if internal details change.
        But idea is that public methods all get a work out when modelSimulation initialised.
        :return:
        """
        # using implicit run of setup.
        expectObs={k:None for k in ['temp@500_nhx', 'temp@500_tropics', 'temp@500_shx']}

        expectParm = collections.OrderedDict()
        expectParm['one'] = 1.0
        expectParm['two'] = 2.0

        self.assertEqual(self.model.get(['name']), 'test')
        self.assertEqual(self.model.get(['ppExePath']), 'postProcess.sh')
        self.assertEqual(self.model.get(['observations']), None)
        self.assertEqual(self.model.get(['parameters']), expectParm)
        self.assertEqual(self.model.get(['ppOutputFile']), 'obs.nc')

        # test that read works. Works means no failures and have observations...

        m = ModelSimulation.ModelSimulation(self.dirPath, verbose=self.verbose)
        # observations should have changed but nothing else.
        self.assertEqual(m.get(['name']), 'test')
        self.assertEqual(m.get(['ppExePath']), 'postProcess.sh')
        self.assertEqual(m.get(['parameters']), expectParm)
        self.assertEqual(m.get(['ppOutputFile']), 'obs.nc')
        obs = m.readObs(obsNames=expectObs.keys())
        self.assertEqual(set(obs.keys()), set(expectObs.keys()))
        self.assertNotEqual(obs, expectObs)
        # updating parameters should trigger an error as we are read only
        with self.assertRaises(Exception):
            m.setParams({'aaa': 99.9}, addParam=True)

        ## now  to test that update works.
        m = ModelSimulation.ModelSimulation(self.dirPath, verbose=self.verbose, update=True)
        # observations should have changed but nothing else.
        self.assertEqual(m.get(['name']), 'test')
        self.assertEqual(m.get(['ppExePath']), 'postProcess.sh')
        # self.assertEqual(m.get(['observations']),expectObs)
        self.assertEqual(m.get(['parameters']), expectParm)
        # self.assertEqual(self.model.config['refDir'], None)
        self.assertEqual(m.get(['ppOutputFile']), 'obs.nc')
        o=m.readObs(obsNames=expectObs.keys())
        self.assertListEqual(list(o.keys()), list(expectObs.keys()))
        self.assertNotEqual(o, expectObs)
        m.setParams({'aaa': 99.9}, addParam=True)

    def test_get(self):
        """
        Test that get works
        :return:
        """
        self.model.set({'fbf': 1.0, 'fbg': 2.0}, write=False)  # set value
        self.assertEqual(self.model.get('fbf'), 1.0)
        self.assertEqual(self.model.get(['fbf', 'fbg']), [1.0, 2.0])



    def test_getParams(self):
        """
        Test that getParam works. 
        :return:
        """
        self.assertEqual(self.model.getParams(), self.model.get('parameters'))
        # test if we specify parameters it works
        self.assertEqual(self.model.getParams(params=['one', 'two']), self.model.get('parameters'))
        # test series works.
        self.assertTrue(self.model.getParams(params=['one', 'two'], series=True).equals(
            pd.Series(self.model.get('parameters'))))

    def test_readModelSimulation(self):
        """
        Test that readModelSimulation works.  Not really needed as init test checks this works
          do anyway.
        :return:
        """
        # compare exisiting config with fresh one we read in.
        self.model.readObs()  # should prob be init.
        m = ModelSimulation.ModelSimulation(self.model.dirPath, verbose=self.verbose)
        self.assertEqual(m.get(), self.model.get())
        # verify we fail to modify
        with self.assertRaises(Exception):
            m.set({'someInfo', {'silly': 2, 'silly_more': 3}})
        # make an updatable version.
        m = ModelSimulation.ModelSimulation(self.model.dirPath, update=True, verbose=self.verbose)
        self.assertEqual(m.get(), self.model.get())
        # verify can  modify
        m.set({'someInfo': {'silly': 2,
                            'silly_more': 3}})  # this will get written through but next read should overwrite.
        # now config should be different from ref case. Read it in and verify
        m = ModelSimulation.ModelSimulation(self.model.dirPath, verbose=self.verbose)
        self.assertNotEqual(m.get(), self.model.get())

    def test_readObs(self):
        """
        Test that readObs works
        :return:
        """
        # do by changing the observations and then rereading..
        Initobs = self.model.readObs()
        self.assertEqual(type(Initobs), dict)

        # modify values
        obs= Initobs.copy()
        obs['rh@500_nhx'] = False
        obs['NOOBS'] = 2.0
        self.model.set({'observations': obs}, write=False)

        # now read from cache so values should be the same
        mobs = self.model.readObs()
        self.assertEqual(mobs, obs) # reading from cache.
        # read from filesystem (flush cache)
        mobs= self.model.readObs(flush=True) # force reread
        self.assertNotEqual(mobs,obs) # different from mod obs
        self.assertEqual(mobs,Initobs) # but should be the same as initial values
        mobs = self.model.readObs() # read using cache
        self.assertEqual(mobs, Initobs)
        # read with specified keys (that do not exist)

        obs = self.model.readObs(obsNames=['Missing_1','Missing_2'])
        expect=dict(Missing_1=None,Missing_2=None)
        self.assertEqual(obs, expect)




        # test that reading json file works. So create a dummy one!
        obsfile = os.path.join(self.dirPath, 'obs.json')
        # add an extra obs which should be ignored unless fill set.
        mobs2 = mobs.copy()
        mobs2['fred'] = 2
        # strip out all the numpy stuff! 
        for k, v in mobs2.items():
            mobs2[k] = float(v)

        with open(obsfile, 'w') as fp:  # just dump the obs to the file.
            json.dump(mobs2, fp, indent=4)

        # hack the model to test json reading works
        self.model.setReadOnly(False)
        self.model.set({'ppOutputFile': 'obs.json'})
        obs = self.model.readObs(flush=True) # and flush the cache.
        self.assertTrue(np.all(pd.Series(obs)==pd.Series(mobs2)))


        # verify that series works...

        expect = pd.Series(self.model.readObs())
        obs = self.model.readObs(series=True)
        self.assertTrue(np.all(obs == expect))

    def test_writeObs(self):
        """
        Test that write obs works.
        :return:
        """
        # use default to write..
        obs = collections.OrderedDict([(key, number) for number, key in enumerate(self.obsNames)])
        self.model.writeObs(obs, verbose=self.verbose)
        # read it back in.
        obsGet= self.model.readObs()
        self.assertEqual(obsGet, obs)

        # now make it write to a json file.
        # need to make a new config.

        model = ModelSimulation.ModelSimulation(self.testDir,
                                                name='test', create=True,
                                                refDirPath=self.refDirPath,
                                                ppExePath='postProcess.sh',
                                                ppOutputFile='obs.json',
                                                parameters=self.parameters,
                                                obsNames=self.obsNames,
                                                verbose=self.verbose)
        obs[self.obsNames[0]] = 4.0
        model.writeObs(obs, verbose=self.verbose)
        # read it back in.
        model.readObs()
        obsGet = model.readObs()
        self.assertEqual(obsGet, obs)

        # now test we can write a series to a json file
        obs = pd.Series(obs)
        model.writeObs(obs,verbose=self.verbose)

        model = ModelSimulation.ModelSimulation(self.testDir,
                                                name='test', create=True,
                                                refDirPath=self.refDirPath,
                                                ppExePath='postProcess.sh',
                                                ppOutputFile='obs.nc',
                                                parameters=self.parameters,
                                                obsNames=self.obsNames,
                                                verbose=self.verbose)
        model.writeObs(obs, verbose=self.verbose)

    def test_set(self):
        """
        Test that set works by setting a value and then getting it back.
        set modified config file so will check that config file has been modified.
        :return:
        """
        # test without writing..
        self.model.set({"fbg": 1.0}, write=False)
        self.assertEqual(self.model.get('fbg'), 1.0)
        # then with writing
        m = ModelSimulation.ModelSimulation(self.model.dirPath, update=True)
        # get config file.
        config = m.readConfig()
        m.set({"fbg": 1.0})
        self.assertEqual(m.get('fbg'), 1.0)
        self.assertNotEqual(m.get(), config)  # on disk config should have changed...

    def test_setParams(self):
        """
        Test that setParams works
        :return:
        """
        param = collections.OrderedDict()
        param['p1'] = 1
        param['p2'] = 2
        self.model.setParams(param, write=False)
        self.assertEqual(self.model.get('parameters'), param)
        # verify addParam works
        self.model.setParams({'p3': 3}, write=False, addParam=True, verbose=self.verbose)
        param['p3'] = 3
        self.assertEqual(self.model.get('parameters'), param)

        # check a pandas series work
        params = pd.Series([1, 2, 3], index=['p1', 'p2', 'p3'])
        self.model.setParams(params, verbose=self.verbose, write=False)
        self.assertEqual(self.model.get('parameters').to_dict(), param)

    def test_genVarToNameList(self):
        """
        Test that genVarToNameList works
        :return:
        """

        self.model.genVarToNameList('RHCRIT', "RHCRIT", 'CNTL', 'ATMOS.NL')
        # check we've got what we think we should have
        expect = [ModelSimulation._namedTupClass(var='RHCRIT', namelist='CNTL', file='ATMOS.NL')]
        self.assertEqual(self.model._convNameList['RHCRIT'], expect)
        # try two var case with list of functions

    def test_registerMetaFn(self):
        """
        Test that registering meta function works.
        :return:
        """

        def fn(x=1.0, inverse=False, namelist=False):
            if inverse:
                return x['fn1'][0]
            else:
                return {'fn1': [x] * 19}

        def fn2(x=1.0, inverse=False, namelist=False):
            if inverse:
                return math.sqrt(x['fn2'][0])
            else:
                return {'fn2': [x ** 2] * 19}

        self.model.registerMetaFn('rhcrit', fn)
        self.model.registerMetaFn('rhcrit2', fn2)
        self.assertEqual(self.model._metaFn['rhcrit'], fn)
        self.assertEqual(self.model._metaFn['rhcrit2'], fn2)

    def test_applyMetaFn(self):
        """
        Test that applying metaFn works
        :return:
        """

        def fn(x=1.0, inverse=False, namelist=False):
            if inverse:
                return x['rhcrit'][0]
            else:
                return {'rhcrit': [x] * 19}

        def fn2(x=1.0, inverse=False, namelist=False):
            if inverse:
                return math.sqrt(x['rhcrit2'][0])
            else:
                return {'rhcrit2': [x ** 2] * 19}

        def fn3(x=2.0, inverse=False, namelist=False):
            if inverse:
                return x['rhcrit3'][2]
            else:
                return {'rhcrit': x + 2, 'rhcrit2': x / 2, 'rhcrit3': [x] * 19}

        self.model.registerMetaFn('rhcrit', fn, verbose=self.verbose)
        self.assertEqual(self.model.applyMetaFns(rhcrit=1.0), ({'rhcrit': [1.0] * 19}, ['rhcrit']))

        # verify failure happens when we force it!
        with self.assertRaises(Exception):
            print(self.model.applyMetaFns(fail=True, rhcrit2=1.0))
        # verify failure *does not* happen when we don't ask for it.
        self.assertEqual(self.model.applyMetaFns(rhcrit2=1.0), ({}, []))
        # add  a 2nd fn
        self.model.registerMetaFn('rhcrit2', fn2, verbose=self.verbose)
        self.assertEqual(self.model.applyMetaFns(rhcrit=1.0, rhcrit2=2.0),
                         ({'rhcrit': [1.0] * 19, 'rhcrit2': [4.0] * 19}, ['rhcrit', 'rhcrit2']))

        # verify failure happens when we force it!
        with self.assertRaises(Exception):
            print(self.model.applyMetaFns(fail=True, rhcrit=1.0, rhcrit3=1.0))

            # verify multiple parameter apply works..
        self.model.registerMetaFn('rhcritAll', fn3, verbose=self.verbose)
        self.assertEqual(self.model.applyMetaFns(rhcritAll=1.0),
                         ({'rhcrit': 3.0, 'rhcrit2': 0.5, 'rhcrit3': [1.0] * 19},
                          ['rhcritAll']))

    def test_writeNameList(self):
        """
        Test that namelist creation works
        :return:
        """
        # copy a trial namelist from configurations.
        infile = os.path.join(self.refDirPath, "CNTLATM")
        outFile = 'CNTLATM'
        outPath = os.path.join(self.dirPath, outFile)
        try:
            os.remove(outPath)  # remove the outPath
        except OSError:
            pass  # failed but that's OK will just copy file over..
        shutil.copyfile(infile, outPath)  # copy test file over
        backFile = outPath + "_nl.bak"
        # now to make configuration and patch outfile
        self.model._readOnly = False  # we can write to it.
        self.model.genVarToNameList('VF1', nameListVar='vf1', nameListName='slbc21', nameListFile=outFile)
        self.model.writeNameList(verbose=self.verbose, VF1=1.5)
        # should have a backup file
        self.assertTrue(os.path.isfile(backFile))
        # read the namelist
        with open(outPath) as nml_file:
            nml = f90nml.read(nml_file)
        self.assertEqual(nml['SLBC21']['vf1'], 1.5)  # check namelist modified
        # make vf1 change two variables...
        os.remove(outPath)
        os.rename(backFile, outPath)  # move backup file back again
        self.model.genVarToNameList('O2MMR', nameListVar='o2mmr', nameListName='runcnst', nameListFile=outFile)

        self.model.writeNameList(verbose=False, fail=True, VF1=1.5, O2MMR=0.21)
        #
        # should have a backup file
        self.assertEqual(os.path.isfile(backFile), True)
        # read the namelist
        with open(outPath) as nml_file:
            nml = f90nml.read(nml_file)
        self.assertEqual(nml['SLBC21']['vf1'], 1.5)  # check namelist modified
        self.assertEqual(nml['runcnst']['o2mmr'], 0.21)
        # add another nl variable.
        self.model.genVarToNameList('RHCRIT', nameListVar='rhcrit', nameListName='runcnst',
                                    nameListFile=outFile)
        # generate the namelist.
        rhcv = [0.65, 0.7]
        rhcv.extend(
            [0.8] * 17)  # there is a bug with f90nml version 0.19 which does not overwrite existing parts of the array
        self.model.writeNameList(verbose=self.verbose, VF1=1.5, RHCRIT=rhcv)
        self.assertEqual(os.path.isfile(backFile), True)
        # read the namelist
        with open(outPath) as nml_file:
            nml = f90nml.read(nml_file)
        self.assertEqual(nml['SLBC21']['vf1'], 1.5)  # check namelist modified
        self.assertEqual(nml['runcnst']['o2mmr'], 0.21)
        self.assertEqual(nml['runcnst']['rhcrit'], rhcv)

    def test_readNameList(self):
        """
        Test cases for readNameListVar
        :return:
        """
        outFile = 'CNTLATM'
        self.model.setReadOnly(False)
        self.model.genVarToNameList('RHCRIT', nameListVar='rhcrit', nameListName='runcnst',
                                    nameListFile=outFile)
        self.model.genVarToNameList('VF1', nameListVar='vf1', nameListName='slbc21', nameListFile=outFile)

        rhcv = [0.65, 0.7]
        rhcv.extend(
            [0.8] * 17)  # there is a bug with f90nml version 0.19 which does not overwrite existing parts of the array
        expect = collections.OrderedDict()
        expect['RHCRIT'] = rhcv
        expect['VF1'] = 1.5
        self.model.writeNameList(verbose=self.verbose, VF1=1.5, RHCRIT=rhcv)
        self.model.setReadOnly(True)
        vars = self.model.readNameList(['RHCRIT', 'VF1'], fail=True)
        self.assertDictEqual(vars, expect)

    def test_readMetaNameList(self):
        """
        test cases for readMetaNameList
        :return:
        """
        # set up meta fn and test it works... HadCM3 as will read the standard file.

        self.model.registerMetaFn('ALPHAM', iceAlbedo)  # register ice fn
        a = self.model.readMetaNameList('ALPHAM', verbose=self.verbose)
        self.assertEqual(a, 0.5)

    def test_cmp(self):
        """
        test cases for comparison operators
        :return:
        """
        import copy
        # create a modelSimulation by copying and compare. Then check that changes work

        model = copy.deepcopy(self.model)
        self.assertEqual(self.model, model)

        model = copy.deepcopy(self.model)
        model.set2(write=False, refDirPath='fred')
        self.assertNotEqual(self.model, model)

        # test change to parameters
        model = copy.deepcopy(self.model)
        model.setReadOnly(False)
        model.setParams({'aNewParam': 2}, addParam=True, write=False)
        self.assertNotEqual(self.model, model)
        # test change to observations
        model = copy.deepcopy(self.model)
        model.setReadOnly(False)
        model.set2(write=False, observations={'obs1': 2, 'obs3': 4})
        self.assertNotEqual(self.model, model)

    def test_runStatus(self):
        """
        Test that runStatus works
        Will do the following
            1) set nothing -- should return 'start'
            2) set it to 'start' -- should return start
            3) set it to 'continue' -- should return 'continue'
            4) set it to 'fred' -- should raise an error.
        :return: nada
        """

        self.assertEqual(self.model.runStatus(),'start') # case 1

        # case 2
        self.model.runStatus('start')
        self.assertEqual(self.model.runStatus(), 'start')

        # case 3
        self.assertEqual(self.model.runStatus('continue'), 'continue')

        # case 4
        with self.assertRaises(ValueError):
            self.model.runStatus('fred')





    def test_continueSimulation(self):
        """
        Test continueSimulation method.
            test that number of values returned is as expected and that runStatus is 'continue'
        :return: nada
        """

        result = self.model.continueSimulation()
        self.assertEqual(len(result), 1)

        result = self.model.continueSimulation()
        self.assertEqual(len(result), 2)
        self.assertEqual(self.model.runStatus(),'continue')

    def test_perturb(self):
        """
        Test perturb method.
        Run it multiple times -- perturbCount should increase each time.
        :return: nada
        """
        plist = ['p1', 'p2', 'p3']
        expect = []
        for p in plist:
            params = {p: 2.0}
            pList = self.model.perturb(params=params)
            expect.append(params)
            self.assertEqual(pList, expect)

        # check giving pertub nothing just returns the same list.
        for a in range(0, 2):
            pList = self.model.perturb()  # just get it back
            self.assertEqual(pList, expect)

    def test_perturbParams(self):
        """
        Tests pertrurbParams method works. Current implementation v crude. But OK as only
          likely want to perturb 1-3 times...
        :return: Nada
        """

        # test it by running twice. Perturbing each time.
        modParams1 = self.model.perturbParams()
        self.assertEqual(len(modParams1), 1, 'Len of pertrubation  dict not 1')
        p2 = self.model.perturb(modParams1)
        self.assertEqual(len(p2), 1, 'Len of pertubed list not 1')
        # do it again..
        modParams2 = self.model.perturbParams()
        self.assertEqual(len(modParams2), 2, 'Len of 2nd pertrubation  dict not 2')
        self.assertNotEqual(modParams1, modParams2, 'params the same..')

    def test_submit(self):
        """
        Test submit
        Following tests done:
            1) no arguments and get the base run.

        :return: nada
        """


if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  ## actually run the test cases

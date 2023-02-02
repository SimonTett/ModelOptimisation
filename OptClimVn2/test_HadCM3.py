"""
test_HadCM3: test cases for HadCM3 methods.
This test routine needs OPTCLIMTOP specified sensibly. (ideally so that $OPTCLIMTOP/um45 corresponds to
  where this routine lives..
"""
import collections
import filecmp
import os
import re
import shutil
import tempfile
import unittest
import pathlib  # TODO move towards using pathlib away from os.path.xxxx

import numpy as np
import pandas as pd

import HadCM3
import optClimLib


def cmp_lines(path_1, path_2, ignore=None, verbose=False):
    """
    From Stack exchange --
       http://stackoverflow.com/questions/23036576/python-compare-two-files-with-different-line-endings
    :param path_1: path to file 1 
    :param path_2: path to file 2
    :param ignore -- list of regep patterns to ignore
    :param verbose: default False -- if True print out lines that don't match.
    :return: True if files the same, False if not.
    """

    if ignore is None: ignore = []  # make ignore an empty list if None is provided.
    l1 = l2 = ' '
    with open(path_1, 'r') as f1, open(path_2, 'r') as f2:
        while l1 != '' and l2 != '':
            l1 = f1.readline()
            l2 = f2.readline()
            skip = False  # skip when set true and set so if ignore pattern found
            for p in ignore:
                if re.search(p, l1) or re.search(p, l2):
                    skip = True
                    continue
            if skip: continue  # go to next while bit of of looping over files.
            if l1 != l2:
                if verbose:
                    print(">>", l1)
                    print("<<", l2)
                return False
    return True


class testHadCM3(unittest.TestCase):
    """
    Test cases for HadCM3. There should be one for every method in HadCM3.

    """

    def setUp(self):
        """
        Setup case
        :return:
        """
        parameters = {"CT": 1e-4, "EACF": 0.5, "ENTCOEF": 3.0, "ICE_SIZE": 30e-6,
                      "RHCRIT": 0.7, "VF1": 1.0, "CW_LAND": 2e-4, "DYNDIFF": 12.0, "KAY_GWAVE": 2e4,
                      "ASYM_LAMBDA": 0.15, "CHARNOCK": 0.012, "G0": 10.0, "Z0FSEA": 1.3e-3, "ALPHAM": 0.5,
                      'START_TIME': [1997, 12, 1], 'RESUBMIT_INTERVAL': 'P40Y',
                      'RUNID': 'a0101', 'ASTART': '$MYDUMPS/fred.dmp',
                      "SCAVENGE": 2.0,'IA_N_DROP_MIN':4E7, 'IA_KAPPA_SCALE':0.5 , 'IA_N_INFTY': 3.75E89,
                      'SPHERICAL_ICE': False, 'OcnIceDiff': 2.5e-5, 'IceMaxConc': 0.99, 'OcnIsoDiff': 800}

        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)  # used throughout.
        refDir = pathlib.Path('Configurations') / 'xnmea'  # need a coupled model.
        simObsDir = 'test_in'
        self.dirPath = testDir
        self.refPath = refDir
        self.tmpDir = tmpDir  # really a way of keeping in context
        self.testDir = testDir
        # refDir = os.path.expandvars(os.path.expanduser(refDir))
        # simObsDir = os.path.expandvars(os.path.expanduser(simObsDir))
        shutil.rmtree(testDir, onerror=optClimLib.errorRemoveReadonly)

        self.model = HadCM3.HadCM3(testDir, name='a0101', create=True, refDirPath=refDir,
                                   ppExePath='postProcess.sh',
                                   ppOutputFile='obs.nc', runTime=1200, runCode='test',
                                   obsNames=['temp@500_nhx', 'temp@500_tropics', 'temp@500_shx'],
                                   verbose=False, parameters=parameters)

        shutil.copy(os.path.join(simObsDir, '01_GN', 'h0101', 'observables.nc'),
                    os.path.join(testDir, 'obs.nc'))  # copy over a netcdf file of observations.

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        optClimLib.delDirContents(self.tmpDir.name)
        # print("Remove by hand ",self.testDir)

    def test_init(self):
        """
        Test init
        :return:
        """
        # possible tests
        # 1) Test that changed name and directory has been changed.
        #  Note name and directory don't need to be the same.
        """
        Test init methods works.
        Probably over-kill and could be a pain if internal details change.
        But idea is that public methods all get a work out when modelSimulation initialised.
        :return:
        """
        # using implicit run of setup.
        expectObs = collections.OrderedDict()
        for k in ['temp@500_nhx', 'temp@500_tropics', 'temp@500_shx']: expectObs[k] = None
        expectParam = {"CT": 1e-4, "EACF": 0.5, "ENTCOEF": 3.0, "ICE_SIZE": 30e-6,
                       "RHCRIT": 0.7, "VF1": 1.0, "CW_LAND": 2e-4, "DYNDIFF": 12.0, "KAY_GWAVE": 2e4,
                       "SPHERICAL_ICE": False, 'RESUBMIT_INTERVAL':'P40Y','OcnIceDiff': 2.5e-5, 'IceMaxConc': 0.99, 'OcnIsoDiff': 800,
                       "ASYM_LAMBDA": 0.15, "CHARNOCK": 0.012, "G0": 10.0, "Z0FSEA": 1.3e-3, "ALPHAM": 0.5,
                       "SCAVENGE": 2.0,'IA_N_DROP_MIN':4E7, 'IA_KAPPA_SCALE':0.5 , 'IA_N_INFTY': 3.75E89,
                       'START_TIME': [1997, 12, 1], 'RUNID': 'a0101', 'ASTART': '$MYDUMPS/fred.dmp'}
        self.assertEqual(self.model.get(['name']), 'a0101')
        self.assertEqual(self.model.get(['ppExePath']), 'postProcess.sh')
        self.assertEqual(self.model.get(['observations']), expectObs)
        self.assertDictEqual(self.model.get(['parameters']), expectParam)
        self.assertEqual(self.model.get(['ppOutputFile']), 'obs.nc')
        self.assertListEqual(list(self.model.get(['observations']).keys()), list(expectObs.keys()))
        # test that read works. Works means no failures and have observations..

        m = HadCM3.HadCM3(self.dirPath, verbose=True)
        # Nothing should have changed except observations have been read in
        self.assertEqual(m.get(['name']), 'a0101')
        self.assertEqual(m.get(['ppExePath']), 'postProcess.sh')
        self.assertDictEqual(m.get(['parameters']), expectParam)

        # self.assertEqual(self.model.config['refDir'], None)
        self.assertEqual(m.get(['ppOutputFile']), 'obs.nc')
        self.assertListEqual(list(m.getObs().keys()), list(expectObs.keys()))
        self.assertNotEqual(m.getObs(), expectObs)
        # test that consistency checks work

        with self.assertRaises(NameError):
            m = HadCM3.HadCM3(self.dirPath, verbose=True, create=True,
                              parameters={'AINITIAL': 'test.dmp', 'ASTART': 'test2.dmp'})

        with self.assertRaises(NameError):
            m = HadCM3.HadCM3(self.dirPath, verbose=True, create=True,
                              parameters={'OINITIAL': 'test.dmp', 'OSTART': 'test2.dmp'})
        # test that read state fails if param provided.

        with self.assertRaises(Exception):
            m = HadCM3.HadCM3(self.dirPath, verbose=True, parameters={'VF1': 2.5})

        ## now  to test that update works.
        m = HadCM3.HadCM3(self.dirPath, verbose=True, update=True,
                          parameters={'RHCRIT': 0.6, 'RUNID': 'abcde'})
        expectParam['RHCRIT'] = 0.6
        expectParam['RUNID'] = 'abcde'
        # observations should have changed but nothing else.
        self.assertEqual(m.get(['name']), 'a0101')
        self.assertEqual(m.get(['ppExePath']), 'postProcess.sh')
        # self.assertEqual(m.get(['observations']),expectObs)
        self.assertDictEqual(m.get(['parameters']), expectParam)
        # self.assertEqual(self.model.config['refDir'], None)
        self.assertEqual(m.get(['ppOutputFile']), 'obs.nc')
        self.assertListEqual(list(m.getObs().keys()), list(expectObs.keys()))
        self.assertNotEqual(m.getObs(), expectObs)

    def test_readMetaParams(self):
        """
        Test that HadCM3 specific meta functions all work..by running the inverse function and checking we got
          what we put in.
        :return:
        """
        verbose = False
        expect_values = {"CT": 1e-4, "EACF": 0.5, "ENTCOEF": 3.0, "ICE_SIZE": 30e-6,
                         "RHCRIT": 0.7, "VF1": 1.0, "CW_LAND": 2e-4, "DYNDIFF": 12.0, "KAY_GWAVE": 2e4,
                         'SPHERICAL_ICE': False,
                         "SCAVENGE":2.0,'IA_N_DROP_MIN':4E7, 'IA_KAPPA_SCALE':0.5 , 'IA_N_INFTY': 3.75E89,
                         "ASYM_LAMBDA": 0.15, "CHARNOCK": 0.012, "G0": 10.0, "Z0FSEA": 1.3e-3, "ALPHAM": 0.5,
                         "OcnIceDiff": 2.5e-5, 'IceMaxConc': 0.99, 'OcnIsoDiff': 800,
                         'RUN_TARGET': [180, 0, 0, 0, 0, 0],
                         'START_TIME': [1997, 12, 1, 0, 0, 0], 'RUNID': 'a0101',
                         'RESUBMIT_INTERVAL': [40, 0, 0, 0, 0, 0],
                         'OSTART': '$DATAW/$RUNID.ostart', 'ASTART': '$MYDUMPS/fred.dmp',
                         'AINITIAL': '$MY_DUMPS/aeabra.daf4c10', 'OINITIAL': '$MY_DUMPS/aeabro.daf4c10'}
        got_values = {}
        for k in self.model._metaFn.keys():  # test all meta functions
            got_values[k] = self.model.readMetaNameList(k, verbose=verbose)

        for k, v in got_values.items():  # iterate over meta functions.
            if isinstance(v, str):
                self.assertEqual(v, expect_values[k])
            elif isinstance(v, list):
                self.assertEqual(v, expect_values[k])
            else:
                self.assertAlmostEqual(v, expect_values[k], delta=expect_values[k] * 1e-4,
                                       msg='Failed to almost compare for %s got %.4g expect %.4g' % (
                                           k, v, expect_values[k]))

        # tests that namelists are as expected -- per function...
        # IceMaxConc  NH 00.99, SH 0.98
        names =dict(IceMaxConc=[0.99, 0.98], CW_LAND=[2e-4, 5e-5], OcnIceDiff=[2.5e-5, 2.5e-5],
                    OcnIsoDiff=[800, 800],SCAVENGE=[1.3e-4, 5.99e-5]) # name of var + expected value for what it sets
        # read in namelists
        cases = self.model.readNameList(names.keys(), verbose=verbose,
                                        full=True)
        for k,expected in names.items(): # test all cases.
            self.assertEqual(list(cases[k].values()), expected,msg=f'failed to compare for {k} ')

    def test_setParams(self):
        """
        Test setParams
        :return:
        """
        # will test that can set namelist variables, that setting something that doesn't exist fails.
        self.model.setReadOnly(False)  # want to modify model.
        # param is dict of parmaters that map directly to namelist variables.
        param = {'VF1': 1.5, 'ICE_SIZE': 32e-6, 'ENTCOEF': 0.6, 'CT': 2e-4, 'ASYM_LAMBDA': 0.2,
                 'CHARNOCK': 0.015, 'G0': 20.0, 'Z0FSEA': 1.5e-3, 'AINITIAL': 'fred.dmp', 'OINITIAL': 'james.dmp',
                 'CLOUDTAU': 1.08E4, 'NUM_STAR': 1.0e6,  'OHSCA': 1.0, 'VOLSCA': 1.0,
                 'ANTHSCA': 1.0,  # Aerosol params
                 'RAD_AIT': 24E-9, 'RAD_ACC': 95E-9,
                 'IA_N_DROP_MIN': 3.5E7, 'IA_N_INFTY':3.75e8, 'IA_KAPPA_SCALE':0.9375}  # aerosol impact on clouds
        metaParam = {'KAY_GWAVE': 2e4, 'ALPHAM': 0.65, 'CW_LAND': 1e-3, 'RHCRIT': 0.666, 'EACF': 0.777,
                     'DYNDIFF': 11.98, 'RUN_TARGET': [2, 1, 1, 0, 0, 0],
                     'OcnIceDiff': 3.0e-5, 'IceMaxConc': 0.99, 'OcnIsoDiff': 800,'SCAVENGE':2}
        un = param.copy()
        expect = un.copy()
        for k, v in metaParam.items():
            un[k] = v
            if type(v) == np.ndarray: v = v.round(3)
            expect[k] = v

        self.model.setParams(un, fail=True, verbose=True)
        # verify namelists are as expected.
        vars = self.model.readNameList(expect.keys(), verbose=True, fail=True)

        for k in expect.keys():
            msg = 'Key is %s' % (k)
            print("vars[%s]=%s got %s" % (k, vars[k], expect[k]))
            if type(expect[k]) == list:
                self.assertEqual(vars[k], expect[k], msg=msg)
            else:
                self.assertAlmostEqual(expect[k], vars[k], msg=msg)

        # check pd.Series works
        series = pd.Series(un)
        series['VF1'] = 1.75
        expect['VF1'] = 1.75
        self.model.setReadOnly(False)  # want to modify model
        self.model.setParams(series, fail=True, verbose=True)
        # verify namelists are as expected.
        vars = self.model.readNameList(expect.keys(), verbose=True, fail=True)

        for k in expect.keys():
            print("vars[%s]=%s got %s" % (k, vars[k], expect[k]))
            if type(expect[k]) == list:
                self.assertEqual(vars[k], expect[k], msg=msg)
            else:
                self.assertAlmostEqual(expect[k], vars[k], msg=msg)

    def test_modifyScript(self):
        """
        Test modifyScript produces expected result
        and fails if it tries to modify something already modifies
        :return: 
        """

        # testing modifyScript is tricky. I did use comparison with a file but I think this is difficult to maintain.
        # better to count the number of lines with ## modified at the end.
        modifyStr = '## modified *$'
        file = os.path.join(self.model.dirPath, 'SCRIPT')
        count = 0
        with open(file, 'r') as f:
            for line in f:
                if re.search(modifyStr, line): count += 1

        # expected changes are  exprID, jobID, MY_DATADIR, 4 DATA[M,W,U,T] +2 more DATA [M,W]+ the postProcessing script -- this depends on config.
        # If emailing then will expect more changes.
        expect = 10
        # Note config this being tested on has no MY_DATADIR/A
        self.assertEqual(count, expect, 'Expected %d %s got %d' % (expect, modifyStr, count))

        self.assertRaises(Exception, self.model.modifyScript)
        #

    def test_modifySubmit(self):
        """
        Test modifySubmit
        :return: 
        """

        # no need to run modifySubmit as already ran when init happens.

        modifyStr = '## modified$'
        file = os.path.join(self.model.dirPath, 'SUBMIT')
        count = 0
        with open(file, 'r') as f:
            for line in f:
                if re.search(modifyStr, line): count += 1
        # expected changes are CJOBN, RUNID, JOBDIR, MY_DATADIR, 2 [NR]RUN+_TIME_LIMIT, ACCOUNT and 5 DATADIR/$RUNID
        expect = 10
        self.assertEqual(count, expect, f'Expected {expect} {modifyStr} got {count}')

        self.assertRaises(Exception, self.model.modifySubmit)  # run modify a 2nd time and get an error.

    ## couple of methods  to allow checking of submit and script

    def check_Submit(self, print_output=False, runType='CRUN', expect=None, expectMod=None):
        """
        Simple way of checking script OK!
        :param self:
        :return:
        """

        modifyStr = r'## modified\s*$'
        modifyStrCont = r'## modifiedContinue\s*$'
        if runType == 'NRUN':
            modifyStrCont = r'## modifiedRestart\s*$'
        file = os.path.join(self.model.dirPath, 'SUBMIT')
        count = 0
        countMC = 0
        with open(file, 'r') as f:
            for line in f:
                line = line[0:-1]  # strip newline
                if re.search(modifyStr, line):
                    count += 1
                    strP = "M :",

                elif re.search(modifyStrCont, line):
                    countMC += 1
                    strP = "MC:"
                else:
                    strP = "  :"
                if print_output: print(strP, line)
                # check crun code
                if re.search("^STYPE=" + runType, line):
                    self.fail(f"Got {line} should have {runType}")
                if re.search("^STEP=1 ", line):
                    self.fail("Got step=1 should have step=4")

        self.assertEqual(count, expect, 'Expected %d %s got %d' % (expect, modifyStr, count))
        expect = 1  # one TYPE=CRUN/NRUN
        self.assertEqual(countMC, expect, 'Expected %d %s got %d' % (expectMod, modifyStrCont, countMC))

    def check_script(self, print_output=False, runType='CRUN', expect=None, expectMod=None):
        modifyStr = r'## modified\s*$'
        modifyStrCont = r'## modifiedContinue\s*$'
        if runType == 'NRUN':
            modifyStrCont = r'## modifiedRestart\s*$'
        file = os.path.join(self.model.dirPath, 'SCRIPT')
        countMC = 0
        count = 0
        line_count = 0
        with open(file, 'r') as f:
            for line in f:
                line = line[0:-1]  # strip newline
                if re.search(modifyStrCont, line):
                    countMC += 1
                    strP = "MC%5d: %s"

                elif re.search(modifyStr, line):
                    count += 1
                    strP = " M%5d: %s"
                else:  # just print line
                    strP = "  %5d: %s"

                if print_output:
                    print(strP % (line_count, line))
                line_count += 1
        self.assertEqual(count, expect, 'Expected %d %s got %d' % (expect, modifyStr, count))
        self.assertEqual(countMC, expectMod, 'Expected %d %s got %d' % (expectMod, modifyStrCont, countMC))

    def test_genContSUBMIT(self):
        """
        test that contSUBMIT works.
        Tests are that it only had two modification marks in the continuation SUBMIT script
        """

        self.model.genContSUBMIT()  # create the file.
        # should be different from std file.
        f1 = self.model.submit()
        f2 = self.model.submit('continue')
        self.assertFalse(filecmp.cmp(f2, f1, shallow=False))
        # and count the number of modfy marks.
        modifyStrCont = '## modifiedContinue$'
        modifyStr = '## modified$'

        count = 0
        mc = 0
        with open(f2, 'r') as f:
            for line in f:
                if re.search(modifyStr, line):
                    count += 1
                elif re.search(modifyStrCont, line):
                    mc += 1

        # expected changes (from orginal creation) are:
        # CJOBN, RUNID, JOBDIR, MY_DATADIR, 2 [NR]RUN+_TIME_LIMIT, ACCOUNT and 5 DATADIR/$RUNID
        expect = 10
        self.assertEqual(count, expect, msg=f'Expected {count}  {modifyStr} got {count}')
        self.assertEqual(count, expect, 'Expected %d %s got %d' % (expect, modifyStr, count))
        # expect a further  changes -- CRUN. Reference case has STEP = 1
        expect = 1
        self.assertEqual(mc, expect, msg=f'Expected {count}  {modifyStrCont} got {count}')

    def test_createWorkDir(self):
        """
        Test that createWorkDir worked as expected
        :return: 
        """
        # no need to run createWorkDIr as already ran by init
        # just check it exists and is a dir
        self.assertTrue(os.path.isdir(os.path.join(self.model.dirPath, 'W')))

    def test_fixClimFCG(self):
        """
        Test that fixClimFCG works as expects 
         
         converting all CLIM_FCG_YEARS(1,..) to CLIM_FCG_YEARS(:,...)
        :return: 
        """

        # test is that have NO lines in CNTLALL that look like (1,\w[0-9])

        file = os.path.join(self.model.dirPath, 'CNTLATM')
        with open(file, 'r') as f:
            for line in f:
                if re.search(r'\(1\w*,\w*[0-9]*\w\)=\w*[0-9]*,', line): self.fail("Found bad namelist %s" % line)

    def test_submit(self):
        """
        Test the submit method works -- returns sensible path.
        Rather trivial test.. 
        :return: 
        """
        dir = pathlib.Path(self.model.dirPath)
        p = pathlib.Path(self.model.submit())
        self.assertEqual(p, dir / 'SUBMIT')
        p2 = pathlib.Path(self.model.submit('continue'))
        self.assertEqual(p2, dir / 'SUBMIT.cont')

    def test_perturb(self):
        """
        Test that perturbation works.
          Need to look at namelists so a bit tricky...
        :return:
        """
        self.model.perturb(verbose=True, params=collections.OrderedDict(VF1=2.0))
        # get the namelist values back in
        values = self.model.readNameList(['VF1'], verbose=True)
        expect = collections.OrderedDict(VF1=2.0)
        self.assertEqual(values, expect)

    def test_createPostProcessFile(self):
        """
        Test that creation of post processing script works.

        :return:
        """
        release_cmd = 'ssh login qrls 999999.1'
        file = self.model.createPostProcessFile(release_cmd)
        # expect file to exist
        self.assertTrue(file.exists())
        # and that it is as expected.
        self.assertEqual(file, pathlib.Path(self.dirPath) / self.model.postProcessFile)
        # need to check its contents...
        # will check two things. 1) that SUBCONT is as expected 2) that the qrls cmd is in the file
        submit_script = self.model.submit('continue')
        cntSUBCONT = 0
        cntqrls = 0
        with open(file, 'r') as f:
            for line in f:
                if line.find(release_cmd) != -1:
                    cntqrls += 1
                elif line.find(f'SUBCONT={submit_script}') != -1:
                    cntSUBCONT += 1

        self.assertEqual(cntqrls, 1, 'Expected only 1 qrls cmd')
        self.assertEqual(cntSUBCONT, 1, 'expected only 1 SUBMITCMD')

    def test_parse_isoduration(self):
        """
        Test parse_isoduration

        :return: Nada. Just runs tests.
        """

        strings = ['P1Y1M1DT1H1M1.11S', 'P2Y', 'PT1M', 'PT1M2.22S', 'P-1Y']
        expected = [[1, 1, 1, 1, 1, 1.11], [2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 2.22],
                    [-1, 0, 0, 0, 0, 0]]

        for s, e in zip(strings, expected):
            got = HadCM3.parse_isoduration(s)
            self.assertEqual(e, got)

        # should fail without a leading P or not a string
        for fail_case in ['12Y', [1, 0, 0, 0, 0, 0]]:
            with self.assertRaises(ValueError):
                got = HadCM3.parse_isoduration(fail_case)
            #

    def test_startTime(self):
        """
        Tests startTime to see if it works
        :return:
        """
        tests = [[2020, 1, 1], [1990], [1999, 9, 9, 1, 1, 1],
                 '2020-01-01', '1990-01-01', '1999-09-09 01:01:01']
        expect = [[2020, 1, 1, 0, 0, 0], [1990, 1, 1, 0, 0, 0], [1999, 9, 9, 1, 1, 1],
                  [2020, 1, 1, 0, 0, 0], [1990, 1, 1, 0, 0, 0], [1999, 9, 9, 1, 1, 1]]

        for t, e in zip(tests, expect):  # loop over tests
            got = HadCM3.startTime(t)  # got is a dict so iterate over keys
            for k, v in got.items():
                self.assertEqual(v, e, msg=f"Failed for {t} in nl info {k}")

    def test_timeDelta_fns(self):
        """
        Tests timeDelta, runTarget & resubInterval  to check they all work

        :return: nada
        """

        tests = [[6], [6, 1], [6, 1, 1], [6, 1, 1, 1, 1, 1],
                 'P7Y', 'P7Y1M', 'P7Y1M1D', 'P7Y1M1DT1H1M1S']
        expect = [[6, 0, 0, 0, 0, 0], [6, 1, 0, 0, 0, 0], [6, 1, 1, 0, 0, 0], [6, 1, 1, 1, 1, 1],
                  [7, 0, 0, 0, 0, 0], [7, 1, 0, 0, 0, 0], [7, 1, 1, 0, 0, 0], [7, 1, 1, 1, 1, 1]]

        for t, e in zip(tests, expect):  # loop over tests
            for fn in [HadCM3.timeDelta, HadCM3.runTarget, HadCM3.resubInterval]:
                got = fn(t)  # got is a dict so iterate over keys

                for k, v in got.items():
                    self.assertEqual(v, e, msg=f"Failed for {t} in nl info {k} using fn {fn}")


if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  ## actually run the test cases

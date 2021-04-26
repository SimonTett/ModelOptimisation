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

import numpy as np
import pandas as pd

import HadCM3
import optClimLib


def cmp_lines(path_1, path_2, ignore=None, verbose=False):
    """
    From Stack exchange -- http://stackoverflow.com/questions/23036576/python-compare-two-files-with-different-line-endings
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
                      'START_TIME': [1997, 12, 1], 'RUNID': 'a0101', 'ASTART': '$MYDUMPS/fred.dmp',
                      'SPHERICAL_ICE': False,'OcnIceDiff':2.5e-5,'IceMaxConc':0.99,'OcnIsoDiff':800}

        tmpDir = tempfile.TemporaryDirectory()
        print("tmpDir is %s" % (tmpDir.name))

        testDir = tmpDir.name  # used throughout.
        refDir = os.path.join('Configurations', 'xnmea')  # need a coupled model.
        simObsDir = 'test_in'
        self.dirPath = testDir
        self.refPath = refDir
        self.tmpDir = tmpDir  # really a way of keeping in context
        self.testDir = testDir
        refDir = os.path.expandvars(os.path.expanduser(refDir))
        simObsDir = os.path.expandvars(os.path.expanduser(simObsDir))
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
                       "SPHERICAL_ICE": False,'OcnIceDiff':2.5e-5, 'IceMaxConc':0.99,'OcnIsoDiff':800,
                       "ASYM_LAMBDA": 0.15, "CHARNOCK": 0.012, "G0": 10.0, "Z0FSEA": 1.3e-3, "ALPHAM": 0.5,
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
        Test that HadCM3 specific meta functions all work..by runnign the inverse function and checking we got
          what we put in.
        :return:
        """
        verbose = False
        expect_values = {"CT": 1e-4, "EACF": 0.5, "ENTCOEF": 3.0, "ICE_SIZE": 30e-6,
                         "RHCRIT": 0.7, "VF1": 1.0, "CW_LAND": 2e-4, "DYNDIFF": 12.0, "KAY_GWAVE": 2e4,
                         'SPHERICAL_ICE': False,
                         "ASYM_LAMBDA": 0.15, "CHARNOCK": 0.012, "G0": 10.0, "Z0FSEA": 1.3e-3, "ALPHAM": 0.5,
                         "OcnIceDiff":2.5e-5, 'IceMaxConc':0.99,'OcnIsoDiff': 800,'RUN_TARGET': [180, 0, 0, 0, 0, 0],
                         'START_TIME': [1997, 12, 1], 'RUNID': 'a0101',
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
                self.assertAlmostEqual(v, expect_values[k], delta=expect_values[k]*1e-4,
                                       msg='Failed to almost compare for %s got %.4g expect %.4g' % (
                                           k, v, expect_values[k]))

        # tests that namelists are as expected -- per function...
        # IceMaxConc  NH 00.99, SH 0.98
        cases = self.model.readNameList(['IceMaxConc',"CW_LAND","OcnIceDiff",'OcnIsoDiff'],verbose=verbose,full=True)
        self.assertEqual(list(cases['IceMaxConc'].values()),[0.99,0.98])
        self.assertEqual(list(cases['CW_LAND'].values()), [2e-4, 5e-5])
        self.assertEqual(list(cases['OcnIceDiff'].values()), [2.5e-5, 2.5e-5])
        self.assertEqual(list(cases['OcnIsoDiff'].values()), [800, 800])

    def test_setParams(self):
        """
        Test setParams
        :return:
        """
        # will test that can set namelist variables, that setting something that doesn't exist fails.
        self.model.setReadOnly(False)  # want to modify model.
        # param is dict of parmaetrs that map directly to namelist variables.
        param = {'VF1': 1.5, 'ICE_SIZE': 32e-6, 'ENTCOEF': 0.6, 'CT': 2e-4, 'ASYM_LAMBDA': 0.2,
                 'CHARNOCK': 0.015, 'G0': 20.0, 'Z0FSEA': 1.5e-3, 'AINITIAL': 'fred.dmp', 'OINITIAL': 'james.dmp',
                 'CLOUDTAU': 1.08E4, 'NUM_STAR': 1.0e6, 'L0': 6.5E-5, 'L1': 2.955E-5, 'OHSCA': 1.0, 'VOLSCA': 1.0,
                 'ANTHSCA': 1.0,  # Aerosol params
                 'RAD_AIT': 24E-9, 'RAD_ACC': 95E-9,
                 'N_DROP_MIN': 3.5E7, 'IA_AERO_POWER': 2.5e-9, 'IA_AERO_SCALE': 3.75E8}  # aerosol impact on clouds
        metaParam = {'KAY_GWAVE': 2e4, 'ALPHAM': 0.65, 'CW_LAND': 1e-3, 'RHCRIT': 0.666, 'EACF': 0.777,
                     'DYNDIFF': 11.98, 'RUN_TARGET': [2, 1, 1, 0, 0, 0],
                     'OcnIceDiff': 3.0e-5, 'IceMaxConc': 0.99, 'OcnIsoDiff': 800}
        un = collections.OrderedDict()
        for k, v in param.items():
            un[k] = v
        expect = un.copy()
        # got problem here.
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

        # testing modifyScript is tricky. I did use comparision with a file but I think this is difficult to maintain.
        # better to count the number of lines with ## modified at the end.
        modifyStr = '## modified *$'
        file = os.path.join(self.model.dirPath, 'SCRIPT')
        count = 0
        with open(file, 'r') as f:
            for line in f:
                if re.search(modifyStr, line): count += 1

        # expected changes are  exprID, jobID, 4 DATA[M,W,U,T] +2 more DATA [M,W]+ MY_DATADIR + mark (5 changes). -- this depends on config.
        # If emailing then will expect more changes.
        expect = 14
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
        self.assertEqual(count, expect, 'Expected %d %s got %d' % (expect, modifyStr, count))

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


    def check_script(self, print_output=False, runType='CRUN',expect=None,expectMod=None):
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

    ###


    def test_continueSimulation(self):
        """
        Test continueSimulation method
        :return:
        """
        import fileinput


        print_output = False  # set True to print out output

        # pretend run has been submitted which means adding something to model.
        model = self.model
        with fileinput.input(model.postProcessFile, inplace=1, backup='.bak2') as f:
            for line in f:
                line = line[0:-1]
                # if m.postProcessFile does not exist then  get an error which is what we want!
                # fix your model method!
                if model.postProcessMark in line:  # got the mark so add some text.
                    print('SOME VERY LONG COMMAND TO RUN MODEL', 'qrls ', '201012.1',
                          '"')  # this releases the post processing job.
                else:
                    print(line)  # just print the line out.


        # expected changes are RUNID, JOBDIR, MY_DATADIR,MY_OUTPUT, MY_DATADIR, MY_OUTPUT
        # NRUN_TIME_LIMIT, CRUN_TIME_LIMIT, CJOBN, ACCOUNT, TMPDIR, MY_OUTPUT
        # expectedMod is TYPE,
        self.model.continueSimulation(minimal=True)
        self.check_Submit(print_output=print_output,expect=10,expectMod=1)

        self.model.continueSimulation()
        self.check_Submit(print_output=print_output,expect=10,expectMod=1)


        # check SCRIPT as well.
        self.check_script(print_output=print_output,expect=9,expectMod=5)

        # Compare SCRIPT & SCRIPT.bakR
        file = os.path.join(self.model.dirPath, 'SCRIPT')
        fileBak = file + '.bakR'
        import difflib
        if not filecmp.cmp(file, fileBak):  # files are different so show differences.
            with open(file) as ff:
                fromlines = ff.readlines()
            with open(fileBak) as tf:
                tolines = tf.readlines()

            delta = difflib.context_diff(fromlines, tolines, fromfile=file, tofile=fileBak)
            for d in delta:
                print(d[:-1])
        else:
            self.fail('Files %s & %s are the same ' % (file, fileBak))


    def test_restartSimulation(self):
        """
        Test that restrtSimulation work.
        basically same as continueSimulation except have NRUN
        :return:
        """
        modifyStr = r'## modified\s*$'
        modifyStrCont = r'## modifiedRestart\s*$'

        print_output = False  # set True to print out output

        # pretend run has been continued so we are undoing this.

        model = self.model
        model.continueSimulation()
        self.model.restartSimulation(minimal=True)
        # check submit
        # expected changes are RUNID, JOBDIR, MY_DATADIR,MY_OUTPUT, MY_DATADIR, MY_OUTPUT
        # NRUN_TIME_LIMIT, CRUN_TIME_LIMIT, CJOBN, ACCOUNT, TMPDIR, MY_OUTPUT
        # expectedMod is TYPE,

        self.check_Submit(print_output=print_output,runType='NRUN',expect=10,expectMod=1)
        # check Script file as expected -- minimal change so no change
        self.check_script(print_output=print_output,runType='NRUN',expect=9,expectMod=0)
        # now do restartSimulation with minimal set to false  (default)
        self.model.restartSimulation()
        # check submit
        self.check_Submit(print_output=print_output,runType='NRUN',expect=10,expectMod=1)
        # check Script file as expected --
        self.check_script(print_output=print_output,runType='NRUN',expect=9,expectMod=5)
        # Compare SCRIPT & SCRIPT.bakR
        file = os.path.join(self.model.dirPath, 'SCRIPT')
        fileBak = file + '.bakR'
        import difflib
        if not filecmp.cmp(file, fileBak):  # files are different so show differences.
            with open(file) as ff:
                fromlines = ff.readlines()
            with open(fileBak) as tf:
                tolines = tf.readlines()

            delta = difflib.context_diff(fromlines, tolines, fromfile=file, tofile=fileBak)
            for d in delta:
                print(d[:-1])
        else:
            self.fail('Files %s & %s are the same ' % (file, fileBak))

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
        p = self.model.submit()
        self.assertEqual(p, os.path.join(self.model.dirPath, 'SUBMIT'))

    def test_perturb(self):
        """
        Test that perturbation works.
          Need to look at namelists so a bit tricky...
        :return:
        """
        self.model.perturb(verbose=True,  params=collections.OrderedDict(VF1=2.0))
        # get the namelist values back in
        values = self.model.readNameList(['VF1'],verbose=True)
        expect  = collections.OrderedDict(VF1=2.0)
        self.assertEqual(values,expect)


if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  ## actually run the test cases

"""
test_HadCM3: test cases for HadCM3 methods.
This test routine needs OPTCLIMTOP specified sensibly. (ideally so that $OPTCLIMTOP/um45 corresponds to
  where this routine lives..
"""

import filecmp
import os
import re
import shutil
import tempfile
import unittest
import pathlib
import logging  # TODO remove all verbose/print and use logging. Means tests need to explicitly set logging up
import copy
from Models.HadCM3 import HadCM3 # so this import puts HadCM3 in the list of known models for the rest of the testing.
import genericLib
import importlib.resources


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
        logging.basicConfig(level=logging.DEBUG)
        parameters = dict(CT=1e-4, EACF=0.5, ENTCOEF=3.0, ICE_SIZE=30e-6,
                          RHCRIT=0.7, VF1=1.0, CW=2e-4, DYNDIFF=12.0, KAY_GWAVE=2e4,
                          ASYM_LAMBDA=0.15, CHARNOCK=0.012, G0=10.0, Z0FSEA=1.3e-3, ALPHAM=0.5,
                          START_TIME='1977-12-01T00:00:00', RESUBMIT_INTERVAL='P40Y',
                          ASTART='$MYDUMPS/fred.dmp',
                           IA_N_DROP_MIN=4E7, IA_KAPPA_SCALE=0.5, IA_N_INFTY=3.75E89,
                          SPHERICAL_ICE=False, ICE_DIFF=2.5e-5, MAX_ICE=0.99, OCN_ISODIFF=800
                          )
        self.parameters = copy.deepcopy(parameters)

        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)  # used throughout.
        refDir = pathlib.Path('Configurations') / 'xnmea'  # need a coupled model.
        simObsDir = HadCM3.expand('$OPTCLIMTOP/test_in')
        #self.dirPath = testDir
        self.refPath = refDir
        self.tmpDir = tmpDir  # really a way of keeping in context
        self.testDir = testDir
        # refDir = os.path.expandvars(os.path.expanduser(refDir))
        # simObsDir = os.path.expandvars(os.path.expanduser(simObsDir))
        shutil.rmtree(testDir, onerror=genericLib.errorRemoveReadonly)
        # create a model and store it.
        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)  # used throughout.
        refDir = pathlib.Path(HadCM3.expand('$OPTCLIMTOP/Configurations')) / 'xnmea'  # need a coupled model.
        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', output_file='obs.json')
        self.post_process = post_process
        self.model = HadCM3(name='testM', reference=refDir,
                            model_dir=testDir, post_process=post_process,
                            parameters=parameters)
        self.tmpDir = tmpDir
        self.testDir = testDir  # for clean up!
        self.refDir = refDir
        self.config_path = self.model.config_path

        shutil.copy(simObsDir / '01_GN' / 'h0101' / 'observables.nc',
                    testDir / 'obs.nc')  # copy over a netcdf file of observations.

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        genericLib.delDirContents(self.tmpDir.name)
        # print("Remove by hand ",self.testDir)

    def test_instantiate(self):
        """
        Test instantiation.
        :return:
        """
        # possible tests
        # 1) Test that changed name and directory has been changed.
        #  Note name and directory don't need to be the same.
        """
        Test instantite works!
        :return:
        """

        self.model.instantiate()
        p = set(self.parameters.keys())

        p = self.model.read_values(p)
        for param,value in p.items():
            self.assertEqual(self.parameters[param],value,msg=f"Comparison failed for {param}")

        self.assertEqual(self.model.status,'INSTANTIATED')
        model = self.model.load_model(self.model.config_path)
        self.assertEqual(vars(model),vars(self.model ))


    def test_hadcm3_params(self):
        """
        Test that HadCM3 parameters workby setting them and reading them back in.
        :return:
        """

        expect_values = {"CT": 1e-4, "EACF": 0.5, "ENTCOEF": 3.0, "ICE_SIZE": 30e-6,
                         "RHCRIT": 0.7, "VF1": 1.0, "CW": 2e-4, "DYNDIFF": 12.0, "KAY_GWAVE": 2e4,
                         'SPHERICAL_ICE': False,
                         'IA_N_DROP_MIN': 4E7, 'IA_KAPPA_SCALE': 0.5, 'IA_N_INFTY': 3.75E89,
                         "ASYM_LAMBDA": 0.15, "CHARNOCK": 0.012, "G0": 10.0, "Z0FSEA": 1.3e-3, "ALPHAM": 0.5,
                         "ICE_DIFF": 2.5e-5, 'MAX_ICE': 0.99, 'OCN_ISODIFF': 800,
                         'RUN_TARGET': 'P180Y',
                         'START_TIME': "1997-12-01T00:00:00",
                         'RESUBMIT_INTERVAL': "P40Y",
                         'OSTART': '$DATAW/$RUNID.ostart', 'ASTART': '$MYDUMPS/fred.dmp',
                         'AINITIAL': '$MY_DUMPS/aeabra.daf4c10', 'OINITIAL': '$MY_DUMPS/aeabro.daf4c10',
                         "ANTHSCA": 0.99, "NAME": "XFRED", "OHSCA": 1.2, "VOLCSCA": 1.2}

        # check we have got all known parameters.
        expect_params = set(self.model.param_info.known_parameters())
        expect_params -= set(expect_values.keys())  # remove the ones we have.
        # and remove ensembleMember which is "special"
        expect_params -= set(['ensembleMember'])
        self.assertEqual(len(expect_params), 0, msg='Missing following parameters: ' + ", ".join(expect_params))
        # test can set and read.
        shutil.copytree(self.refDir, self.model.model_dir, symlinks=True, dirs_exist_ok=True)
        self.model.set_params(expect_values)
        got = self.model.read_values(list(expect_values.keys()))
        self.assertEqual(got, expect_values)


        # test that "functions" work as expected.,
        # tests that namelists are as expected -- per function...


        names = dict(MAX_ICE=[0.99, [0.99, 0.98]], CW=[2e-4, [2e-4, 5e-5]], ICE_DIFF=[2.5e-5, [2.5e-5, 2.5e-5]],
                     OCN_ISODIFF=[800, [800, 800]])  # name of var + expected value for what it sets
        for k, v in names.items():
            result = self.model.param_info.param(self.model, k, v[0])
            r = [v for (nl, v) in result]  # extract the values.
            self.assertEqual(r, v[1], msg=f'failed to compare for {k}')

    def notest_f90nml_patch(self):
        """
        Test to show problems with f90nml patch at vn 1.4.3
        :return:
        """
        import f90nml
        import time

        shutil.copy2(self.refDir / 'CNTLOCN', self.model.model_dir)
        ice_values = dict(EDDYDIFFN=2.50e-05, EDDYDIFFS=2.50e-05, AMXNORTH=0.9, AMXSOUTH=0.980)
        nl = f90nml.namelist.Namelist(SEAICENL=ice_values)
        #
        parser = f90nml.Parser()
        got = parser.read(self.model.model_dir / 'CNTLOCN')
        for k, v in got['seaicenl'].items():
            print("read", k, v)

        with open(self.model.model_dir / 'CNTLOCN.tmp2', 'w') as fd:
            got = parser.read(self.model.model_dir / 'CNTLOCN', nl, fd)
        time.sleep(1)
        got2 = f90nml.read(self.model.model_dir / 'CNTLOCN.tmp2')
        for k, v in got['seaicenl'].items():
            print("parser patch", k, v)
        for k, v in got2['seaicenl'].items():
            print("read patch", k, got2['seaicenl'][k])

        nl2 = f90nml.patch(str(self.model.model_dir / 'CNTLOCN'), nl, str(self.model.model_dir / 'CNTLOCN.tmp'))
        got = f90nml.read(self.model.model_dir / 'CNTLOCN.tmp')
        for k, v in got['seaicenl'].items():
            print("got", k, v)
            print("nl2", k, nl2['seaicenl'][k])

        for k, v in ice_values.items():
            self.assertEqual(got['seaicenl'][k], v, msg=f'failed to compare for {k}')



    def test_modifyScript(self):
        """
        Test modifyScript produces expected result
        and fails if it tries to modify something already modifies
        :return: 
        """

        # testing modifyScript is tricky. I did use comparison with a file but I think this is difficult to maintain.
        # better to count the number of lines with ## modified at the end.
        modifyStr = '## modified *$'
        shutil.copy2(self.refDir / 'SCRIPT', self.model.model_dir)
        resource = importlib.resources.files("OptClimVn3")
        set_status_script = str(resource.joinpath("scripts/set_model_status.py"))
        self.model.modifyScript(set_status_script)
        file = self.model.model_dir / 'SCRIPT'
        count = 0
        with open(file, 'r') as f:
            for line in f:
                if re.search(modifyStr, line): count += 1

        # expected changes are  set_status running, exprID, jobID, MY_DATADIR, 4 DATA[M,W,U,T] +2 more DATA [M,W]+ the postProcessing script -- this depends on config.
        # If emailing then will expect more changes.
        expect = 12
        # Note config this being tested one has no MY_DATADIR/A
        self.assertEqual(count, expect, f'Expected {expect} {modifyStr} got {count}')

        self.assertRaises(Exception, self.model.modifyScript)
        #

    def test_modifySubmit(self):
        """
        Test modifySubmit
        :return: 
        """

        modifyStr = '## modified$'
        shutil.copy2(self.refDir / 'SUBMIT', self.model.model_dir)
        self.model.modifySubmit()
        file = self.model.model_dir / 'SUBMIT'
        count = 0
        with open(file, 'r') as f:
            for line in f:
                if re.search(modifyStr, line): count += 1
        # expected changes are CJOBN, RUNID, JOBDIR, MY_DATADIR,  and 5 DATADIR/$RUNID
        expect = 7
        self.assertEqual(count, expect, f'Expected {expect} {modifyStr} got {count}')

        self.assertRaises(Exception, self.model.modifySubmit)  # run modify a 2nd time and get an error.
        # check there is a .bakR file identical to the original file.

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
        file = self.model.mdoel_dir/'SUBMIT'
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
        file = self.model.model_dir/'SCRIPT'
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
        shutil.copy2(self.refDir / "SUBMIT", self.model.model_dir)
        self.model.genContSUBMIT()  # create the file.
        # should be different from std file.
        f1 = self.model.model_dir / self.model.submit_script
        f2 = self.model.model_dir / self.model.continue_script
        self.assertFalse(filecmp.cmp(f2, f1, shallow=False))
        # and count the number of modfy marks.
        modifyStrCont = '## modifiedContinue$'

        mc = 0
        with open(f2, 'r') as f:
            for line in f:
                if re.search(modifyStrCont, line):
                    mc += 1

        # expect a further  changes -- CRUN. Reference case has STEP = 1
        expect = 1  # CRUN-> NRUN as case started with STEP=4
        self.assertEqual(mc, expect, msg=f'Expected {expect}  {modifyStrCont} got {mc}')

    def test_createWorkDir(self):
        """
        Test that createWorkDir worked as expected
        :return: 
        """
        self.model.createWorkDir()

        self.assertTrue((self.model.model_dir / 'W').is_dir())

    def test_fixClimFCG(self):
        """
        Test that fixClimFCG works as expects . TODO remove this. I think only needed with "bad" old code.
         
         converting all CLIM_FCG_YEARS(1,..) to CLIM_FCG_YEARS(:,...)
        :return: 
        """

        # test is that have NO lines in CNTLALL that look like (1,\w[0-9])
        # need to copy data over.
        shutil.copy2(self.refDir / 'CNTLATM', self.model.model_dir)
        self.model.fixClimFCG()
        file = self.model.model_dir / 'CNTLATM'
        with open(file, 'r') as f:
            for line in f:
                if re.search(r'\(1\w*,\w*[0-9]*\w\)=\w*[0-9]*,', line): self.fail("Found bad namelist %s" % line)

    def test_set_time_code(self):
        # test that set_time_code works
        model = self.model
        model.instantiate() # actually create the model,
        # two checks. If nothing requests then nothing changes.
        # runTime -- two changes (With runTime value as expected
        # runCode -- one change (with runCode as expected)
        file = model.submit_script
        pth = model.model_dir/file
        bak_path = model.model_dir/'SUBMIT_backup'
        shutil.copy2(pth,bak_path)
        time = pth.stat().st_mtime
        model.set_time_code(model.model_dir/"SUBMIT", None, None)
        time2 = pth.stat().st_mtime
        self.assertEqual(time,time2)
        # now change runTime
        model.set_time_code(model.model_dir/"SUBMIT",runTime=3000)
        time2 = pth.stat().st_mtime
        self.assertNotEqual(time, time2) # times should be different.
        modifyStr = r'## modified time/code\s*$'
        count = 0
        with open(pth, 'r') as f:
            for line in f:
                line = line[0:-1]  # strip newline
                if re.search('[NC]RUN_TIME_LIMIT.*'+modifyStr, line):
                    count += 1

        self.assertEqual(count,2) # should be two cases
        shutil.copy2(bak_path,pth) # copy nbackup back in!
        # runCode set
        time = pth.stat().st_mtime
        model.set_time_code("SUBMIT", runCode="someRunCode")
        time2 = pth.stat().st_mtime
        self.assertNotEqual(time, time2)  # times should be different.
        count = 0
        with open(pth, 'r') as f:
            for line in f:
                line = line[0:-1]  # strip newline
                if re.search(r'ACCOUNT\w*=\w*someRunCode.*' + modifyStr, line):
                    count += 1

        self.assertEqual(count, 1)  # should be one case
        shutil.copy2(bak_path,pth) # copy backup back in!
        # both set
        time = pth.stat().st_mtime
        model.set_time_code("SUBMIT", runTime=3000,runCode="someRunCode")
        time2 = pth.stat().st_mtime
        self.assertNotEqual(time, time2)  # times should be different.
        count = 0
        with open(pth, 'r') as f:
            for line in f:
                line = line[0:-1]  # strip newline
                if re.search(r'ACCOUNT\w*=\w*someRunCode.*' + modifyStr, line):
                    count += 1
                elif re.search(r'[NC]RUN_TIME_LIMIT.*' + modifyStr, line):
                    count += 1
                else:
                    pass

        self.assertEqual(count, 3)  # should be three  cases


    def test_perturb(self):
        """
        Test that perturbation works.
          Need to look at namelists so a bit tricky...
        :return:
        """
        import copy
        self.model.instantiate()
        params = copy.deepcopy(self.model.parameters)
        for pcount,param in enumerate(['VF1', 'ICE_SIZE', 'ENTCOEF', 'CT', 'ASYM_LAMBDA', 'CHARNOCK']): # params to be perturbed
            self.model.status='FAILED'
            self.model.perturb()
            # get the namelist values back in
            values = self.model.read_values(param)
            expect = {param:params[param]*(1+1e-6)}
            self.assertEqual(values, expect)
            self.assertEqual(self.model.perturb_count,pcount+1)


    def test_createPostProcessFile(self):
        """
        Test that creation of post processing script works.

        :return:
        """
        release_cmd = 'set_model_state SUCCEEDED'
        file = self.model.createPostProcessFile(release_cmd)
        # expect file to exist
        self.assertTrue(file.exists())
        # and that it is as expected.
        self.assertEqual(file, self.model.model_dir / self.model.post_process_file)
        # need to check its contents...
        # will check two things. 1) that SUBCONT is as expected 2) that the qrls cmd is in the file
        submit_script = self.model.continue_script
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
            # now reverse
            got = HadCM3.parse_isoduration(e)
            self.assertEqual(got, s)

        # should fail without a leading P or not a string
        for fail_case in ['12Y', '1Y', '3MT2S']:
            with self.assertRaises(ValueError):
                got = HadCM3.parse_isoduration(fail_case)
            #

    def test_startTime(self):
        """
        Tests startTime to see if it works
        :return:
        """
        model = HadCM3('aa001',self.refDir)
        tests = [ '2020-01-01', '1990-01-01', '1999-09-09 01:01:01']
        expect = [[2020, 1, 1, 0, 0, 0], [1990, 1, 1, 0, 0, 0], [1999, 9, 9, 1, 1, 1]]

        for t, e in zip(tests, expect):  # loop over tests
            lst = model.start_time(t)  #
            self.assertEqual(len(lst),2) # expect 2 namelist value pairs.
            self.assertEqual(e, lst[0][1], msg=f"Failed for {e} in {lst[0]}")

    def test_timeDelta_fns(self):
        """
        Tests timeDelta, runTarget & resubInterval  to check they all work

        :return: nada
        """
        model = HadCM3('aa001',self.refDir)
        tests = [  'P7Y', 'P7Y1M', 'P7Y1M1D', 'P7Y1M1DT1H1M1S']
        expect = [[7, 0, 0, 0, 0, 0], [7, 1, 0, 0, 0, 0], [7, 1, 1, 0, 0, 0], [7, 1, 1, 1, 1, 1]]

        for t, e in zip(tests, expect):  # loop over tests
            for fn in [model.time_delta, model.resub_interval, model.run_target]:
                lst = fn(t)
                self.assertEqual(lst[0][1],e)




if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  ## actually run the test cases



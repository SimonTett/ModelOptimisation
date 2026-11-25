# test cases for engine
import logging
import platform
import unittest
import pathlib
import subprocess
from time import sleep
import tempfile
import os
import engine
import genericLib
from Models import Model
import socket
from unittest.mock import patch

genericLib.setup_env()

my_engine = engine.abstractEngine.guess_engine() # engine we want to use
args=dict()
# test if we are on computers where we need to set run_code from env var PROJECT_CODE.
fqdn = socket.getfqdn()
# need run_code set for some machines.
if any([fqdn.endswith(x) for x in ['cirrus.ac.uk']]):
    #Need a run_code for this machine. It should be set in the environment variable PROJECT_CODE
    run_code = os.environ.get('PROJECT_CODE')
    if run_code is None:
        raise ValueError(f"On {fqdn} but PROJECT_CODE env var not set")
    args.update(
        run_code=run_code, # project code
    )

class TestEngine(unittest.TestCase):
    # tests for engines
    # one time setup


    def setUp(self) -> None:

        self.sge_engine = engine.sge_engine()
        self.slurm_engine = engine.slurm_engine()


        self.engine = my_engine
        self.args=args

    def test_expect_instance(self):
        """ Very generic tests. Just checks get expected type.
        But at least runs each method """
        os.environ['JOB_ID']='123456'
        os.environ['SLURM_JOB_ID']='123456'
        for eng in [self.sge_engine, self.slurm_engine]:
            self.assertIsInstance(eng.submit_cmd(['ls'], 'fred'), list)
            self.assertIsInstance(eng.release_job('45645'), list)
            self.assertIsInstance(eng.kill_job('45645'), list)
            if eng == self.slurm_engine:
                self.assertIsInstance(eng.job_id('Submitted job xx 123456'), str)
            elif eng == self.sge_engine:
                self.assertIsInstance(eng.job_id('Submitted job  123456'), str)
            self.assertIsInstance(eng.my_job_id(),str)

    def test_my_job_id(self):
        # test my_job_id
        vars = ['JOB_ID','SLURM_JOB_ID']
        for v in vars:
            os.environ[v]= '123456'

        for eng in [self.sge_engine, self.slurm_engine]:
            self.assertEqual(eng.my_job_id(), "123456")
        # remove env vars. Should get None
        for v in vars:
            del(os.environ[v])
        for eng in [self.sge_engine, self.slurm_engine]:
            with self.assertRaises(ValueError):
                eng.my_job_id()

    def test_run_cmds(self):
        # test commands work. Needs to be done on a system basis. 
        # Only runs on system where engine found.
        # runs a simple network of 4 jobs none of which do much:
        # job1 -> job2, job4
        # job2, job1 -> job3
        #
        logging.basicConfig(force=True,level=logging.DEBUG)

        if self.engine is None:
            logging.warning("No engine defined. Skipping test_run_cmds")
            return
        log_pth = pathlib.Path(os.environ['OPTCLIMTOP'])/'tmp_test_engine' # where we are going to put log files
        print(f'log_pth is {log_pth}')
        log_pth.mkdir(exist_ok=True,parents=True) 
        for file in log_pth.glob("*"):
            file.unlink()
        tdir = tempfile.TemporaryDirectory()
        dpth = pathlib.Path(tdir.name)
        # need to make a script file.
        cmd_pth = dpth / 'script.sh'
        with open(cmd_pth, 'wt') as fp:
            fp.write("#!/usr/bin/env bash \n")
            fp.write('echo "The date is "$(date)\n')

        cmd_pth.chmod(0o755)


        # will submit 4 jobs. 3 held and then submit a release job which releases the first  job.
        cmd1 = self.engine.submit_cmd([str(cmd_pth)], 'datejob', outdir=log_pth,
                                      time=10, mem=500, hold=True,**self.args)
        print(" ".join(cmd1))
        output = subprocess.check_output(cmd1, text=True)
        jid_1 = self.engine.job_id(output)
        print("job 1 is", jid_1)
        cmd2 = self.engine.submit_cmd([str(cmd_pth)], 'datejob2', outdir=log_pth,
                                      time=10, mem=500, hold=jid_1,**self.args)
        output2 = subprocess.check_output(cmd2, text=True)
        jid_2 = self.engine.job_id(output2)  # job 2 is held and will run once job 1 runs
        print("job 2 is", jid_2)
        # job 3 depends on job 1 & 2
        cmd3 = self.engine.submit_cmd([str(cmd_pth)], 'datejob3', outdir=log_pth,
                                      time=10, mem=500, hold=[jid_1, jid_2],**self.args)
        output3 = subprocess.check_output(cmd3, text=True)
        jid_3 = self.engine.job_id(output3)  #
        print("job 3 is", jid_3)

        # job 4 just depends on job 1.  -- it should run before job #3
        cmd4 = self.engine.submit_cmd([str(cmd_pth)], 'datejob4', outdir=log_pth,
                                      time=10, mem=500, hold=jid_1,**self.args)
        output4 = subprocess.check_output(cmd4, text=True)
        jid_4 = self.engine.job_id(output4)
        print("job 4 is", jid_4)

        # check have 4jobs as held.
        jobs = [jid_1, jid_2, jid_3, jid_4]
        for job_id in jobs:
            status = self.engine.job_status(job_id)
            print(job_id, status)
            self.assertEqual(status, 'Held')
        # now release the first job -- that should trigger all the rest. We will then sleep with times doubling.
        sleep_time = 0.1
        release_cmd = self.engine.release_job(jid_1)
        subprocess.check_output(release_cmd, text=True)
        all_done = True
        while sleep_time < 240:  # sleep up to 240 seconds or all jobs done
            all_done = True
            for job_id in jobs:
                status = self.engine.job_status(job_id)
                print(job_id, status, end=" --  ")
                all_done = all_done and (status.lower() == 'notfound')
            if all_done:
                print("All jobs ran")
                break

            print(f"Sleeping for {sleep_time:.1f} seconds")
            sleep(sleep_time)
            sleep_time *= 1.25  # increase time by 25%.

        self.assertTrue(all_done)  # we should have completed.
        # lets see what is in log_pth
        files = list(log_pth.glob("*"))
        for file in files:
            print(file)
            file.unlink() # remove file
        log_pth.rmdir()
        # should be 8 files in all -- 4 * stderr + 4 * stdout
        self.assertEqual(len(files), 8)

    def test_dump_load(self):
        """Test that dumping and loading works."""

        def dump_load(eng):
            eng.ssh_node = 'login03.ecdf.ed.ac'  # check have ssh_node set.
            with tempfile.NamedTemporaryFile(suffix='.cfg', delete=False) as tfile:
                tfile.close()
                tf = pathlib.Path(tfile.name)
                eng.dump(tf)
                new = eng.load(tf)
                self.assertEqual(new, eng)

        dump_load(self.sge_engine)  # test works for SGE engine
        dump_load(self.slurm_engine)  # test works for SLURM engine

    def test_create_engine(self):
        """
        Test that can create an engine via name and node.
        :return:
        """
        eng = engine.abstractEngine.create_engine('SGE', ssh_node='login.supercomputer.edu')
        self.assertIsInstance(eng, engine.sge_engine)

        eng = engine.abstractEngine.create_engine('SLURM', ssh_node='login.supercomputer.edu')
        self.assertIsInstance(eng, engine.slurm_engine)

    def test_connect_fn(self):
        """
        Test that connect_fn returns expected command.
        Some AI written code.
        """
        eng = engine.abstractEngine()
        result = eng.connect_fn(['ls'])
        self.assertEqual(result,['ls'])
        eng = engine.abstractEngine(ssh_node='ssh_node')
        with patch('pathlib.Path.cwd', return_value=pathlib.Path('/current/dir')):
            result = eng.connect_fn(['ls','-ltr','fred'])
        expected = ['ssh','-q','-o','batchmode=yes','-o','StrictHostKeyChecking=yes','ssh_node']
        self.assertEqual(result[0:len(expected)],expected)
        cmd = result[-1][1:-1].split('; ') # extract last element, remove 1st and last char and split in
        self.assertEqual(cmd[0],"cd /current/dir")
        # check have right env vars
        self.assertEqual('export PYTHONPATH=$PYTHONPATH', cmd[1])
        self.assertEqual('export PATH=$PATH', cmd[2])
        self.assertEqual('export OPTCLIMTOP=$OPTCLIMTOP', cmd[3])
        self.assertEqual('ls -ltr fred',cmd[-1])

        result = eng.connect_fn(['ls'], rundir=pathlib.Path('/tmp/test'))
        self.assertTrue(result[-1].startswith("'cd /tmp/test"))

    def test_submit_cmd(self):
        """
        Test that submit_cmd returns expected command.
        """


        expected = dict(SLURM = ['sbatch', '--cpus-per-task=1', '--time=0:00:10','--output', '/tmp/%x_%A_%a.out',
         '--error',  '/tmp/%x_%A_%a.err',  '-J','jobname','--ntasks=1','--export=ALL','--mem=500','-H',
         'ls'],
                        SGE = ['qsub', '-l','h_vmem=500M','-l','h_rt=10', '-V', '-e', '/tmp/', '-o', '/tmp/',
                        '-N', 'jobname', '-cwd', '-h', 'ls'])
        for key,expected_list in expected.items():
            eng = engine.abstractEngine.create_engine(key)
            result = eng.submit_cmd(['ls'], 'jobname', outdir=pathlib.Path('/tmp'), time=10, mem=500, hold=True)
            self.assertEqual(result,expected_list)
            # check that ssh node works
            eng = engine.abstractEngine.create_engine(key, ssh_node='login.supercomputer.ac.uk')
            result = eng.submit_cmd(['ls'], 'jobname', outdir=pathlib.Path('/tmp'), time=10, mem=500, hold=True)
            expect = eng.connect_fn(expected_list)
            self.assertEqual(result,expect)

    def test_extract_job_submission_params(self):
        # AI-generated using Qwen, reviewed by Simon Tett
        # Test extraction with various combinations of params and defaults
        # Test 1: All params present in run_info
        run_info = {
            'runTime': 3600,
            'runCode': 'n02-test',
            'runQueue': 'standard',
            'qos': 'slow'
        }
        result = engine.abstractEngine.extract_job_submission_params(run_info)
        self.assertEqual(result['time'], 3600)
        self.assertEqual(result['run_code'], 'n02-test')
        self.assertEqual(result['run_queue'], 'standard')
        self.assertEqual(result['qos'], 'slow')

        # Test 2: Empty run_info with defaults provided
        run_info = {}
        default_values = dict(runTime=1800, runCode='default_code')
        result = engine.abstractEngine.extract_job_submission_params(run_info, default_values)
        self.assertEqual(result['time'], 1800)
        self.assertEqual(result['run_code'], 'default_code')
        self.assertIsNone(result['run_queue'])
        self.assertIsNone(result['qos'])

        # Test 3: Partial override - run_info values override defaults
        run_info = {'runTime': 7200}
        default_values = dict(runTime=1800, runCode='n02-default')
        result = engine.abstractEngine.extract_job_submission_params(run_info, default_values)
        self.assertEqual(result['time'], 7200)
        self.assertEqual(result['run_code'], 'n02-default')

        # Test 4: No defaults provided (None)
        run_info = {'runTime': 900}
        result = engine.abstractEngine.extract_job_submission_params(run_info)
        self.assertEqual(result['time'], 900)
        self.assertIsNone(result['run_code'])
        self.assertIsNone(result['run_queue'])

    def test_slurm_job_status_parsing(self):
        # AI-generated using Qwen, reviewed by Simon Tett
        # Test SLURM job_status parsing for all status codes
        # squeue --long --noheader format: JOBID PARTITION NAME USER STATE TIME TIMELIMIT NODES REASON
        # (5th field is STATE, reason in parentheses for PENDING jobs)
        test_cases = [
            # (mock_output, expected_status)
            ("123456 debug jobname user COMPLETED 0:00 10:00 1 node001", 'Finished'),
            ("123456 debug jobname user RUNNING 0:00 10:00 1 node001", 'Running'),
            ("123456 debug jobname user CONFIGURING 0:00 10:00 1 node001", 'Running'),
            ("123456 debug jobname user COMPLETING 0:00 10:00 1 node001", 'Running'),
            ("123456 debug jobname user PENDING 0:00 10:00 1 (JobHeldUser)", 'Held'),
            ("123456 debug jobname user PENDING 0:00 10:00 1 (JobHeldAdmin)", 'Held'),
            ("123456 debug jobname user PENDING 0:00 10:00 1 (Dependency)", 'Held'),
            ("123456 debug jobname user PENDING 0:00 10:00 1 (Resources)", 'Queueing'),
            ("123456 debug jobname user FAILED 0:00 10:00 1 node001", 'Failed'),
            ("123456 debug jobname user CANCELLED 0:00 10:00 1 node001", 'Failed'),
            ("123456 debug jobname user TIMEOUT 0:00 10:00 1 node001", 'Failed'),
            ("123456 debug jobname user NODE_FAIL 0:00 10:00 1 node001", 'Failed'),
            ("", 'notFound'),
        ]

        for mock_output, expected_status in test_cases:
            with patch('subprocess.check_output', return_value=mock_output):
                status = self.slurm_engine.job_status('123456')
                self.assertEqual(status, expected_status,
                    f"Failed for output: {mock_output[:50]}...")

        # Test full_output=True returns raw output
        mock_output = "Full raw output from squeue --long"
        with patch('subprocess.check_output', return_value=mock_output):
            status = self.slurm_engine.job_status('123456', full_output=True)
            self.assertEqual(status, mock_output)

    def test_sge_job_status_parsing(self):
        # AI-generated using Qwen, reviewed by Simon Tett
        # Test SGE job_status parsing for all status codes
        # Grid Engine qstat format: job_id priority job_name user status time queue
        # (5th field is job_state)
        test_cases = [
            # (stdout, returncode, expected_status)
            ('123456 0.00000 jobname user r 00:00:00 queue@host', 0, 'Running'),
            ('123456 0.00000 jobname user E 00:00:00 queue@host', 0, 'Error'),
            ('123456 0.00000 jobname user hqw 00:00:00 queue@host', 0, 'Held'),
            ('123456 0.00000 jobname user qw 00:00:00 queue@host', 0, 'Queuing'),
            ('123456 0.00000 jobname user S 00:00:00 queue@host', 0, 'Suspended'),
            ('123456 0.00000 jobname user s 00:00:00 queue@host', 0, 'Suspended'),
        ]

        for stdout, returncode, expected_status in test_cases:
            mock_result = type('MockResult', (), {
                'returncode': returncode,
                'stdout': stdout,
                'check_returncode': lambda self: None
            })()
            with patch('subprocess.run', return_value=mock_result):
                status = self.sge_engine.job_status('123456')
                self.assertEqual(status, expected_status,
                    f"Failed for stdout: {stdout}")

        # Test job not found (returncode=1)
        mock_result = type('MockResult', (), {
            'returncode': 1,
            'stdout': '',
            'stderr': 'followed job does not exist'
        })()
        with patch('subprocess.run', return_value=mock_result):
            status = self.sge_engine.job_status('123456')
            self.assertEqual(status, 'notFound')

    def test_detect_scheduler(self):
        # AI-generated using Qwen, reviewed by Simon Tett
        # Test scheduler detection via env vars, FQDN, and edge cases
        # Test 1: Environment variable detection
        env_test_cases = [
            ({'SLURM_JOB_ID': '123456'}, 'SLURM'),
            ({'SLURM_STEP_ID': '123456'}, 'SLURM'),
            ({'PBS_JOBID': '123456.pbs'}, 'PBS'),
            ({'PBS_JOB_ID': '123456'}, 'PBS'),
            ({'PBS_O_QUEUE': 'normal'}, 'PBS'),
            ({'SGE_ROOT': '/opt/sge', 'QUEUE': 'all.q'}, 'SGE'),
            ({'JOB_ID': '123456', 'QUEUE': 'all.q.SGE'}, 'SGE'),
            ({'LSB_JOBID': '123456'}, 'LSF'),
        ]

        for env_vars, expected in env_test_cases:
            with patch.dict(os.environ, env_vars, clear=False),patch('socket.getfqdn', return_value='not_special.ed.ac.uk') :
                result = engine.abstractEngine.detect_scheduler()
                self.assertEqual(result, expected,
                    f"Failed for env_vars: {env_vars}")

        # Test 2: FQDN detection (Cirrus)
        with patch('socket.getfqdn', return_value='login0.cirrus.ac.uk'):
            result = engine.abstractEngine.detect_scheduler()
            self.assertEqual(result, 'SLURM_CIRRUS')

        # Test 3: Nothing found - returns None
        env_to_remove = ['SLURM_JOB_ID', 'SLURM_STEP_ID', 'PBS_JOBID',
                        'PBS_JOB_ID', 'PBS_O_QUEUE', 'SGE_ROOT', 'JOB_ID',
                        'QUEUE', 'LSB_JOBID']

        with patch.dict(os.environ, {}, clear=False):
            for var in env_to_remove:
                if var in os.environ:
                    del os.environ[var]

            with patch('socket.getfqdn', return_value='localhost'):
                with patch('shutil.which', return_value=None):
                    result = engine.abstractEngine.detect_scheduler()
                    self.assertIsNone(result)

        # Test 4: Env var priority over binary detection
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123456'}), patch('socket.getfqdn', return_value='localhost'):
            with patch('shutil.which', return_value='/usr/bin/qsub'):
                result = engine.abstractEngine.detect_scheduler()
                self.assertEqual(result, 'SLURM')

class TestCirrus_Engine(unittest.TestCase):
        # tests for cirrus slurm engine
        # one time setup

        def setUp(self) -> None:

            self.engine = engine.slurm_cirrus_engine()
            self.args = args

        def test_submit_cmd(self):
            """
            Test cirrus version
            """
            # must provide a run code. If not should get ValueError

            with self.assertRaises(ValueError):
                cmd = self.engine.submit_cmd(['ls'],name='testjob')
            # then run. Should get "standard" for qos and partition.
            got = self.engine.submit_cmd(['ls'],time=10,outdir=pathlib.Path('/tmp'),name='testjob',run_code='n02-notallowed')
            expected = ['sbatch', '--cpus-per-task=1', '--time=0:00:10','--output', '/tmp/%x_%A_%a.out',
            '--error',  '/tmp/%x_%A_%a.err',  '-J','testjob','--ntasks=1','--export=ALL',
                        "--qos=standard","-A","n02-notallowed","--partition=standard",
            'ls']
            self.assertEqual(got,expected)
            # now call memory set. Should get same and a warning.
            with self.assertWarns(UserWarning):
                got = self.engine.submit_cmd(['ls'], time=10, outdir=pathlib.Path('/tmp'), mem=1000,name='testjob',
                                             run_code='n02-notallowed')

            self.assertEqual(got,expected)
if __name__ == '__main__':
    unittest.main()

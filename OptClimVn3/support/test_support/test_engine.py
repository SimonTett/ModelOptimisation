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

genericLib.setup_env()

my_engine = engine.abstractEngine.guess_engine() # engine we want to use
args=dict()
# test if we are on Archer by calling hostname -A and that stdout contains archer2.ac.uk
stat = subprocess.run(['hostname','-A'],capture_output=True,text=True)
if (stat.returncode == 0) and 'archer2.ac.uk' in stat.stdout:
    # Specials needed for archer
    print("On archer 2")
    run_code = os.environ.get('PROJECT_CODE')
    if run_code is None:
        raise ValueError("On Archer2 but PROJECT_CODE env var not set")
    args.update(
        run_queue='serial', # running in the serial q
        run_code=run_code, # project code
        extra_args=['--qos=serial'] # and need to set qos
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
        breakpoint()
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
        eng = engine.abstractEngine.create_engine('SGE',ssh_node='ssh_node')
        result = eng.connect_fn([])
        expected = ['ssh','-q','-o','batchmode=yes','-o','StrictHostKeyChecking=yes']
        self.assertEqual(result[0:len(expected)],expected)



if __name__ == '__main__':
    unittest.main()

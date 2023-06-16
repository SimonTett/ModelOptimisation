# test cases for engine
import logging
import platform
import unittest
import engine
import pathlib
import subprocess
from time import sleep


class MyTestCase(unittest.TestCase):
    # first tests for engines

    def setUp(self) -> None:

        self.sge_engine = engine.sge_engine()
        self.slurm_engine = engine.slurm_engine()

    def test_expect_instance(self):
        """ Very generic tests. Just checks get expected type.
        But at least runs each method """
        for eng in [self.sge_engine,self.slurm_engine]:
            self.assertIsInstance(eng.submit_cmd(['ls'], 'fred'), list)
            self.assertIsInstance(eng.release_job('45645'), list)
            self.assertIsInstance(eng.kill_job('45645'), list)
            self.assertIsInstance(eng.job_id('Submitted job 123456'), str)

    def test_run_cmds(self):
        # test commands work. Needs to be done on a system basis. Only runs on linux systems and setup for SGE with no connect fn.
        # runs a simple network of 4 jobs none of which do much:
        # job1 -> job2, job4
        # job2, job1 -> job3
        #
        if platform.system() != "linux":
            logging.warning(f"Skipping test as can only run on linux not on {platform.system()}. ")
            return

        connect_fn = None
        system_name = 'SGE' # change for the system you want to run on.
        eng = engine.setup_engine(engine_name=system_name,connect_fn=connect_fn)
        # will submit 4 jobs. 3 held and then submit a release job which releases the first  job.
        cmd1=eng.submit_cmd(['date'],'datejob', outdir=pathlib.Path('/tmp/user_testing'), time=10, mem=500, hold=True)
        output = subprocess.check_output(cmd1,text=True)
        jid_1 = eng.job_id(output)
        cmd2 = eng.submit_cmd(['echo','job 2 ran at ','$(date)'], 'datejob2', outdir=pathlib.Path('/tmp/user_testing'),
                              time=10, mem=500, hold=jid_1)
        output2 = subprocess.check_output(cmd2,text=True)
        jid_2 = eng.job_id(output2) # job 2 is held and will run once job 1 runs
        # job 3 depends on job 1 & 2
        cmd3 = eng.submit_cmd(['echo','job 3 ran at ','$(date)'], 'datejob3', outdir=pathlib.Path('/tmp/user_testing'),
                              time=10, mem=500, hold=[jid_1,jid_2])
        output3 = subprocess.check_output(cmd3, text=True)
        jid_3 = eng.job_id(output3)  #
        # job 4 just depends on job 1.  -- it should run before job #3
        cmd4 = eng.submit_cmd(['echo','job 3 ran at ','$(date)'], 'datejob4', outdir=pathlib.Path('/tmp/user_testing'),
                              time=10, mem=500, hold=jid_1)
        output4 = subprocess.check_output(cmd4, text=True)
        jid_4 = eng.job_id(output4)

        # check have 4jobs as held.
        jobs = [jid_1,jid_2,jid_3,jid_4]
        for job_id in jobs:
            status = eng.job_status(job_id)
            self.assertEqual(status,'Held')
        # now release the first job -- that should trigger all the rest. We will then sleep with times doubling.
        sleep_time = 0.1
        release_cmd = eng.release_job(jid_1)
        subprocess.check_output(release_cmd,text=True)
        all_done = False
        while (sleep_time < 20) or all_done: # sleep up to 20 seconds or all jobs done
            for job_id in jobs:
                all_done = (all_done) or (eng.job_status(job_id) == 'NotFound')
                if all_done:
                    break
                sleep(sleep_time)
                sleep_time *= 2 # double time.
        self.assertTrue(all_done) # we should have completed.






if __name__ == '__main__':
    unittest.main()

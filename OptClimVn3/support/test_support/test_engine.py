# test cases for engine
import logging
import platform
import unittest
import engine
import pathlib
import subprocess
from time import sleep
import tempfile

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
        # test commands work. Needs to be done on a system basis. 
        # Only runs on linux systems and setup for SGE with no connect fn.
        # runs a simple network of 4 jobs none of which do much:
        # job1 -> job2, job4
        # job2, job1 -> job3
        #
        #logging.basicConfig(force=True,level=logging.DEBUG)
        if platform.system() != "Linux":
            logging.warning(f"Skipping test as can only run on linux not on {platform.system()}. ")
            return
        # hardwired logging dir as /tmp/ does not aappear to be accessible 
        # from sge on eddie. 
        log_pth=pathlib.Path("/exports/csce/eddie/geos/groups/OPTCLIM/tmp_test_engine/")
        for file in log_pth.glob("*"):
            file.unlink()
        tdir = tempfile.TemporaryDirectory()
        dpth = pathlib.Path(tdir.name)
        # need to make a script file.
        cmd_pth = dpth/'script.sh'
        with open(cmd_pth,'wt') as fp:
            fp.write("#!/bin/env bash \n")
            fp.write('echo "The date is "$(date)\n')

        cmd_pth.chmod(0o755)
        connect_fn = None
        system_name = 'SGE' # change for the system you want to run on.
        eng = engine.setup_engine(engine_name=system_name,connect_fn=connect_fn)
        # will submit 4 jobs. 3 held and then submit a release job which releases the first  job.
        cmd1=eng.submit_cmd([str(cmd_pth)],'datejob', outdir=log_pth,
                            time=10, mem=500, hold=True)
        print(" ".join(cmd1))
        output = subprocess.check_output(cmd1,text=True)
        jid_1 = eng.job_id(output)
        print("job 1 is",jid_1)
        cmd2 = eng.submit_cmd([str(cmd_pth)], 'datejob2', outdir=log_pth,
                              time=10, mem=500, hold=jid_1)
        output2 = subprocess.check_output(cmd2,text=True)
        jid_2 = eng.job_id(output2) # job 2 is held and will run once job 1 runs
        print("job 2 is",jid_2)
        # job 3 depends on job 1 & 2
        cmd3 = eng.submit_cmd([str(cmd_pth)], 'datejob3', outdir=log_pth,
                              time=10, mem=500, hold=[jid_1,jid_2])
        output3 = subprocess.check_output(cmd3, text=True)
        jid_3 = eng.job_id(output3)  #
        print("job 3 is",jid_3)

        # job 4 just depends on job 1.  -- it should run before job #3
        cmd4 = eng.submit_cmd([str(cmd_pth)], 'datejob4', outdir=log_pth,
                              time=10, mem=500, hold=jid_1)
        output4 = subprocess.check_output(cmd4, text=True)
        jid_4 = eng.job_id(output4)
        print("job 4 is",jid_4)

        # check have 4jobs as held.
        jobs = [jid_1,jid_2,jid_3,jid_4]
        for job_id in jobs:
            status = eng.job_status(job_id)
            print(job_id,status)
            self.assertEqual(status,'Held')
        # now release the first job -- that should trigger all the rest. We will then sleep with times doubling.
        sleep_time = 0.1
        release_cmd = eng.release_job(jid_1)
        subprocess.check_output(release_cmd,text=True)
        while (sleep_time < 120): # sleep up to 120 seconds or all jobs done
            all_done = True
            for job_id in jobs:
                status = eng.job_status(job_id)
                print(job_id,status,end=" --  ")
                all_done = (all_done) and ( status == 'notFound')
            if all_done:
                print("All jobs ran")
                break

            print(f"Sleeping for {sleep_time}") 
            sleep(sleep_time)
            sleep_time *= 2 # double time.
        self.assertTrue(all_done) # we should have completed.
        # lets see what is in log_pth
        files = list(log_pth.glob("*"))
        for file in  files:
            print(file)
        # should be 8 files in all -- 4 * stderr + 4 * stdout
        self.assertEqual(len(files),8)
        






if __name__ == '__main__':
    unittest.main()
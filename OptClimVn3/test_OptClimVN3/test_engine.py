# test cases for engine
import unittest
import engine
import pathlib


class MyTestCase(unittest.TestCase):
    # first tests for engines

    def setUp(self) -> None:

        self.sge_engine = engine.sge_engine()
        self.slurm_engine = engine.slurm_engine()

    def test_expect_instance(self):
        """ Very generic tests. Just checks get expected type.
        But at least runs each method """
        for eng in [self.sge_engine,self.slurm_engine]:
            self.assertIsInstance(eng.submit_fn(['ls'], 'fred'), list)
            self.assertIsInstance(eng.release_fn('45645'), list)
            self.assertIsInstance(eng.kill_fn('45645'), list)
            self.assertIsInstance(eng.jid_fn('Submitted job 123456'),str)




if __name__ == '__main__':
    unittest.main()

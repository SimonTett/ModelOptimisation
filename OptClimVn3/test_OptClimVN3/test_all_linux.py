# tests when we are on a linux system where the subprocess.check_output can sensibly do something.
# Tests are needed for Model and SubmitStudy

import unittest
import platform
import Model
import SubmitStudy


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # do what ever setup is needed...
        if platform.system() != 'Linux':
            raise NotImplementedError(
                f"Need to be on linux system to run models and other scripts. You are on {platform.system()}")

        raise NotImplementedError(f"Implement Linux tests.")

    def test_Model(self):
        # test Model functionality when can run Model properly.
        raise NotImplementedError("Implement tests for Model")

    def test_SubmitStudy(self):
        # test SubmitStudy functionality when can run SubmitStudy properly
        raise NotImplementedError("Implement tests for SubmitStudy")



if __name__ == '__main__':
    unittest.main()

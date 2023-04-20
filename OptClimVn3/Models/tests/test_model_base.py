import pathlib
import tempfile
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from model_base import model_base


class TestModelBase(unittest.TestCase):
    def assertAllequal(self,data1,data2):
        for key,value in data1.items():
            value2 =  data2[key]
            if isinstance(value,pd.Series):
                pdtest.assert_series_equal(value,value2)
            elif isinstance(value,pd.DataFrame):
                pdtest.assert_frame_equal(value,value2)
            elif isinstance(value,np.ndarray):
                nptest.assert_equal(value,value2)
            else:
                self.assertEqual(value,value2)

    def setUp(self):
        class dummy_model(model_base):
            def __init__(self,**kwargs):
                super().__init__()
                for k in ['p1', 'p2','p3','p4']:
                    setattr(self, k,None)
                for k, v in kwargs.items():
                    setattr(self, k, v)

        valid_dict = {"p1": 10, "p2": "test", "p3": True, 'p4': np.zeros(10)}
        self.model = dummy_model(**valid_dict)
        self.init_class=dummy_model

    def test_from_dict(self):
        # Test a valid dictionary with all expected keys
        valid_dict = {"p1": 22, "p2": "testa", "p3": False, 'p4': np.ones(10)}
        instance = self.init_class.from_dict(valid_dict)
        self.assertIsInstance(instance, self.init_class)
        d=instance.to_dict()
        self.assertAllequal(d,valid_dict)


        # Test a dictionary with an extra key that should not be set
        invalid_dict = { "param4": "extra"}
        instance = self.init_class.from_dict(invalid_dict)
        self.assertIsInstance(instance, self.init_class)

        self.assertFalse(hasattr(instance, "param4"))

    def test_to_dict(self):
        # Test a valid instance with some parameters
        instance = model_base()
        instance.param1 = 5
        instance.param2 = "test"
        instance.param3 = False
        expected_dict = {"param1": 5, "param2": "test", "param3": False}
        self.assertEqual(instance.to_dict(), expected_dict)

    def test_dump_load(self):
        """
        Test that dumping and loading works.
        :return:
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            print(type(tmpdir))
            dirname = pathlib.Path(tmpdir)
            file=dirname/'test/test.cfg'
            self.model.dump(file)
            # verify file exists, is a file and has size >0
            self.assertTrue(file.is_file())
            self.assertTrue(file.stat().st_size > 100)
            new = self.init_class.load(file)
            self.assertAllequal(vars(new),vars(self.model))







if __name__ == '__main__':
    unittest.main()

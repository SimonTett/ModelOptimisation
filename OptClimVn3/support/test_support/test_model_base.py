import copy
import pathlib
import subprocess
import tempfile
import unittest
import unittest.mock

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

from model_base import model_base
from model_base import journal
import datetime

def gen_time():
    # used to mock Model.now()
    time = datetime.datetime(2000, 1, 11, 0, 0, 0)
    timedelta = datetime.timedelta(seconds=1)
    while True:
        time += timedelta
        yield time

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
            dirname = pathlib.Path(tmpdir)
            file=dirname/'test/test.cfg'
            self.model.dump(file)
            # verify file exists, is a file and has size >0
            self.assertTrue(file.is_file())
            self.assertTrue(file.stat().st_size > 100)
            new = self.init_class.load(file)
            self.assertAllequal(vars(new),vars(self.model))

    def test_eq__(self):
        # test __eq__
        def fn_test(x):
            return x**2

        def fn_test2(x):
            return x**3

        m1 = model_base()
        m2 = model_base()
        for m in [m1,m2]:
            m.fred='fred'
            m.fn = fn_test
            m.dct = dict(fred=1,james=2)
            m.integer = 2
            m.pi = 3.14156
            m.lst=[1,2,3]

        for k,v in vars(m2).items() : # iterate over all items
            v_orig = copy.deepcopy(getattr(m2, k)) # copy whatever is there
            if isinstance(v,dict):
                v.update(somevalue=2)
            elif callable(v):
                v=fn_test2
            else:
                v *= 2 # modify it.

            setattr(m2,k,v) # set the value in the objust
            self.assertNotEqual(m1,m2) # should be different
            setattr(m2,k,v_orig) # put original value back
        self.assertEqual(m1,m2) # and should be equal

class Test_history(unittest.TestCase):

    def setUp(self):
        self.history = journal()

    @unittest.mock.patch.object(journal, 'now', side_effect=gen_time())
    def test_update_history(self,mck_now):
        """ Verify that update_history works.
        Two cases to consider
            1) Multiple updates in short time. Need to "mock" datetime.datetime.now(tz=datetime.timezone.utc)
            2) Updates at different times.

         """

        # NB gen_time() returns an interator and mock (magically) runs next on iterators.
        hist = self.history
        # first update times using mock.
        for count in range(0, 20):
            msg = f"Count is {count}"
            hist.update_history(msg)
        self.assertEqual(len(hist._history), 20)  #  20 count msgs
        #hist.print_history()

        test_now = datetime.datetime(1999, 12, 31, 23, 59, 59)
        with unittest.mock.patch.object(hist, 'now', autospec=True, return_value=test_now):
            hist._history = {}  # set history empty
            # always have the same time here.
            lst = []
            for count in range(0, 20):
                msg = f"Count is {count}"
                lst += [msg]
                hist.update_history(msg)
            self.assertEqual(len(hist._history), 1)
            k, v = hist._history.popitem()
            self.assertEqual(k, str(test_now))
            self.assertEqual(v, lst)

    @unittest.mock.patch.object(journal, 'now', side_effect=gen_time())
    def test_store_output(self,mck_now):
        hist = self.history
        # check we can store output.
        cmds=[['ls','*.nc'],['echo','once'],['mv','..']]
        results = ['fred.nc','once','..']
        for cnt,(cmd,result) in enumerate(zip(cmds,results)):
            hist.store_output(cmd,result)
            self.assertEqual(len(hist._output),cnt+1)
        for (k,lst),cmd,result  in zip(hist._output.items(),cmds,results):
            self.assertEqual(lst[0]['cmd'],cmd)
            self.assertEqual(lst[0]['result'],result)

        #hist.print_output()

    @unittest.mock.patch("subprocess.check_output",autospec=True)
    def test_run_cmd(self,mck_check):
        """
        Test we can run a cmd and output is as expected.

        mock subprocess.check_output so nothing actually ran.
        :return:
        """

        result = "ran a command"
        mck_check.return_value = result
        self.history.run_cmd(['ls'])
        mck_check.assert_called()
        time,got = self.history._output.popitem()
        self.assertEqual(got,[dict(cmd=['ls'],result=result)])
        # now test some error conds.
        mck_check.side_effect = FileNotFoundError
        with self.assertRaises(subprocess.SubprocessError):
            self.history.run_cmd(['ls'])

        mck_check.side_effect = subprocess.SubprocessError #
        with self.assertRaises(subprocess.SubprocessError):
            self.history.run_cmd(['ls'])






if __name__ == '__main__':
    unittest.main()

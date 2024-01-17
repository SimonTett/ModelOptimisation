"""
Unit tests generated (largely) by chatGPT3 2023-04-07.  Then fixed/modified by SFBT.
"""
import unittest
import json
import numpy as np
import pandas as pd
import pathlib
from generic_json import JSON_Encoder, load, loads, dump, dumps,obj_to_from_dict
import logging
import pandas.testing as pdtest
import numpy.testing as nptest
import tempfile
import namelist_var
import param_info
from model_base import model_base # so we can detect my own derived types.





class TestJsonEncoder(unittest.TestCase):


    def test_register_TO_VALUE(self):

        class Dummy:
            @classmethod
            def from_value(cls, value):
                pass
            def to_value(dct):
                pass

        obj_to_from_dict.register_TO_VALUE(Dummy, Dummy.to_value)
        self.assertEqual(obj_to_from_dict.TO_VALUE['Dummy'], Dummy.to_value)

    def test_register_FROM_VALUE(self):
        class Dummy:
            @classmethod
            def from_value(cls, value):
                pass

        obj_to_from_dict.register_FROM_VALUE(Dummy, Dummy.from_value)
        self.assertEqual(obj_to_from_dict.FROM_VALUE['Dummy'], Dummy.from_value)

    def test_value_to_obj(self):
        class Dummy:
            @classmethod
            def from_value(cls, values):
                return cls(*values)

            def __init__(self, *values):
                self.values = values

        obj_to_from_dict.register_FROM_VALUE(Dummy, Dummy.from_value)

        values = (1, 2, 3)
        decode = obj_to_from_dict()
        obj = decode.value_to_obj('Dummy', values)
        self.assertIsInstance(obj, Dummy)
        self.assertEqual(obj.values, values)

    def test_decode(self):
        """
        Test decode method
        :return:
        """

        test = {"__cls__name__": "ndarray", "object": [1, 2, 3]}
        expect=np.array([1, 2, 3])
        decode = obj_to_from_dict()
        got = decode.decode(test)
        nptest.assert_equal(expect, got)

        test = dict(harry=3.2, fred=2)
        got = decode.decode(test)
        nptest.assert_equal(test,got) # shou.d be identical

        # bad dict should raise a value error
        test = {"__cls__name__": "ndarray", "object": [1, 2, 3],"comment":'some comment'}
        with self.assertRaises(TypeError):
            got = decode.decode(test)



    def test_default(self):
        """
        Test default encoding method
        :return:
        """
        class Dummy:
            def __init__(self, *values):
                self.values = values
        class Dummy2:
            def __init__(self, *values):
                self.values = values
        obj_to_from_dict.register_TO_VALUE(Dummy, lambda x: list(x.values))
        e= JSON_Encoder()
        test=Dummy(1,2,43,66)
        got = e.default(test)
        expect=dict(__cls__name__="Dummy",__module__=test.__module__,object=[1,2,43,66])
        self.assertEqual(expect, got)  # should be identical
        # test that unknown class works
        with self.assertRaises(TypeError):
            test = Dummy2([1, 2, 43, 66])
            got = e.default(test)


    def test_obj_to_value(self):
        class Dummy:
            def __init__(self, *values):
                self.values = values

        obj_to_from_dict.register_TO_VALUE(Dummy, lambda x: list(x.values))

        obj = Dummy(1, 2, 3)
        result = obj_to_from_dict.obj_to_value(obj)
        self.assertEqual(result, [1, 2, 3])

    def test_encode_decode(self):
        class Dummy:
            def __init__(self, *values):
                self.values = values

        obj_to_from_dict.register_TO_VALUE(Dummy, lambda x: x.values)
        obj_to_from_dict.register_FROM_VALUE(Dummy, lambda lst: Dummy(*lst))

        obj = Dummy(1, 2, 3)

        # test encoding
        encoded = dumps(obj)
        expected = dict(__cls__name__="Dummy",object=[1, 2, 3],__module__=obj.__module__)
        s_expected=str(expected).replace("'",'"')
        self.assertEqual(encoded, s_expected)

        # test decoding
        decoded = loads(encoded)
        self.assertIsInstance(decoded, Dummy)
        self.assertEqual(decoded.values, (1, 2, 3))




class TestJsonUtils(unittest.TestCase):
    def setUp(self):
        nl= namelist_var.namelist_var(filepath=pathlib.Path('test_nl'), namelist='atmos', nl_var='fred')
        self.data = {'a': np.array([1, 2, 3]),
                     'b': pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}),
                     'c': pd.Series([1, 2, 3]),
                     'd': pathlib.Path('test.py'),
                     'e':dict(fred=True,james=2,harry=4.5,gordon='some text'),
                     'f':[1,2,3,'more test'],
                     'g':nl,
                     'h': param_info.param_info()}
        self.data['h'].register('VF1',nl)
    def assertAllequal(self,data1,data2):
        for key,value in data1.items():
            value2 =  data2[key]
            if isinstance(value,pd.Series):
                pdtest.assert_series_equal(value,value2)
            elif isinstance(value,pd.DataFrame):
                pdtest.assert_frame_equal(value,value2)
            elif isinstance(value,np.ndarray):
                nptest.assert_equal(value,value2)
            elif isinstance(value,model_base):
                self.assertEqual(vars(value), vars(value2))
            else:
                self.assertEqual(value,value2)


    def test_dump_load(self):
        decode = obj_to_from_dict()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file=f.name
            with open(file, 'w') as f:
                json.dump(self.data, f, cls=JSON_Encoder)

            with open(file, 'r') as f:
                loaded_data = json.load(f, object_hook=decode.decode)

            self.assertAllequal(loaded_data,self.data)
            with open(file, 'w') as f:
                dump(self.data, f)

            with open(file, 'r') as f:
                loaded_data = load(f)
            self.assertAllequal(loaded_data, self.data)


    def test_dumps_loads(self):
        decode = obj_to_from_dict()
        s=json.dumps(self.data, cls=JSON_Encoder)
        loaded_data = json.loads(s,object_hook=decode.decode)
        self.assertAllequal(loaded_data,self.data)
        s=dumps(self.data)
        loaded_data = loads(s)
        self.assertAllequal(loaded_data,self.data)

if __name__ == '__main__':
    unittest.main()

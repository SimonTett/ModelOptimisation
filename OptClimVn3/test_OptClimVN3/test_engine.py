# test cases for engine
import unittest
from engine import engine
import pathlib


class MyTestCase(unittest.TestCase):
    # first tests for engines
    def test_eng_setup_engine(self):
        # get some thing sensible..
        cmd = ["echo"], ['fred']  # cmd that will be ran
        jid = "203675"
        for name in ['SGE', 'SLURM']:
            eng = engine.setup_engine(name)
            # should be name and, except for engine_name, be functions
            for k, v in vars(eng).items():
                if k == "engine_name":
                    self.assertEqual(v, name)
                else:
                    self.assertTrue(callable(v))
                    # run the command -- should be a list but the arguments depend on what it is!
                    if k == "submit_fn":
                        result = v(cmd, 'tst_sub', pathlib.Path.cwd())
                        self.assertIsInstance(result, list)
                    elif k == "array_fn":
                        result = v(cmd, 'tst_sub', pathlib.Path.cwd(), 10)
                        self.assertIsInstance(result, list)
                    elif k in ['release_fn', 'kill_fn']:
                        result = v(jid)
                        self.assertIsInstance(result, list)
                    elif k in ['jid_fn']:
                        jid = "56745"
                        if name == 'SGE':
                            result = v("cmd name " + jid)
                            self.assertEqual(jid, result)
                        elif name == 'SLURM':
                            with self.assertRaises(NotImplementedError):
                                result = v("cmd name " + jid)
                        else:
                            raise NotImplementedError(f" implement test for {name} jid_fn")
                    else:
                        raise NotImplementedError(f"Do not know how to test {k}")

        # test unknown engin causes failure
        with self.assertRaises(ValueError):
           engine.setup_engine('fred')

    def test_eng_to_dict(self):
        # test that the dct we get makes sense. It should be the function *names* and engine_name
        eng = engine.setup_engine('SLURM')
        dct = eng.to_dict()
        for k, v in dct.items():
            if k == 'engine_name':
                self.assertEqual(v, eng.engine_name)
            else:
                eng_v = getattr(eng, k)
                self.assertIsInstance(v, str)
                self.assertTrue(callable(eng_v))
                self.assertEqual(v, eng_v.__name__)

    def test_eng_from_dict(self):
        # test that can convert a dict -- functions get generated from engine_name
        dct = dict(submit_fn='sge_submit_fn', array_fn='sge_array_fn',
                   release_fn='sge_release_fn', kill_fn='sge_kill_fn',
                   jid_fn='sge_jid_fn',
                   engine_name='SGE')
        eng = engine.from_dict(dct)
        self.assertEqual(eng.to_dict(), dct)
        # fail cases. See engine_name to SLURM. Should fail with the existing names.
        dct.update(engine_name='SLURM')
        with self.assertRaises(ValueError):
            eng = engine.from_dict(dct)
        # fix all names and should run.
        for key in dct.keys():
            dct[key] = dct[key].replace("sge", "slurm")
        eng = engine.from_dict(dct)
        self.assertEqual(eng.to_dict(), dct)


if __name__ == '__main__':
    unittest.main()

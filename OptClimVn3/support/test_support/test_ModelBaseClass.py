import pathlib
import unittest
from unittest.mock import patch, MagicMock

import Model
import genericLib
import namelist_var
from ModelBaseClass import ModelBaseClass
import logging


logging.basicConfig(level=logging.INFO, force=True)
import importlib
class TestModelBaseClass(unittest.TestCase):

    def setUp(self):
        """
        Setup for each test case.
        """
        self.model_base_class = ModelBaseClass()
        self.class_info = ModelBaseClass.class_registry
        self.model_base_class.remove_class(all_classes=True)
        importlib.invalidate_caches()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        """
        Teardown for each test case.
        """
        ModelBaseClass.class_registry = self.class_info 
    def test_register_functions(self):
        """
        Test cases:
        - Ensure all functions with the `_is_param` attribute are registered.
        - Verify that the `param_info` is correctly populated.
        """
        pass

    def test___init_subclass__(self):
        """
        Test cases:
        - Ensure `param_info` is correctly inherited and updated from base classes.
        - Verify that the class is registered in `class_registry`.
        """
        pass

    def test_register_class(self):
        """
        Test cases:
        - Ensure the class is added to `class_registry`.
        """
        from Model import Model
        cls = ModelBaseClass.register_class(Model)
        self.assertEqual(ModelBaseClass.class_registry['Model'].__name__,'Model')
        # test register works with specified name
        ModelBaseClass.register_class(newcls=cls,name='MODEL')
        self.assertEqual(ModelBaseClass.class_registry['MODEL'].__name__,'Model')
        # try loading from module.class
        cls = ModelBaseClass.register_class(name='HadCM3.HadCM3')
        self.assertEqual(ModelBaseClass.class_registry['HadCM3.HadCM3'].__name__,'HadCM3')


    def test_remove_class(self):
        """
        Test cases:
        - Ensure the class is removed from `class_registry`.
        - Verify behaviour when the class is not in the registry.
        """

        # add some classes then check removal works
        from GAMIL3_new import GAMIL3_new # so we can check instance
        # remove it!

        model_base=self.model_base_class
        model_base.register_class(name='simple_model.simple_model')
        model_base.register_class(name='GAMIL3_new.GAMIL3_new')
        # check removing a class works giving a class.
        cls = model_base.remove_class('GAMIL3_new.GAMIL3_new')
        self.assertEqual(cls.__name__,'GAMIL3_new')
        known_models = model_base.known_models()
        self.assertEqual(len(known_models),1)
        self.assertNotIn('GAMIL3_new.GAMIL3_new',known_models)


    def test_known_models(self):
        """
        Test cases:
        - Ensure the list of known models is returned correctly.
        """
        model_base_class = self.model_base_class
        model_base_class.remove_class(all_classes=True)
        self.assertEqual(model_base_class.known_models(),[])
        # import two models and check they are OK.
        model_base_class.register_class(name='Model.Model')
        model_base_class.register_class(name='HadCM3.HadCM3')
        self.assertEqual(['Model.Model','HadCM3.HadCM3'], model_base_class.known_models())




    def test_model_init(self):
        """
        Test cases:
        - Ensure the correct class is instantiated.
        - Verify behaviour when the class is specified by module.
        - Verify behaviour when the class is not found.
        """
        from Model import Model # FIXME. Ths is not generating a fresh import  as python caches modules.
        # sigh what to do!  Wil test by using the module.class import.
        root_dir = genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations')
        model = ModelBaseClass.model_init('Model.Model','fred',
                                          reference=root_dir/"example_Model/reference")
        self.assertIsInstance(model, Model)
        model2 = ModelBaseClass.model_init('HadCM3.HadCM3','dd001',
                                           reference=root_dir/"example_HadAM3/reference")
        from HadCM3 import HadCM3
        self.assertIsInstance(model2, HadCM3)
        # test for missing model.
        with self.assertRaises(ValueError):
            model = ModelBaseClass.model_init('Model_special','fred',
                                              reference=root_dir/"example_Model/reference")
        # and a not found module
        with self.assertRaises(ModuleNotFoundError):
            model = ModelBaseClass.model_init('Model_special.very_special','fred',
                                              reference=root_dir/"example_Model/reference")
        # missing function.
        with self.assertRaises(AttributeError):
            model = ModelBaseClass.model_init('Model.very_special','fred',
                                              reference=root_dir/"example_Model/reference")




    def test_add_param_info(self):
        """
        Test cases:
        - Ensure parameter information is added correctly.
        - Verify behaviour when duplicates are allowed or not allowed.
        """
        nl1=namelist_var.NamelistVar(type_name='json_nl',filepath=pathlib.Path('test.json'),namelist='model_params',nl_var='vf1')
        nl2 = namelist_var.NamelistVar(type_name='json_nl', filepath=pathlib.Path('test.json'), namelist='model_params',
                                       nl_var='vf2')
        nl3 = namelist_var.NamelistVar(type_name='json_nl', filepath=pathlib.Path('test.json'), namelist='model_params',
                                       nl_var='vf1a')
        vars_to_add=dict(VF1=nl1,VF2=nl2)
        self.model_base_class.add_param_info(vars_to_add,duplicate=False)
        self.assertEqual(len(self.model_base_class.param_info.param_constructors),2)
        for key,namelists in self.model_base_class.param_info.param_constructors.items():
            self.assertEqual(len(namelists),1)
        # add in a second namelist with same name. Should have a list of two.
        self.model_base_class.add_param_info(dict(VF1=nl3), duplicate=True)
        self.assertEqual(len(self.model_base_class.param_info.param_constructors['VF1']),2)
        # add in namelist with duplicate not set. Should go back to 1.
        self.model_base_class.add_param_info(dict(VF1=nl3), duplicate=False)
        self.assertEqual(len(self.model_base_class.param_info.param_constructors['VF1']),1)

    def test_update_from_file(self):
        """
        Test cases:
        - Ensure parameter information is updated from a CSV file.
        - Verify behavior when duplicates are allowed or not allowed.
        """
        pass



if __name__ == '__main__':
    unittest.main()
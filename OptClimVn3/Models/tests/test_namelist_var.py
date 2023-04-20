import logging
import os.path
import pathlib
import shutil
import tempfile
import unittest

import genericLib
from namelist_var import namelist_var


class namelist_var_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Setup for reads. Will have a model + bunch of namelists
        :return:
        """
        # copy reference case to tempdir.
        logging.basicConfig(format='%(asctime)s %(module)s.%(funcName)s %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)  # used throughout.
        refDir = namelist_var.expand('$OPTCLIMTOP/Configurations/xnmea')  # need a coupled model.
        simObsDir = 'test_in'
        self.dirPath = testDir
        self.refPath = refDir
        self.tmpDir = tmpDir  # really a way of keeping in context
        self.testDir = testDir

        shutil.rmtree(self.testDir, onerror=genericLib.errorRemoveReadonly)
        shutil.copytree(refDir,self.testDir) # copy everything over.
        nl_list=[]
        self.values=[1.0,10.0,
                      1e-4,
                     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     3.0]
        for file,nl,var,name in zip(['CNTLATM','CNTLATM','CNTLATM','CNTLATM','CNTLATM'],
                               ['SLBC21','RUNCNST','SLBC21','SLBC21','SLBC21'],
                               ['VF1','DTICE','CT','EACF','ENTCOEF'],
                                ['vf1','DTICE','ct','eacf','entcoef']):

            nl=namelist_var(filepath=pathlib.Path(file),namelist=nl,nl_var=var,name=name)
            nl_list.append(nl)
        self.namelist=nl_list
    def tearDown(self):
        shutil.rmtree(self.testDir, onerror=genericLib.errorRemoveReadonly)


    def test_file_cache(self):
        """
        Test file_cache. Read twice get the same result.
        :return:
        """
        nl=self.namelist[0]

        for nl in self.namelist:
            v1= namelist_var.file_cache(self.dirPath/nl.filepath,clean=True) # clear cache
            v2= namelist_var.file_cache(self.dirPath/nl.filepath) # read (hopefully using the cache)
            self.assertEqual(v1,v2) #values are the same



    def test_modify_namelists(self):
        """
        Test modify namelists
        :return:
        """
        file_dict = namelist_var.modify_namelists([], dirpath=self.dirPath)  # should be empty
        self.assertEqual(len(file_dict),0)
        # expect len of unique files
        file_dict = namelist_var.modify_namelists(zip(self.namelist, self.values), dirpath=self.dirPath)
        files = set([nl.filepath for nl in self.namelist])
        self.assertEqual(len(files),len(file_dict))
        for nl in self.namelist:
            self.assertEqual(file_dict[self.dirPath/nl.filepath][nl.namelist][nl.nl_var],nl.read_value(dirpath=self.dirPath))

        file_dict = namelist_var.modify_namelists(zip(self.namelist, [v * 2 for v in self.values]), dirpath=self.dirPath,
                                                  update=True)
        # now have set of namelists and values. Check they are as expected.

        for nl in self.namelist:
            self.assertEqual(file_dict[self.dirPath/nl.filepath][nl.namelist][nl.nl_var],nl.read_value(dirpath=self.dirPath)*2,msg=f"Failed for {nl}")

    def test_nl_patch(self):
        """
        Test namelist patching
        Patch the values. Should have expected bak files and values as expected
        :return:
        """
        values = []
        for v in self.values:
            if isinstance(v,list):
                lst=[vv+1 for vv in v]
                values.append(lst)
            else:
                values.append(v+1)
        nl_items =list(zip(self.namelist, values))
        patch = namelist_var.nl_patch(nl_items,dirpath=self.dirPath)
        self.assertTrue(patch) # worked
        # check namelists are as expected
        for nl,value in nl_items:
            got=nl.read_value(dirpath=self.dirPath)
            self.assertEqual(got,value)



        for nl,v,v2 in zip(self.namelist,self.values,values):
            pth = self.dirPath/nl.filepath
            bak = pth.parent/(pth.name+'.bak')      #check for backup files
            nl2=namelist_var(filepath=bak.relative_to(self.dirPath),namelist=nl.namelist,nl_var=nl.nl_var)
            self.assertTrue(bak.exists() and bak.is_file())
            self.assertEqual(nl2.read_value(dirpath=self.dirPath),v)
            self.assertEqual(nl.read_value(dirpath=self.dirPath),v2)



    def test_nl_read(self):
        """
        Test reading namelist.
        :return:
        """

        for nl,v in zip(self.namelist,self.values):
            got=nl.read_value(dirpath=self.dirPath)
            self.assertEqual(got,v,msg=f"Failed for {nl}")
        # try again cleaning cache each time. Should get same results
        for nl,v in zip(self.namelist,self.values):
            got=nl.read_value(dirpath=self.dirPath,clean=True)
            self.assertEqual(got,v,msg=f"Failed for {nl}")


    def test_nl_name(self):
        """ Test nl_Name is as expected"""
        nl = namelist_var(filepath=pathlib.Path('../test.nl'),namelist='BIG_NL',nl_var='small_var',name='TINY')
        self.assertEqual(nl.Name(),'TINY')
        nl = namelist_var(filepath=pathlib.Path('../test.nl'),namelist='BIG_NL',nl_var='small_var')
        self.assertEqual(nl.Name(),f'{str(nl.filepath)}&BIG_NL small_var')






if __name__ == '__main__':

    unittest.main()

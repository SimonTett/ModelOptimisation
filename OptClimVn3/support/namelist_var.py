import copy
import dataclasses
import logging
import pathlib
import shutil
import tempfile
import f90nml
import numpy as np

from model_base import model_base


@dataclasses.dataclass(frozen=True)
class namelist_var(model_base):
    """
    Class to handle namelist variables. Provides several **class** methods.
    """
    filepath: pathlib.Path
    namelist: str
    nl_var: str
    name: str = None
    default: any = None


    def __repr__(self):
        """ Representation -- the name and the cpts"""
        r = f"{self.filepath}&{self.namelist} {self.nl_var}"
        if self.default is not None:
            r += f" default:{self.default}"
        if self.name is not None:
            r = f"{self.name}: "+r

        return r

    _file_cache = dict()  # where we cache files
    def read_value(self,
                   dirpath:pathlib.Path =pathlib.Path.cwd(),
                   clean: bool =False):
        """ Read  value from disk file containing namelists. Files will be cached to speed up subsequent reads.
        :param dir: Directory where namelist is.
        :param clean -- If True clean the cache before reading.
        """
        namelists = self.file_cache(dirpath / self.filepath, clean=clean)
        try:
            value = namelists[self.namelist].get(self.nl_var.lower(),self.default)
        except KeyError: # namelist does not exist. Use default value
            value = self.default
            logging.info(f"Failed to read {self.namelist} returning {self.default} for {self.nl_var}")
        if value is None:
            raise KeyError(f"{self} not found")
        return value

    def to_dict(self):
        """ Return a dictionary representation, suitable to conversion to JSON,  of namelist-var.  """
        d = dataclasses.asdict(self)
        return d

    # class methods now!
    @classmethod
    def from_dict(cls,dct):
        """
        Generate namelist_var from dictionary
        :param dct: dct
        :return: return  namelist initialised from dict.
        """
        return cls(**dct)
    @classmethod
    def clean_cache(cls):
        """
    Clean out the file cache
        :return:
        """
        cls._file_cache = dict()

    @classmethod
    def file_cache(cls, filepath:pathlib.Path, 
                   clean:bool=False, make_copy:bool=True):
        """
        Read/cache file
        :param filepath: path to file containing namelist
        :param clean: If True,clean the cache
        :param make_copy: If True, make a deep copy of the cached data
        :return: namelists in the file. Will read in data if needed.
        """
        if clean:
            cls._file_cache = dict()  # reset the cache.
        if filepath in cls._file_cache.keys():
            namelists = cls._file_cache[filepath]
        else:
            namelists = f90nml.read(filepath)
            cls._file_cache[filepath] = namelists
            logging.info(f"Read in data from {filepath}")
        if make_copy:
            namelists = copy.deepcopy(namelists)
        return namelists

    @classmethod
    def modify_namelists(cls, nl_info:iter, 
                         dirpath: pathlib.Path = pathlib.Path.cwd(), 
                         update: bool = False,
                         clean: bool = False) -> dict:
        """
        Update dict indexed by files. Each containing a f90nml namelist.
        Really for internal use by this class.
        :param nl_info -- iterable of namelist, values.
        :param dirpath: path to root directory for namelists
        :param clean: clean cache if set to True
        :param update If true will update namelist from (cached) file system. If not will return nl info for changes.
        :return: dict indexed by the namelist filepath

        Example usage files=files_to_change([(VF1_nl,3.0), (RHCRIT_nl,[0.8,0.8,0.8,...0.9,0.95])],dirpath=pathlib.Path('test_dir'))
        """
        file_dict = {}
        for (nl, value) in nl_info:
            if not isinstance(nl,namelist_var):
                raise ValueError(f"{nl} is {type(nl)} expecting type: namelist_var")
            path = dirpath / nl.filepath
            if path not in file_dict.keys():  # got this file? if not add it in.
                if update:
                    file_dict[path] = cls.file_cache(path, clean=clean)  # not got it so use cache
                    logging.debug(f"Updating namelists in {path}")
                else:
                    file_dict[path] = f90nml.namelist.Namelist()  # initialise to empty namelist.
                    logging.debug(f"Setting {path} empty")

            if nl.namelist.lower() not in file_dict[path].keys():
                logging.debug(f"Setting {nl.namelist.lower()} to empty")
                file_dict[path][nl.namelist.lower()] = f90nml.namelist.Namelist()

            file_dict[path][nl.namelist.lower()][nl.nl_var.lower()] = value
            if isinstance(value,np.ndarray): # convert numpy arrays.
                file_dict[path][nl.namelist.lower()][nl.nl_var.lower()] = value.tolist()
            logging.debug(f"Setting {nl}  to {value}")
        return file_dict

    @classmethod
    def nl_modify(cls, nl_info: iter, dirpath=pathlib.Path.cwd()):
        """
        Modifiy namelist files. Sadly f90nml.patch() is a bit flaky.
          So, for each file we read in the entire contents. 
         Update using the changes, write to a temp file, remove the input file
            and move the temp file to the original location.
        :param nl_info: iterable of namelist_var, value pairs,
          will also clear cache after all modification done.
        :return: nada though all files used will be modified.

        Example usage namelist_var.nl_modify({VF1_nl:0.5,ENTCOEF_nl:[0.8,0.8,0.85,...0.9,0.95])
        """

        namelists = cls.modify_namelists(nl_info, dirpath=dirpath)
        for filepath, nl_patch in namelists.items():
            bak_file = filepath.with_name(filepath.name + ".bak")
            shutil.copy2(filepath, bak_file, follow_symlinks=False)  # keep symlinks as symlinks.
            logging.debug(f" {filepath} copied to {bak_file}")
            with tempfile.NamedTemporaryFile(dir=dirpath, delete=False, mode='w') as tmpNL:
                # control how namelist is output.
                nl_patch.end_comma = True
                nl_patch.uppercase = True
                nl_patch.logical_repr = ('.FALSE.', '.TRUE.')  # how to represent false and true
                full_nl = f90nml.read(filepath)
                full_nl.update(**nl_patch)
                f90nml.write(full_nl,tmpNL)
                tmpNL.close()
            filepath.unlink() # remove the input file
            pathlib.Path(tmpNL.name).rename(filepath) # move temp file to original location.
            logging.info(f"Modified {filepath}")
        cls.clean_cache()  # cache now "dirty" (been modified) and so needs to  be cleaned.
        return True  # modification succeeded






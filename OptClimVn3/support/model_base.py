import logging
import os
import pathlib
import typing
import pandas as pd
import numpy as np
import subprocess
import typing
import datetime

import generic_json


class journal:
    """
    Provide history information, ability to run commands and record output.
    """

    @staticmethod
    def now():
        """
        staticmethod to provide now. Done so can mock it for testing.
        Times are such a pain!
        :return: utcnow datetime.datetime
        """
        return datetime.datetime.utcnow()

    def update_history(self, message: typing.Optional[str]):
        """
        Update the _history directory. Key will be self.now()
        If self._history does not exist, then it will be created.
        Routine updates existing values so that multiple updates in a short time will preserve _history.
        Short time defined as less than precision of str(now))
        :param message:message text to be stored.
        If message is None then  self._history will be created and the control returns
        :return:
        """
        if not hasattr(self, '_history'):  # no history so create it as an empty dict
            self._history = {}
        if message is None:
            return
        dt = self.now()
        dtkey = str(dt)  # datetime for now as string

        h = self._history.get(dtkey, [])  # get any existing history for this time
        h += [message]  # add on the message
        self._history[dtkey] = h  # store it back again.
        logging.debug(f"Updated history at {dtkey} ")

    def last_history_key(self) -> typing.Optional[str]:
        """

        :return: last history  or None if no history.
        """

        if hasattr(self,'_history') and (len(self._history) > 0):
            last_hist_key = list(self._history.keys())[-1]
        else:
            last_hist_key = None

        return last_hist_key

    def print_history(self):
        """
         Print out history
         :return:
         """
        if not hasattr(self,'_history'):
            print("No History")
        for time, messages in self._history.items():
            str_msg = '\n'.join(messages)
            print(f"{time}:", str_msg)

    def store_output(self, cmd: typing.Optional[list], result: typing.Optional[str]):
        """
        Store output and cmd ran in self._output with key the time.
           If self._output does not exist it will be created.
        :param cmd: command that was ran
        :param result: result from the command
        If both  are None then only the creation of _output will be done
        :return: Nothing
        """
        if not hasattr(self, '_output'):  # no output so create it as an empty dict
            self._output = {}

        if (cmd is None) and (result is None):
            return

        key = str(self.now())
        store = [dict(cmd=cmd, result=result)]  # store them as a dict so can round trip store them in a json file.
        try:
            self._output[key] += store
        except KeyError:
            self._output[key] = store

        return

    def print_output(self):
        """
        Print out the output
        :return: Nothing
        """

        for key, lst in self._output.items():
            for dct in lst:
                str_cmd = [str(c) for c in dct['cmd']]
                print(f"Command {' '.join(str_cmd)} stored at {key} returned {dct['result']}")
        return

    def run_cmd(self, cmd: list, **kwargs):
        """
        Run a command using subprocess.check_output and record output.
        :param cmd: command to run. Any shell variables ($VARNAME) in the cmd will be expanded at the time of running.
        :**kwargs -- kwargs to be passed to subprocess.check_output. Will update defaults which is just text=True and stderr=subprocess.DEVNULL 
        :return: output from running command
        """
        args = dict(text=True, stderr=subprocess.DEVNULL)  #
        # issue is that fileNotFound will get returned if a file does not exist. Would need to
        # convert to subprocess.CalledProcessError
        args.update(**kwargs)
        cmd_to_run = [os.path.expandvars(c) for c in cmd]
        # using expandvars so any shell variables in command are expanded.
        # this little code fragment from chatGPT (with a bit of nudging/editing) traps that.
        try:
            logging.debug(f"Running {' '.join(cmd_to_run)}")
            output = subprocess.check_output(cmd_to_run, **args)  # run cmd
        except subprocess.CalledProcessError as e:
            print("Failed")
            print("stdout\n",e.output)
            print("="*60)
            print("stderr\n",e.stderr)
            raise
        except FileNotFoundError as e:  # cmd not found
            raise subprocess.CalledProcessError(
                returncode=e.errno,
                cmd=cmd,
                output=None,
                stderr=e.strerror,
            ) from None

        self.store_output(cmd, output)
        return output


class model_base:
    """
    Generic base class for models. Provides default methods for reading/writing data.
    Note that it is not expected to actually be instantiated. See sub-classes (Model for exmaple) for larger functionality.

        Methods:
    --------
    from_dict(cls, dct: dict) -> Any:
        Return class initialised from dct by copying keys to attributes in obj but only that exist after initialisation.
        This is really a factory method.

    to_dict(self) -> dict:
        Convert an object to a dict.

    load(cls, class_json: pathlib.Path) -> Any:
        Load an object configuration from specified file. The correct type of object will be returned.

    dump(self, config_path: pathlib.Path) -> Any:
        Write object (as json) to config_path.
    """

    def __init_subclass__(cls, *args, **kwargs):
        # obscure python. See https://peps.python.org/pep-0487/#new-ways-of-using-classes
        """
        initialise the subclass
        First call superclass __init__subclass__ method
        Then store default methods in registries
        :poram args: args to be passed to supper class __init__subclass__
        :param kwargs:keywords to be passed to super class __init__subclass__
        :return: Nada
        """
        super().__init_subclass__(*args, **kwargs)
        # if subclassing you can change the methods but better to define from_dict and to_dict for your own purposes.
        generic_json.obj_to_from_dict.register_FROM_VALUE(cls, cls.from_dict)
        generic_json.obj_to_from_dict.register_TO_VALUE(cls, cls.to_dict)

    # class methods

    @classmethod
    def from_dict(cls, dct: dict):
        """
        Return  class initialised from dct by copying keys to attributes in obj
        but only those that  exist after initialisation.
        Make sure your dct is sensible...This is very generic.
        This is really a factory method
        :param dct: dict containing information needed by class_name.from_dict()
        :return: initialised object
        """
        obj = cls()  # create an default instance
        obj.fill_attrs(dct) # fill in the values.
        return obj

    def fill_attrs(self,dct:dict):
        """
        Fill in the attributes in self from dct. Used by from_dict.
        :param dct: dict of key value. self.key=value if self.key exists
        :return: Nothing. Changes self in place
        """
        for name, value in dct.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                logging.warning(f"Did not setattr for {name} as not in obj")

    def to_dict(self):
        """
        Convert an object to a dict.
        :return: dct
        """
        dct = dict()
        for key, value in vars(self).items():
            dct[key] = value

        return dct

    @classmethod
    def load(cls, file: pathlib.Path):
        """
        Load an object configuration from specified file.
        The correct type of object will be returned.
        :param file path to file to be read in.
        :return: model configuration of appropriate type.
        """
        # read it in.

        with open(file, 'r') as fp:
            cfg = generic_json.load(fp)
            # this runs all the magic needed to create objects that we know about
        for k, v in vars(cfg).items():  # debug info.
            logging.debug(f"{k}: {v} ")
        logging.info(f"Read configuration from {file}")
        return cfg

    # class methods.

    def __eq__(self, other):
        """
        Equality! and print delta
        :param other: other object
        :return: True if equal
        """
        if type(other) != type(self):
            print(f"Types differ = {type(self), type(other)}")
            return False

        # iterate over the vars of the two objects.
        for (k, v), (k2, v2) in zip(vars(self).items(), vars(other).items()):
            if k != k2:  # names differ. Should not happen.
                raise ValueError("Something wrong")
            if type(v) != type(v2):  # types differ so different
                print(f"Types for {k} differ")
                return False  # types differ -- return False
            elif callable(v):  # callable function so check names are the same.
                if (v.__name__ != v2.__name__):
                    print(f" Fn names for {k} differ")
                    return False  # names differ.
            elif isinstance(v, (pd.Series, np.ndarray, pd.DataFrame)):
                if not np.allclose(v, v2):
                    # check for FP consistency  between the values.
                    print(f"for {k}\n{v}\n differs from\n{v2}")
                    return False  # pandas series differ
            elif v != v2:  # test for different
                print(f"{k} differ")
                return False
            else:
                pass  # equal -- keep going

        return True  # equal if here!

    def class_name(self):
        """
        Return the class name.
        """
        return self.__class__.__name__

    def dump(self, config_path: pathlib.Path):
        """
        Write object (as json) to config_path
        :param config_path: path to write data to. Directory where this goes will be created if necessary.
        :return:whatever result of generic_json.dump is
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as fp:
            result = generic_json.dump(self, fp, indent=2)  # JSON encoder does the magic needed
        logging.info(f"Wrote to {config_path}")
        return result

    @classmethod
    def expand(cls, filestr: typing.Optional[str]) -> typing.Optional[pathlib.Path]:
        """
        Expand any env vars, convert to path and then expand any user constructs.
        :param filestr: path like string or None
        :return:expanded path or, if filestr is None, None.
        """
        if filestr is None:
            return None
        path = os.path.expandvars(filestr)
        path = pathlib.Path(path).expanduser()
        return path

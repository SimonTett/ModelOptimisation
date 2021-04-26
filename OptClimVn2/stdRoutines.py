"""
stdRoutines: provide generic functions (which are not methods) to all modules.
"""
import os
import errno
import stat
import json


class modelEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Convert obj into something that can be converted to JSON
        :param obj -- object to be converted
        :return "primitive" objects that can be converted + type of object
        """
        stype = str(type(obj))  # string rep of data-type
        # try various approaches to generate json file
        fnsToTry = ['to_dict', 'tolist']
        for fn in fnsToTry:
            f = getattr(obj, fn, None)
            if f is not None:
                return dict(stype=stype, data=f())  # run the method we've found
        # failed to run fns that made stuff can turn into JSON test
        if 'dtype' in dir(obj):
            return dict(data=str(obj), dtype=stype)
        else:
            return json.JSONEncoder(self,
                                    obj)  # Let the base class default method try to convert and raise the TypeError
        
def errorRemoveReadonly(func, path, exc):
    """
    Function to run when error found in rmtree.
    :param func: function being called
    :param path: path to file being removed
    :param exc: failure status
    :return: None
    """

    excvalue = exc[1]
    print("Func is ", func)
    # if func in (os.rmdir, os.remove,builtins.rmdir) and excvalue.errno == errno.EACCES:
    if excvalue.errno == errno.EACCES:
        # change the file to be readable,writable,executable: 0777
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        # retry
        try:
            func(path)
        except WindowsError:  # dam windows.
            os.chmod(path, stat.S_IWRITE)
            print("Dealing with windows error for %s." % path)
            func(path)

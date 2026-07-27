# Plan to make paths in Study/SubmitStudy/runSubmitStudy mostly relative paths

## Goals
 Simplify path handling for SubmitStudy, Study & runSubmit objects.

## Why change paths in Study/SubmitStudy?
- There is a lack of clarity within Study, SubmitStudy & runSubmitStudy about model paths, paths to config and rootDir.
- Only reused models from  outside the directory where things are running are the initial perturbation cases. (though others are possible).
- Will handle this by copying (and optionally updating)  model cases from a single old SubmitStudy config.
- Relative paths make archiving and reuse so much easier. No need to adjust paths in the archive.
- Downside – may break a lot of exising configs. Will handle this by picking it up in from_dict.
- To make generalised updating easier in the future, we will introduce version numbering within from_dict/to_dict in Model and Study.


## Path Contract.

- Most paths in Study, except study_dir, (and anything that  inherits it) should be relative to study_dir.
- study_dir should be the absolute path to the directory where the study is being run which will be the parent of the config file.
- Study etc objects then deal with converting paths to absolute paths when IO is needed.
- When reading in a config file study_dir should be updated to the absolute path of the parent directory.
- This should also be so for Model and model_dir. [I think this is the case now.]
- the rule is configurations store relative paths, when needed they become absolute paths.
- code will be provided to do the needed conversions.

## Path refactoring
The aim is to make it easier to move the study to a different location, simplify logic and support reuse of existing models.
- rename rootDir to study_dir.
- Most paths in Study (and anything it inherits from) should be relative to study_dir.
- Most paths in Model should be relative to model_dir. I think this is the case now.
     Exception is config_path which is absolute. Will need changing
- When using relative paths will need to generate full paths for any file access.
- Done by prepending study_dir or model_dir to the relative path as neeed.
- This implies changes to config_path which should now be relative to study_dir.
- study_dir (and possibly model_dir) get modified on load (they need to be set to the absolute path of the parent directory).
- add to Study a method to generate full paths for any file access. Add similar to Model.
- Add new **small** class to handle conversion of relative paths to absolute paths. This to be used everywhere.
- could be done by using dataclass. Use @property to generate full paths for any file access. Add similar to Model. Paths needed:
- Model:
     config_path (be made relative to model_dir??)
     config_dir (already relative to model_dir)
     submit_script (already relative to model_dir)
     continue_script (already relative to model_dir)
 - Study:
        change rootDir to study_dir.
 - SubmitStudy:
        model_index. Only for saving/read in.  See to_dict/from_dict to handle this.
        config_path -- path to config. On load make relative and update study_dir
        Change:   copyConfig(), archive(), create_model(), load_SubmitStudy() and dump_config()
        Modify runAlgorithm to use study_dir rather than rootDir.


## Handling old configs.
- If a config is read in from a file it will be updated to the new format.
- Will need to audit various old configs to see what the mixture of relative and absolute paths is.


## Implementation approach
- add data_version to Study & Model. from_dict then updates old versions to new versions.
- Relative paths have an attribute XXXX_rel_path which is a PurePosixPath.
- Use a focussed class to handle conversion of relative paths to absolute paths.
 This to be used everywhere and so goes into genericLib
- make this class a descriptor class:

```
from pathlib import Path, PurePath

class RelativePath:
    def __init__(self, rel_attr: str, base_attr: str):
        self.rel_attr = rel_attr
        self.base_attr = base_attr

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        rel = getattr(obj, self.rel_attr)
        base = getattr(obj, self.base_attr)
        return Path(base) / rel
```

and use it like this, as part of the class definition:
```
config_path = RelativePath("_config_rel_path", "study_dir")
```
Making sure to set _config_rel_path to a PurePosixPath in init. 
- Stage 1 do this for things that are currently abs paths.
    * Run tests and think of more tests.  
    * Generate a new config and check that it works. 
    * Read an old config to check update works
    * Add tests to genericLib for RelativePath class. 
- Stage 2 do this for things that are relative paths. [This will be more complicated.]




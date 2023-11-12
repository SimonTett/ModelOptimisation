from model_base import model_base,journal
import typing
import pathlib
import platform
import tempfile
import os
from SubmitStudy import SubmitStudy
import tarfile
import logging
my_logger = logging.getLogger(f"OPTCLIM.{__name__}")

class archive_study(model_base, journal):
    # Holds info for archive
    archive_info: dict  # archive info
    archive_path: typing.Optional[pathlib.Path]  # path to the archive. or None
    config_file: typing.Optional[pathlib.PurePath]  # path to the configuration or None.
    rootDir:typing.Optional[pathlib.PurePath] # the rootDir -- used to support file path rewritting
    """
    Archive class for studies
    """

    def __init__(self):
        """
        Generate an archive_study instance. Mostly sets up platform info.

        """

        # add on archive_info
        # information wanted. Computer name. Time created. OS
        self.archive_info = dict(platform_uname=platform.uname()._asdict(),
                                 processor=platform.processor(),
                                 python_version=platform.python_version())
        self.archive_path = None
        self.config_file = None
        self.rootDir = None
        self.update_history("Archive created")

    def archive(self, submit:SubmitStudy,
                archive_path: typing.Optional[pathlib.Path] = None,
                extra_paths: typing.Optional[typing.List[pathlib.Path]] = None):
        """
        Archive a SubmitStudy (or anything that has an archive method)
        :param submit: SubmitStudy to be archived
        :param archive_path: name of the archive file. If None then submit.roodDir/archive.tar is used.
        :param extra_paths: Extra files to be archived. Passed to SubmitStudy.archive.
        :return: archive_path
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            submit.dump(pathlib.Path(tmpdir)/submit.config_path.name) # dump it to tmpdir. Note no changes made.
            if archive_path is None:
                archive_path = submit.rootDir/ f'archive_{submit.name}.tar'

            self.update_history(f"Archiving data for {submit.config_path}")
            self.archive_path = archive_path
            self.config_file = pathlib.PurePath(submit.config_path.name)
            self.rootDir = submit.rootDir
            # dump the archive to somewhere temp.
            apath = pathlib.Path(tmpdir)/'archive.acfg'
            self.dump(apath)

            with tarfile.open(archive_path,mode='w') as archive:
                archive.add(apath,apath.name)  # archive the archive info.
                submit.archive(archive,extra_paths=extra_paths)  # now archive the SubmitStudy.

        return archive_path


    @classmethod
    def extract_archive(cls,
                        archive_path: pathlib.Path,
                        direct: typing.Optional[pathlib.Path]=None) -> ("archive_study", SubmitStudy):
        """
        Extract an archive.
        :param archive_path: Path to the archive.
        :param direct: Directory where data will be extracted to. If None will be current working directory.
        :return: Archive_study (useful for querying archive info) & SubmitStudy -- ready to analyse/continue SubmitStudy.
        """
        if direct is None:
            direct = pathlib.Path.cwd()
        with tarfile.open(archive_path, 'r') as archive:
            archive.extractall(path=direct)  # extract all data
            archive_config: archive_study = model_base.load(direct / 'archive.acfg', check_types=[archive_study])
            cfg_path = direct / archive_config.config_file
            # we are reading an archive which means paths need rewriting.
            model_base._translate_path_var = [archive_config.rootDir,direct]  # setup for translation.
            model_base._convert_path2pure = True  # convert paths to pure paths.
            # QUITE ugly to use class variables to translate. Needed because can't pass args into from_dict
            cfg = SubmitStudy.load_SubmitStudy(cfg_path)  # read the extracted data
            # Save the configuration in the new space which should be normally readable.
            cfg.dump_config(dump_models=True) # and write it back again
            #turn off translation.
            model_base._translate_path_var = None
            model_base._convert_path2pure = False

        return archive_config, cfg

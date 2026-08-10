#!/usr/bin/env python
"""
Small control script for OptClim SubmitStudy configurations.

Usage: OptClim_control.py CONFIG command [options]

Commands implemented: stop, continue, kill, update &  plot

This script intentionally keeps behaviour minimal: it loads the SubmitStudy
object from the configuration, acquires the study lock (non-blocking by
default) and performs the requested action. Plotting also uses the lock as
requested.

No changes are made to other project files.

First version AI generated but then was extensively edited.
"""

import argparse
import logging
import pathlib

import matplotlib

import genericLib
from StudyConfig import readConfig
from runSubmit import runSubmit

matplotlib.use('Agg')  # make sure no matplotlib windows.


def main(argv=None):
    parser = argparse.ArgumentParser(description="Control OptClim SubmitStudy configurations")
    parser.add_argument('--timeout', type=float, default=0.0,
                        help='Lock timeout in seconds (immediate-fail by default)')
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='WARNING',
                        help="Logging level (default: WARNING)")
    parser.add_argument('CONFIG', type=genericLib.expand,
                        help='Path to SubmitStudy file. Will search for .scfg files in current dir if not specified. ',
                        nargs='?')



    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # update
    p_update = subparsers.add_parser('update', help='Update study configuration from a JSON file')
    p_update.add_argument('NEWCONFIG', type=genericLib.expand, nargs='?',
                          help='Alternate config file to read for update. If not provided will use filename originally used')
    p_update.add_argument('--output', type=genericLib.expand,
                          help='Output path for modified configuration. If not provided will use same path as original config file')

    # plot
    p_plot = subparsers.add_parser('plot', help='Plot study')
    p_plot.add_argument('MONITORFILE', type=genericLib.expand,
                        help='Output file (for plot)', nargs='?')

    p_kill = subparsers.add_parser('kill', help='Kill and Stop study')
    p_stop = subparsers.add_parser('stop',
                                   help='Stop study. Will run any instantiated cases but generate no new runs')
    p_continue = subparsers.add_parser('continue', help='Continue study (after previously stopping )')

    cmds_to_dump = ['continue','stop','kill','update'] # list of commands where config gets dumped.
    args = parser.parse_args(argv)

    # configure logging
    my_logger = genericLib.setup_logging(level=args.log_level, rootname='OPTCLIM.control')

    # noinspection PyUnreachableCode
    if args.CONFIG is None:  # try and find a config file to read in.
        sconfig_files = list(pathlib.Path.cwd().glob("*.scfg"))
        if len(sconfig_files) != 1:
            raise FileNotFoundError(f"Did not find one .scfg file in {pathlib.Path.cwd()} found {sconfig_files} ")

        args.CONFIG = sconfig_files[0]
        my_logger.info(f"Using config file {args.CONFIG}")

    # check args.command is defined.
    if args.command is None:
        parser.print_help()
        raise ValueError("No command specified. ")
    dump_models = False # default is not to dump models.
    with genericLib.ContextFileLock(args.CONFIG,timeout=args.timeout) as lock:     # acquire lock with requested timeout
        study = runSubmit.load(args.CONFIG)  # load config file.
        if args.command == 'stop':
            study.next_command = "stop"
            study.update_history("Stopping algorithm")
            my_logger.info("Stopping")
        elif args.command == 'continue':
            if study.next_command != "stop":
                my_logger.warning("Study is not currently stopped. Setting next_command to None anyway.")
            study.next_command = None
            study.update_history("Continuing algorithm")
            my_logger.info("Continuing Algorithm")
        elif args.command == 'kill':
            my_logger.info("Killing study")
            study.kill()
            study.next_command = "stop"
            # if this running at same time as runAlgorithm is running but runAlgorithm is waiting on file lock
            # making next_command 'stop' will stop any more runs being made. Though will run an instantated cases.
        elif args.command == 'update':
            cfg_file = args.NEWCONFIG or study.config.fileName()
            cfg_file = genericLib.expand(cfg_file)
            if not cfg_file.exists():
                raise FileNotFoundError(f"NEWCONFIG {cfg_file} not found")
            my_logger.info(f"Updating study configuration from {cfg_file}")
            cfg = readConfig(cfg_file)
            study.update_config(cfg)
            if args.output:
                study.config_path = pathlib.Path(args.output)
                dump_models = True
        elif args.command == 'plot':

            plot_file = args.MONITORFILE or pathlib.Path(f"monitor_{study.name}.png")
            my_logger.info(f"Plotting study to {plot_file}")
            study.plot(fname=plot_file)
        else:
            raise ValueError(f"Unknown command {args.command}")
        if args.command in cmds_to_dump:
            study.dump_config(dump_models=dump_models)
    return 0


if __name__ == '__main__':
    rc = main()

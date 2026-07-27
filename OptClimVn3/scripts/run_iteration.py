#!/usr/bin/env python

"""
Run next iteration of OptClimVn3. This script is intended to be run from the command line and will:
1) read the study configuration file
2) update parameters,
3) and run the next iteration of the optimization process.


This is very draft code. Will eventually replace runAlgorithm. Need to work out how to set run_next_iter in the
rSUBMIT object.  Can either do at the creation stage (so frozen) or when running this command, which would allow updating.
Can also allow values to be put into the configuration and then retrieved.  That feels best and consistent with everything
else.
"""
import runSubmit
import logging
import argparse  # parse command line arguments
import optclim_exceptions
import functools
import archive_study
import numpy as np
import genericLib
import sys
import pathlib
import typing



# do minimum startup stuff. Really so can have logging

## main script



def namespace_to_argv(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    drop: typing.Iterable[str] = (),
    include_defaults: bool = False,
    pick_long_option: bool = True,
) -> list[str]:
    """
    AI generated.
    Convert a parsed argparse Namespace back into CLI tokens.

    Parameters
    ----------
    parser
        The ArgumentParser used to parse `args`.
    args
        Parsed Namespace from `parser.parse_args(...)`.
    drop
        Dest names to exclude (e.g. {"dry_run", "submit"}). Note no -- or - should be included. Just provice the name,
    include_defaults
        If False (default), omit args whose value equals parser default.
        If True, include them.
    pick_long_option
        If True, prefer the first '--long' option string when available.
        Otherwise use the first declared option string.

    Returns
    -------
    list[str]
        CLI tokens, e.g. ["--model", "abc", "--dry_run", "input.nc"].
        Suitable for subprocess calls without shell=True.

    Notes
    -----
    1. For flags:
       - store_true  -> include flag only when True
       - store_false -/> include flag only when False
    2. If an argument has multiple aliases, this picks one canonical alias;
       it cannot recover exactly which alias the user originally typed.
    3. Positional args are emitted in parser declaration order.
    """
    drop = set(drop)
    out: list[str] = [] # where the return values go

    def pick_option(action: argparse.Action,pick_long_option: bool = True) -> str:
        """
        Extract the text for the option string to use for this action.
        :param action: The argparse.Action for which to extract the option string.
        :param pick_long_option: Whether to pick the longest option string or not.
        :return:the text for the option string to use for this action.
        """
        if not pick_long_option:
            return action.option_strings[0]
        long_opts = [o for o in action.option_strings if o.startswith("--")]
        return long_opts[0] if long_opts else action.option_strings[0]

    def to_tokens(v) -> list[str]:
        """
        Convert v to a string. If v is list or tuple, then individual elements are converted to strings and returned as a list of strings.
        :param v: The value to convert.
        :return: A list of strings.
        """
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return [str(v)]

    for action in parser._actions: #
        dest = action.dest
        print(dest)
        if dest == "help" or dest in drop or not hasattr(args, dest):
            # help or in dest or has no dest -- so skip
            continue

        value = getattr(args, dest)

        if not include_defaults and value == parser.get_default(dest):
            continue # have a default value. Don't bother

        # Positional argument
        if not action.option_strings:
            if value is not None:
                out.extend(to_tokens(value))
            continue

        opt = pick_option(action,pick_long_option=pick_long_option) # what the option is.

        if isinstance(action, argparse._StoreTrueAction):
            if value:
                out.append(opt)

        elif isinstance(action, argparse._StoreFalseAction):
            if value is False:
                out.append(opt)

        elif isinstance(action, argparse._CountAction):
            out.extend([opt] * int(value or 0))

        elif isinstance(action, argparse._AppendAction):
            for item in value or []:
                out.append(opt)
                out.extend(to_tokens(item))

        else:
            if value is not None:
                out.append(opt)
                out.extend(to_tokens(value))

    return out


def set_next_iter_cmd(rSUBMIT:runSubmit.runSubmit,next_iter_cmd:list[str|pathlib.Path]):
    """
    Set next_iter_cmd in runSubmit. If it is the same as the current value, no change will be made.
    TODO: Merge into SubmitStudy.
    :param rSUBMIT:
    :param next_iter_cmd:
    :return:
    """
    if next_iter_cmd and ((not rSUBMIT.next_iter_cmd) or (rSUBMIT.next_iter_cmd != next_iter_cmd)):
        # only update self.next_iter_cmd if next_iter_cmd is truthy (list with more than one element)
        # and
        # if either self.next_iter_cmd is empty/None or self.next_iter_cmd is not the same as next_iter_cmd
        rSUBMIT.update_history(f'Modifying {rSUBMIT.next_iter_cmd} to {next_iter_cmd}')
        logging.debug(f'Setting next_iter_cmd to {next_iter_cmd}')
        rSUBMIT.next_iter_cmd = next_iter_cmd


def main(argv=None):
    ## set up command line args

    expected_env_vars = ['OPTCLIM_ROOT_DIR', 'OPTCLIM_LOG_DIR', 'OPTCLIM_JOB_ID']
    parser = argparse.ArgumentParser(
        description=f"Run next iteration of study providing the follow env variables: {' '.join(expected_env_vars)}",
        allow_abbrev=False)
    parser.add_argument("config_path", type=genericLib.expand, help='Path to the existing configuration file (.scfg)')
    parser.add_argument("--log_level", default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="level of logging info to display. Default is WARNING")
    parser.add_argument(
        "--mode",
        choices=["run", "fake-run", "dry-run", "read-only"],
        default="run",
        help="Mode to run the script. Choices are: run (default), fake-run (simulate model runs), "
             "dry-run (no models submitted), read-only (no changes made to configuration)."
    )
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout in seconds for acquiring lock on configuration file. Default is 60 seconds.")
    parser.add_argument("--monitor", action='store_true', help='Produce monitoring plot after running')
    parser.add_argument("--archive", action='store_true', help="Archive configuration as tar file if  complete.")

    parser.add_argument("--guess_fail", action='store_true',
                        help="If set then use guess_fail to see if Running models have failed and set them failed.")
    fail_help_str = """Behaviour for models that failed. Choices are:
                    fail (default), 
                    continue (continue run with no changes)), 
                    perturb (perturb run, restart run), 
                    perturbc (perturb run, continue run),
                    delete (delete model).
                These cases will be submitted (unless delete or fail are choice)"""
    parser.add_argument("--fail", help=fail_help_str,
                        default='fail',
                        choices=['fail', 'continue', 'perturb', 'perturbc', 'delete'])


    args = parser.parse_args()

    # work out what next_iter_cmd is. Most of the work done by namespace_to_argv
    next_iter_cmd = [sys.executable, pathlib.Path(__file__).resolve()]
    parsed_args = namespace_to_argv(parser, args, drop=['fail','guess_fail'], include_defaults=False)
    next_iter_cmd+=parsed_args

    level = args.log_level
    mode = args.mode
    fail=args.fail
    ## set up the default logging.
    fmt = '%(levelname)s:%(name)s:%(funcName)s: %(message)s'
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format=fmt,
        force=True
    )
    logging.info(f"next_iter_cmd is {next_iter_cmd}")
    rSUBMIT:runSubmit.runSubmit = runSubmit.runSubmit.load(args.config_path)
    configData = rSUBMIT.config  # get the stored configuration data. This is a StudyConfig object
    my_logger = genericLib.setup_logging(
        level=level, # as specifying a level will always get logging from the config.
        log_config=configData.logging_config()
    )
    #
    if mode == 'run':  # running models.
        # First deal with failed models and then models that are already instantiated and so need running.
        # This happens if not all models that were instantiated were submitted.
        if args.guess_fail: # guess if models have failed. See Model.guess_failed to see how that is done.
            models_guess_failed = rSUBMIT.guess_failed()
        # test for RUNNING models. If any fail.
        running_models = rSUBMIT.running_models()
        if len(running_models) > 0:
            raise ValueError(f"{rSUBMIT} has {len(running_models)} running. Try --guess_fail if those have failed. Otherwise wait...")
        failed_models = rSUBMIT.failed_models()
        if len(failed_models):  # Some runs failed. Use fail to decide what to do
            my_logger.info(f"{len(failed_models)} models failed. {rSUBMIT}")
            if fail == 'fail':
                raise ValueError(f"{rSUBMIT} has FAILED models. Try --fail option: \n"+fail_help_str)
            for model in failed_models:
                my_logger.debug(f"Dealing with fail = {fail} for model {model}")
                if fail in ['perturb', 'perturbc']:
                    model.perturb()  # perturb  model
                if fail in ['perturbc', 'continue']:  # model to continue
                    model.status = 'CONTINUE'
                if fail == 'delete':  # delete model
                    rSUBMIT.delete_model(model)

    # check status is only PROCESSED or INSTANTIATED.
    status = rSUBMIT.status()
    if any(s not in ['PROCESSED', 'INSTANTIATED'] for s in status):
        raise ValueError(f"Have unexpected status rSUBMIT:{rSUBMIT}")

    if mode == 'fake-run':
        fakeFn = functools.partial(genericLib.fake_fn, configData)
        fakeFn.__name__ = 'partial genericLib.fake_fn with configData'
    else:
        fakeFn = None

    # these should really be in the config. And the output should be algorithm-specific.
    json_name = configData.filename().stem
    final_JSON_file = rSUBMIT.rootDir / (json_name + "_final.json")
    monitor_file = rSUBMIT.rootDir / (json_name + "_monitor.png")
    # set up known algorithms
    algorithm_handlers = {
        "DFOLS": lambda: rSUBMIT.runDFOLS(scale=True),
        "PYSOT": lambda: rSUBMIT.runPYSOT(scale=True), # these needs refactoring...
        "GAUSSNEWTON": lambda: rSUBMIT.runGaussNewton(scale=True),
        "JACOBIAN": rSUBMIT.runJacobian,
        "RUNOPTIMISED": rSUBMIT.runOptimized,
        "RUN_PARAMS": lambda: rSUBMIT.run_params(scale=True),
    }
    algorithm_name = configData.optimise()['algorithm'].upper()
    if algorithm_name not in algorithm_handlers:
        raise ValueError(f"Algorithm {algorithm_name} is not supported.")
    non_deterministic = configData.optimise().get('nondeterministic', 'fail')
    my_logger.debug(f"Algorithm is {algorithm_name} and non_deterministic is {non_deterministic}")
    want_cost = algorithm_name not in {"RUNOPTIMISED", "JACOBIAN"} # no costs for these two cases.
    save_final_config = algorithm_name != "DFOLS" # DFOLS handles saving itself.
    final_config = None  # so we have something!

    with rSUBMIT.lock(timeout=args.timeout) as lock:
        try:  # run an algorithm iteration.
            np.random.seed(123456)
            rSUBMIT.reset_logical_info()
            final_config = algorithm_handlers[algorithm_name]() # run the algorithm.
            # This, for now, will return a StudyConfig object with the final configuration.
            # refactor to produce an algorithm-specific file with std stuff in runSubmit.
            rSUBMIT.check_deterministic(error=non_deterministic)
        except optclim_exceptions.submitModel:
            # error which triggers need to instantiate and run more models.
            rSUBMIT.check_deterministic(error=non_deterministic)  # check still deterministic.
            if mode not in ['read_only']: # if read_only then don't instantiate anything.
                iter_count = rSUBMIT.instantiate()  # instantiate all cases that need instantiation.
                # This also generates iteration information.
                my_logger.info(f"Instantiated {rSUBMIT}")
            if mode in ["read-only",'dry-run']: # nothing else to do now.
                my_logger.info(f"{mode} -- exiting")
                return 0 # finished.
            set_next_iter_cmd(rSUBMIT, next_iter_cmd)  # set the next_iter_cmd in the rSUBMIT object.
            n_models = rSUBMIT.submit_all_models(fake_fn=fakeFn)  # submit runs which also saves the config.
            my_logger.info(f"On iteration {iter_count} submitted {n_models} models")
            if save_final_config:
                final_config = rSUBMIT.runConfig(scale=True, add_cost=want_cost)  # generate final configuration
        # end of try.

        rSUBMIT.dump_config(dump_models=True)  # dump the configuration & all models. Can now release the lock.
    # released the lock

    # Deal with final stuff -- happens after lock released so can be done in parallel with other processes.
    if args.monitor:
        rSUBMIT.plot(fname=monitor_file)  # plot "std plot"

    if final_config is not None and save_final_config:  # have a finalConfig. If so save it. We could not have it if dry_run or read_only set.
        final_config.save(final_JSON_file)

    if args.archive:
        archive = archive_study.archive_study()
        archive.archive(rSUBMIT,
                        extra_paths=[final_JSON_file.relative_to(rSUBMIT.rootDir),
                                     monitor_file.relative_to(rSUBMIT.rootDir)])
    return 0 # all done

if __name__ == "__main__":
    rc = main()
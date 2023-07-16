from __future__ import annotations

import logging
import typing  # TODO add type hints to all functions/methods.

import numpy as np

from ModelBaseClass import register_param
from Model import Model # note this seems to be quite important. Import Model from Model means the registration does not happen..
import importlib.resources
from namelist_var import namelist_var
import pathlib
import datetime
import fileinput
import shutil
import re
import stat
import engine
import genericLib


def IDLinterpol(inyold, inxold, xnew):
    """
    :param inyold: 3 element tupple of y values
    :param inxold: 3 element tupple of x values
    :param xnew:  x value for interpolation
    :return: interpolated values
    """

    assert len(inyold) == 3, "IDLinterpol has yold len of %d" % (len(inyold))
    assert len(inxold) == 3, "IDLinterpol has yold len of %d" % (len(inxold))
    from numpy import zeros

    yold = zeros(4)
    xold = zeros(4)
    yold[1:] = inyold
    xold[1:] = inxold

    if xnew < xold[1]:
        raise NameError("xnew outside function range %f, min was %f" % (xnew, xold[1]))
    elif xnew <= xold[2]:
        ynew = (yold[1] - (((yold[2] - yold[1]) / (xold[2] - xold[1])) * xold[1])) + (
                (yold[2] - yold[1]) / (xold[2] - xold[1])) * xnew
    elif (xnew <= xold[3]):
        ynew = (yold[2] - (((yold[3] - yold[2]) / (xold[3] - xold[2])) * xold[2])) + (
                (yold[3] - yold[2]) / (xold[3] - xold[2])) * xnew
    else:
        raise NameError("xnew outside function range %f, max was %f" % (xnew, xold[3]))
    return ynew


import math


class HadCM3(Model):
    """
    HadCM3 class.
      Not much different from Model except defines  a bunch of parameters and functions used to modify namelists.
      Which, by definition, are specific to HadCM3.
    """

    def __init__(self, name: str, reference: pathlib.Path, **kwargs):
        """"
        HadCM3 Init -- calls super().__init__(*args,**kwargs)
        then sets submit_script to "SUBMIT" and continue script to SUBMIT.cont
        See Model for documentation on key word parameters
        :param name -- name of the model. Should be 5 characters or less
        :param reference -- where reference config lives.
        """
        if len(name) > 5:
            raise ValueError("HadXM3 limited to 5 character names")
        super().__init__(name, reference,**kwargs)  # call super class init and then override
        # modify submit_script & continue_script

        self.submit_script = 'SUBMIT'
        self.continue_script = 'SUBMIT.cont'
        self.post_process_file = 'post_process.sh' # extra attributed needed.
        # self.runInfo=dict()

    def Name(self):
        """

        :return: The name checking length is 5
        """
        name = self.name
        if len(name) != 5:
            raise ValueError(f"For HadCM3 Name must be 5 characters and is {name} ")
        return name

    def modify_model(self):
        """
        HadCM3 modify_model. Overwrites Model version.
        Call the superclass. modify_model in case it does something!
        Then work out the path to set_status_script
        Cleans up fixClimFCG
        Modify the Submit, then script, create work dir and copy any astart ostart files there might be into it.,
        Then create the continue submit file.
        Finally create the script that tests if finished or not.
        :return:
        """
        super().modify_model()

        self.fixClimFCG()  # fix the ClimFGC namelist
        # Step 1 --  copy SUBMIT & SCRIPT to SUBMIT.bak & SCRIPT.bak so we have the originals
        # Then modify SUBMIT & SCRIPT in place
        #
        self.modifySubmit()  # modify Submit script
        self.modifyScript(self.set_status_script)  # modify Script
        self.createWorkDir()  # create the work directory (and fill it in)
        self.genContSUBMIT()  # generate the continuation script.
        self.createPostProcessFile(self.set_status_script)



    def perturb(self, parameters: typing.Optional[dict] = None):
        """
        Perturb HadCM3 model. Default works through up to 6 parameters then gives up!
        :return: nothing
        """
        if parameters is not None:
            logging.warning("Passing parameters into HadCM3 perturb method. These are ignored")
        parameters_to_perturb = ['VF1', 'ICE_SIZE', 'ENTCOEF', 'CT', 'ASYM_LAMBDA', 'CHARNOCK']
        if self.perturb_count >= len(parameters_to_perturb):
            raise ValueError(
                f"HadCM3 perturbation count is {len(parameters_to_perturb)}"
                f"but only {len(self.parameters)} parameters are available")

        parameter = parameters_to_perturb[self.perturb_count]
        parameters = self.read_values(parameter)
        parameters[parameter] *= (1 + 1e-6)  # small parameter perturbation
        return super().perturb(parameters=parameters)

    def createPostProcessFile(self, set_status_cmd:pathlib.Path):
        """
        Used by the submission system to allow the post-processing job to be submitted when the simulation has
        completed. This code also modifies the UM so that when a NRUN is finished it automatically runs the
        continuation case. This HadCM3 implementation generates a file call optclim_finished which is sourced by
        SCRIPT. SCRIPT needs to be modified to actually do this. (See modifySCRIPT).
        :param set_status_cmd. path to set_status_cmd
        """
        # postProcessCmd is the command that gets run when the model has finished. It should release_job the post-processing!
        outFile = self.model_dir / self.post_process_file  # needs to be same as used in SCRIPT which actually calls it
        # Eddie does not require login in to a login node to subbmit jobs
        with open(outFile, 'w') as fp:
            print(
                f"""#  script to be run in ksh from UM45 SCRIPT
    # it does two things:
    # 1) releases the post-processing script when the whole simulation has finished.
    # 2) When the job has finished (first time) submits a continuation run.
    # I (SFBT) suspect it is rather hard wired for eddie. But will see later.
    export OUTPUT=$TEMP/output_test_finished.$$ # where the output goes
    export RSUB=$TEMP/rsub1.$$
    qshistprint $PHIST $RSUB # get the status info from the history file
    FLAG=$(grep 'FLAG' $RSUB|cut -f2 -d"=" | sed 's/ *//'g)
    if  [ $FLAG = 'N' ]
      then # release_job the post processing job. 
      {set_status_cmd} {str(self.config_path)} SUCCEEDED ## code inserted
      echo "FINISHED releasing the post-processing"
    else 
         echo "$TYPE: Still got work to do"
    fi

    # test for NRUN but not finished.
    SUBCONT={self.continue_script}
    if [[ ( $FLAG -eq 'Y' ) -a ( $TYPE -eq 'NRUN' ) ]
      then
      if [[ -n "$TESTING"  [] # for testing.
      then
    	echo "Testing: Should run $SUBCONT"
    	ls -ltr $SUBCONT
      else
        echo "$SUBCONT"
        $SUBCONT
      fi
    fi
    echo "contents of $OUTPUT "
    echo "========================================="
    cat $OUTPUT
    echo "==================================="
    rm -f $RSUB $OUTPUT # remove temp file.
                    """, file=fp)
            return outFile

    def createWorkDir(self):
        """
        Create the workdir and if self.reference/W has .astart & .ostart copy those into created workDir
        :return: nada
        """

        workDir = self.model_dir / 'W'
        workDir.mkdir(parents=True, exist_ok=True)
        ref_workDir = self.reference / 'W'
        for f in ['*.astart', '*.ostart']:  # possible start files
            files = ref_workDir.glob(f)
            for file in files:
                try:
                    shutil.copy(file, workDir)
                    logging.debug(f"Copied {file} to {workDir}")
                except IOError:
                    logging.warning(f"Failed to copy {file} to {workDir}")

    def modifyScript(self, set_status_script:pathlib.Path):
        """
        modify script.
        :param set_status_script -- path to script that sets status.
         set ARCHIVE_DIR to runid/A -- not in SCRIPT??? WOnder where it comes from. Will look at modified script..
         set EXPTID to runid  --  first ^EXPTID=xhdi
         set JOBID to jobid  -- first ^JOBID

          After . submitchk insert:
         . $JOBDIR/optclim_finished ## run the check for job release_job and resubmission.
            This will be modified by the submission system
         :return:
        """

        runid = self.Name()
        experID = runid[0:4]
        jobID = runid[4]
        modifystr = '## modified'
        with fileinput.input(self.model_dir / 'SCRIPT', inplace=True, backup='.bak') as f:
            for line in f:
                if re.search(modifystr, line):
                    raise Exception("Already modified Script")
                #
                elif f.filelineno() == 1:  # first line
                    print(
                        f"{set_status_script} {str(self.config_path)} RUNNING {modifystr}")  # we are running so set status to RUNNING.
                    print(line[0:-1])  # print line out.
                elif re.match('^EXPTID=', line):
                    print("EXPTID=%s %s" % (experID, modifystr))
                elif re.match('^JOBID=', line):
                    print("JOBID=%s %s" % (jobID, modifystr))
                elif re.match('MESSAGE="Run .* finished. Time=`date`"', line):
                    print('MESSAGE="Run %s#%s finished. Time=`date`"' % (experID, jobID))
                elif re.match('MY_DATADIR=', line):  # fix MY_DATADIR
                    print("MY_DATADIR=%s %s" % (self.model_dir, modifystr))
                elif r'DATADIR/$RUNID/' in line:  # replace all the DATADIR/$RUNID stuff.
                    line = line.replace('DATADIR/$RUNID/', 'DATADIR/')
                    print(line[0:-1], modifystr)
                elif re.match('^. submitchk$', line):  # need to modify SCRIPT to call the postProcessFile.
                    print(line[0:-1])  # print the line out stripping of the newline
                    print(f'. $JOBDIR/{self.post_process_file} {modifystr}')
                    # Self postProcessFile is run.
                    # When it is completed then post-processing gets run.
                elif re.match(r'^exit \$RCMASTER', line):  # add code to deal with failure
                    # Success is when the potentially multiple simulations have completed.
                    # that's handled separately in self.post_process_file
                    print(
                        f"if [[ $RCMASTER -ne 0 ]]; then ;{set_status_script} {str(self.config_path)} FAILED ; fi  {modifystr}")
                    print(line[0:-1])  # print out the original line.
                else:  # default line
                    print(line[0:-1])  # remove newline


    def submit_cmd(self, run_info: dict, engine: engine) -> typing.List[str]:
        """
        :param run_info -- run information.
          should include runCode and runTime.
        :param engine -- engine info. Not used  but provided with superclass method

        :return:  cmd to be run.
        """
        # HadCM3 runs a script which creates the job and then submits it...
        if self.status in ['INSTANTIATED', 'PERTURBED']:
            script = self.model_dir/"SUBMIT"
        elif self.status == 'CONTINUE':
            script = self.model_dir/"SUBMIT.cont"
        else:
            raise ValueError(f"Status {self.status} not expected ")
        runCode = run_info.get('runCode')
        runTime = run_info.get("runTime")
        ok = self.set_time_code(script, runTime=runTime, runCode=runCode)
        if not ok:
            raise ValueError("Something went wrong with set_time_code")
        cmd = [script]
        if engine.connect_fn is not None:
            cmd = engine.connect_fn(cmd)
        return cmd

    def set_time_code(self, script: str,
                      runTime: typing.Optional[int] = None,
                      runCode: typing.Optional[str] = None) -> bool:
        """
        :param script: Name of script file to modify to set runTime and runCode.
          self.model_dir/script will be modified.
        :param runTime: Time in seconds model should run for
        :param runCode:  Code to use to do the run.
        :return: True if succeeded. False if failed.
        """
        if (runTime is None) and (runCode is None):
            return True  # nothing to be done.
        logging.debug(f"Setting runTime to {runTime} and runCode to {runCode} for model {self}")
        modifyStr = '## modified time/code'
        # no try/except here. If it fails then all will fail!
        with fileinput.input(self.model_dir/script, inplace=True) as f:
            for line in f:
                if (runTime is not None) and re.match(r'[NC]RUN_TIME_LIMIT\w*=', line):  # got a time specified
                    l2 = line.split('=')[0]  # split line at = and keep stuff to the left
                    print(f'{l2}={runTime:d} {modifyStr}')  # Change the time
                elif (runCode is not None) and re.match(r'ACCOUNT\w*=', line):  # got a project code specified
                    l2 = line.split('=')[0]  # split line at = and keep stuff to the left.
                    print(f"{l2}={runCode} {modifyStr}")  # Change the runCode
                else:
                    print(line[0:-1])  # remove trailing newline

        return True

    def modifySubmit(self):
        """
        Modify  Submit script for HadCM3
        Changes to SUBMIT
          set RUNID to runid in SUBMIT   == first ^export RUNID=
          set JOBDIR to dirPath   -- first ^JOBDIR=
          set CJOBN to runid in SUBMIT  -- first ^CJOBN=
          set MY_DATADIR to dirPath
          set DATAW and DATAM to $MY_DATADIR/W & $MY_DATADIR/M respectively.


        :return: nada
        """
        # TODO Check that step is 4. Decide if fix (i.e. set STEP=4) or fail with a sensible error message.
        # first work out runID
        runid = self.Name()
        # modify submit
        maxLineNo = 75  # maximum lines to modify
        # modify SUBMIT
        modifyStr = '## modified'
        # check SUBMIT.bakR does not exist. Error if it does.
        if (self.model_dir / 'SUBMIT.bakR').exists():
            raise ValueError(f"Already got backup for SUBMIT = {self.model_dir / 'SUBMIT.bakR'}")
        with fileinput.input(self.model_dir / 'SUBMIT', inplace=True, backup='.bakR') as f:
            for line in f:
                # need to make these changes only once...
                if re.search(modifyStr, line):
                    raise Exception("Already modified SUBMIT")
                elif re.match('export RUNID=', line) and f.filelineno() < maxLineNo:
                    print("export RUNID=%s %s" % (runid, modifyStr))
                elif re.match('MY_DATADIR=', line) and f.filelineno() < maxLineNo:
                    print("MY_DATADIR=%s %s" % (self.model_dir, modifyStr))
                # now find and replace all the DATADIR/$RUNID stuff.
                elif r'DATADIR/$RUNID/' in line:
                    line = line.replace('DATADIR/$RUNID/', 'DATADIR/')
                    print(line[0:-1], modifyStr)
                elif re.match('JOBDIR=', line) and f.filelineno() < maxLineNo:
                    print("JOBDIR=%s %s" % (self.model_dir, modifyStr))
                elif re.match('CJOBN=', line) and f.filelineno() < maxLineNo:
                    print("CJOBN=%s %s" % (runid + '000', modifyStr))

                else:
                    print(line[0:-1])  # remove trailing newline

    def fixClimFCG(self):
        """
        Fix problems with the CLIM_FCG_* namelists so they can be parsed by f90nml.
        Generates CNTLATM.bak and modifies CNTLATM
        :return: None
        """

        with fileinput.input(self.model_dir / 'CNTLATM', inplace=True, backup='.bakR') as f:
            for line in f:
                if re.search(r'CLIM_FCG_.*\(1,', line):

                    line = line.replace('(1,', '(:,')  # replace the leading 1 with :
                print(line[0:-1])  # remove trailing newline.

    def genContSUBMIT(self):
        """
        Generate continuation script for model so that it is a CRUN and STEP=4
        Must run *after* the modifyScript method has ran as the contine script needs those changes too!
        :return: script name of continue simulation.
        """
        # Copy self.submit_script to self.continue_script and then change it.
        modifyStr = '## modifiedContinue'
        submit_script = self.model_dir / self.submit_script
        contScript = self.model_dir / self.continue_script
        with fileinput.input(submit_script) as f:  # file for input
            with open(contScript, mode='w') as fout:  # and where the output file is.
                for line in f:
                    line = line[0:-1]  # remove trailing newline
                    if re.match('^TYPE=NRUN', line):  # DEAL with NRUN
                        line = line.replace('NRUN', 'CRUN', 1) + modifyStr
                        print(line, file=fout)
                    elif re.match('^STEP=1', line):  # Deal with STEP=1
                        print(line.replace('STEP=1', 'STEP=4', 1) + modifyStr, file=fout)
                    else:
                        print(line, file=fout)
        # need to make the script +rx for all readers. Might not work on windows
        fstat = contScript.stat().st_mode
        fstat = fstat | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        fstat = fstat | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        contScript.chmod(fstat)
        return contScript

    ### Stuff for Namelists.

    def atmos_time_step(self):
        """
        Get timestep info and return it in seconds. (Model internally uses steps per day (I think))
        :return: timestep in seconds
        """
        timestep_nl = namelist_var(filepath=pathlib.Path('CNTLATM'), namelist='NLSTCATM', nl_var='A_ENERGYSTEPS')
        steps_per_day = timestep_nl.read_value(dirpath=self.model_dir)
        timestep = 3600.0 * 24 / steps_per_day
        return timestep

    def dx_dy(self, expect=None, units=None):
        """
        Work out the dx and dy values for the model (in radians).
        :param expect: two element np.array of expected values. First element is dx, and second element is dy
        :param units: units to return dx and dy in. None (radians), 'degrees' or 'meters'.
        :return: dx and dy values in specified units.
        """
        dx_nl, dy_nl = (namelist_var(filepath=pathlib.Path('SIZES'), nl_var=var, namelist='NLSIZES')
                        for var in ['ROW_LENGTH', 'P_ROWS'])
        result = np.array([2 / dx_nl.read_value(dirpath=self.model_dir),
                           1 / (dy_nl.read_value(dirpath=self.model_dir) - 1)]) * np.pi

        # do some unit conversions
        RADIUS = 6.37123e06  # radius of earth
        if units is None:
            pass
        elif units == 'degrees':
            result = np.rad2deg(result)
        elif units == 'meters':
            result *= RADIUS
        else:
            raise ValueError(f"Unknown units {units}")
        if (expect is not None) and np.any(result != expect):
            raise ValueError(f"Expected {expect} values but got {result}")
        return result

    def nlev(self, expect=None):
        """
        Work out the number of levels by reading  p_levels
        :param expect If not None if nlev does not equal expect an error will be raised
        :return: the number of levels.
        """

        lev_nl = namelist_var(filepath=pathlib.Path('SIZES'), nl_var='P_LEVELS', namelist='NLSIZES')
        no_levels = lev_nl.read_value(dirpath=self.model_dir)
        if (expect is not None) and (no_levels != expect):
            raise ValueError(f"Expected {expect} levels but got {no_levels}")

        return no_levels

    # functions that are registered.
    # Generically these all take one parameter which is the value to be set.
    # The function then works out what values actually get set.
    # Function returns either a namelist_var, value pair or a list of such.
    # If value is None then the inverse calculation is done by reading the namelist from the model config
    #  and doing the inverse! Consistency checks are also **sometimes** made.
    # Functions have access to the model state via self.

    @register_param('EACF')
    def cloudEACF(self, eacf):
        """
        Compute array of eacf values for each model level. If inverse set work out what meta-parameter is from array.
        :param eacf: meta-parameter value for eacf. If None then inverse will be assumed regardless of actual value
            inverse relationship.  If namelist value not found then default value of 0.5 will be returned.
        """

        nlev = self.nlev(expect=19)  # test have 19 levels
        eacf_nl = namelist_var(filepath=pathlib.Path('CNTLATM'), namelist='SLBC21', nl_var='EACF',default=0.5)
        inverse = eacf is None
        if inverse:
            eacf_val = eacf_nl.read_value(dirpath=self.model_dir)
            if len(eacf_val) != nlev:
                raise ValueError("EA CF namelist {eacf_nl} has len {len(eacf_val)} not {nlev}")
            return eacf_val[0]
        else:
            assert eacf >= 0.5, "eacf seed must be ge 0.5, according to R code but it is %f\n" % eacf
            eacfbl = [0.5, 0.7, 0.8]
            eacftrp = [0.5, 0.6, 0.65]
            eacf1 = IDLinterpol(eacftrp, eacfbl, eacf)
            cloud_eacf = np.repeat(eacf1, 19)
            cloud_eacf[0:5] = eacf
            cloud_eacf[5] = (2. * eacf + eacf1) / 3.
            cloud_eacf[6] = (eacf + 2. * eacf1) / 3.
            return eacf_nl, cloud_eacf.tolist()

    @register_param("SPHERICAL_ICE")
    def sph_ice(self, sphIce: None | bool = True):
        """
        Set/read namelists for spherical/non-spherical ice.
        :param sphIce: True if want spherical ice (which is default). If None inverse calc is done
        :return:  returns nl/tuple pair or, if inverse, the value of sphIce
        """
        nl = [namelist_var(nl_var='I_CNV_ICE_LW', namelist='R2LWCLNL', filepath=pathlib.Path('CNTLATM'),default=1),
              # default 1, perturb 7
              namelist_var(nl_var='I_ST_ICE_LW', namelist='R2LWCLNL', filepath=pathlib.Path('CNTLATM'),default=1),
              # default 1, perturb 7
              namelist_var(nl_var='I_CNV_ICE_SW', namelist='R2SWCLNL', filepath=pathlib.Path('CNTLATM'),default=3),
              # default 3, perturb 7
              namelist_var(nl_var='I_ST_ICE_SW', namelist='R2SWCLNL', filepath=pathlib.Path('CNTLATM'),default=2),
              # default 2, perturb 7
              ]
        values_sph = [1, 1, 3, 2]
        values_nonSph = [7, 7, 7, 7]
        inverse = sphIce is None
        if inverse:  # inverse -- so  extract value and test for consistency
            # check all is OK
            sph = (nl[0].read_value(dirpath=self.model_dir) == 1)
            if sph:
                values = values_sph
            else:
                values = values_nonSph
            # check  namelist values are consistent
            for n, v in zip(nl, values):
                vr = n.read_value(dirpath=self.model_dir)
                assert vr == v, f"Got {vr} but expected {v} for nl {n}"

            return sph

        # set values
        if sphIce:
            values = values_sph
        else:
            values = values_nonSph
        result = list(zip(nl, values))  # return namelist info.
        return result

    @register_param("START_TIME")
    def start_time(self, time_input: str | None):
        """
        :param time_input:  start time as ISO format  string. If None inverse will be done
        :return:  namelist info and values as array. (or if inverse set return the start time)
        """
        # It would be really nice to use cftime rather than datetime
        # but I don't think there is a from isoformat method for cftime..

        # set up all the var/namelist/file info.
        # Need to set MODEL_BASIS_TIME in CNTLALL/&NLSTCALL & CONTCNTL/&NLSTCALL
        namelistData = [namelist_var(nl_var='MODEL_BASIS_TIME', namelist=nl, filepath=pathlib.Path(file)) for
                        file, nl in zip(['CNTLALL', 'CONTCNTL'],
                                        ['NLSTCALL', 'NLSTCALL'])]
        inverse = time_input is None
        if inverse:
            time = namelistData[0].read_value(dirpath=self.model_dir)  # read the first one!
            # check times are the same
            for nl in namelistData[1:]:
                t2 = nl.read_value(dirpath=self.model_dir)
                if t2 != time:
                    raise ValueError(f"Times differ for {namelistData[0]} and {nl}")
            time = datetime.datetime(*time)  # convert to datetime
            time = time.isoformat()  # and format as ISO time string
            return time  # just return the value
        else:
            time = datetime.datetime.fromisoformat(time_input)  # convert iso-format time to datetime
            time = [time.year, time.month, time.day, time.hour, time.minute, time.second]  # what the UM wants
            result = [(nl, time) for nl in namelistData]
            return result  # return the namelist info.

    def time_delta(self, duration, runTarget=True):
        """
        :param duration:  Duration as string in ISO 8061 format (PnYnMnDTnHnMn.nnS)
        :param runTarget: namelist info suitable for runTarget. (default). If False namelist info suitable for resubInterval
        :return: namelist info and values as array. (or if inverse set return the run target )
             return  Duration as 6 element array  TIODO: string in ISO 8061 format (PnYnMnDTnHnMn.nnS)
        """
        # set up all the var/namelist/file info.
        if runTarget:  # suitable for runTarget
            namelistData = [namelist_var(nl_var='RUN_TARGET_END', namelist=nl, filepath=pathlib.Path(file)) for
                            file, nl in zip(['CNTLALL', 'CONTCNTL', 'RECONA', 'SIZES'],
                                            ['NLSTCALL', 'NLSTCALL', 'STSHCOMP', 'STSHCOMP'])]
        else:  # suitable for resubInterval
            namelistData = [namelist_var(nl_var='RUN_RESUBMIT_INC', namelist=nl, filepath=pathlib.Path(file)) for
                            file, nl in zip(['CNTLALL', 'CONTCNTL'],
                                            ['NLSTCALL', 'NLSTCALL'])]
        inverse = duration is None
        if inverse:
            durn = namelistData[0].read_value(dirpath=self.model_dir)
            # check rest are OK
            for nl in namelistData[1:]:
                d2 = nl.read_value(dirpath=self.model_dir)
                if d2 != durn:
                    raise ValueError(f"Durations differ between {nl} and {namelistData[0]}")
            # convert to string
            durn = genericLib.parse_isoduration(durn)
            return durn  # just return the value as a 6 element list.
        else:
            durn = genericLib.parse_isoduration(duration)
            # verify that len of target is >= 1 and <= 6 and if not raise error.
            if len(durn) != 6:
                raise Exception(f"Durn {durn} should have 6 elements. Computed from {duration} ")
            result = [(nl, durn) for nl in namelistData]
            return result  # return the namelist info.

    @register_param("RUN_TARGET")
    def run_target(self, target_input):
        """
        set runTarget -- see timeDelta for documentation.
        :param target_input: target length of run as ISO string
        :return: Return list of nl,value pairs or run_target
        """
        return self.time_delta(target_input)

    @register_param('RESUBMIT_INTERVAL')
    def resub_interval(self, interval_input):
        """
        Set resubmit durations -- see timeDelta for documentation
        :param interval_input:
        """
        return self.time_delta(interval_input, runTarget=False)

    @register_param("NAME")
    def run_name(self, name: None | str):
        """
        Compute experiment and job id.
        :param name: 5-character UM name. If None inverse (read file) calculation is done
        :return: namalises and values to set. unless name is None. Then name will be returned
        """
        # make namelist information.
        jobname_nl = namelist_var(nl_var='RUN_JOB_NAME', namelist='NLCHISTO', filepath=pathlib.Path('INITHIS'))
        exper_nl = namelist_var(nl_var='EXPT_ID', namelist='NLSTCALL', filepath=pathlib.Path('CNTLALL'))
        jobid_nl = namelist_var(nl_var='JOB_ID', namelist='NLSTCALL', filepath=pathlib.Path('CNTLALL'))
        exper2_nl = namelist_var(nl_var='EXPT_ID', namelist='NLSTCALL', filepath=pathlib.Path('CONTCNTL'))
        jobid2_nl = namelist_var(nl_var='JOB_ID', namelist='NLSTCALL', filepath=pathlib.Path('CONTCNTL'))
        inverse = (name is None)
        if inverse:
            name = exper_nl.read_value(self.model_dir) + jobid_nl.read_value(self.model_dir)
            name2 = exper_nl.read_value(self.model_dir) + jobid_nl.read_value(self.model_dir)
            if name != name2:
                raise ValueError(f"Name1 {name} and name2 {name} differ")
            return name  #
        else:
            if len(name) != 5:
                raise ValueError("HadCM3 expects 5 character names not {name}")
            result = [(exper_nl, name[0:4]), (jobid_nl, name[4]),
                      (exper2_nl, name[0:4]), (jobid2_nl, name[4]),
                      (jobname_nl, name + "000")]  # split name to make experID, jobID and jobname
            return result

    @register_param("CW")
    def cloud_water(self, cw_land):
        """
        Compute cw_sea from  cw_land
        :param cw_land: value of cw_land. If None then inverse calculation is done
        :return: returns dict containing cloud_cw_land and cloud_cw_sea values.
        """
        cw_land_nl = namelist_var(nl_var='CW_LAND', namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
        cw_sea_nl = namelist_var(nl_var='CW_SEA', namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
        inverse = (cw_land is None)
        if inverse:
            return cw_land_nl.read_value(dirpath=self.model_dir)
        else:
            cwl = [1e-04, 2e-04, 2e-03]
            cws = [2e-05, 5e-05, 5e-04]
            cw_sea = IDLinterpol(cws, cwl, cw_land)
            return [(cw_land_nl, cw_land), (cw_sea_nl, cw_sea)]

    @register_param("KAY_GWAVE")
    def gravity_wave(self, kay):
        """
        Compute gravity wave parameters given kay parameter.
        :param kay: value of kay. If None then inverse calculation is done.
        :return: list  containing kay_gwave and kay_lee_gwave parameters. (or value of kay if inverse set)
        """

        # name list info
        gwave, lee_gwave = (namelist_var(nl_var=var, namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
                            for var in ['KAY_GWAVE', 'KAY_LEE_GWAVE'])
        inverse = (kay is None)
        if inverse:
            v = gwave.read_value(dirpath=self.model_dir)
            return v
        else:  # from MIke's code (in turn from R code)
            gwd_pt = [1e04, 1.5e04, 2e04]
            lee_pt = [1.5e05, 2.25e05, 3e05]
            lee = IDLinterpol(lee_pt, gwd_pt, kay)
            return [(gwave, kay), (lee_gwave, lee)]

    @register_param("ALPHAM")
    def ice_albedo(self, alpham):
        """
        Compute ice albedo values (alpham & dtice) given alpham
        :param alpham: value of alpham (if forward). If None inverse calculation will be done.

        :return:
        """
        # namelist information
        alpham_nl, dtice_nl = (namelist_var(nl_var=var, namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
                               for var in ['ALPHAM', 'DTICE'])
        inverse = (alpham is None)
        if inverse:
            return alpham_nl.read_value(dirpath=self.model_dir)
        else:
            mins = [0.5, 0.57, 0.65]  # alpham values
            maxs = [10., 5., 2.]  # corresponding dtice values
            dtice = IDLinterpol(maxs, mins, alpham)  # from Mike's code.
            return [(dtice_nl, dtice), (alpham_nl, alpham)]

    # Stuff for diffusion calculation. Rather ugly code. Probably not worth doing much with as diffusion does
    # not seem that important!
    def diff_fn(self, dyndiff, dyndel=6, inverse=False):
        """
        Support Function to compute diff  coefficient
        :param dyndiff: diffusion value in hours. If None inverse calculation will be done
        :param dyndel: order of diffusion. 6 is default corresponding to 3rd order.
        :return:
        """
        timestep = self.atmos_time_step()
        DPHI = self.dx_dy()[1]
        RADIUS = 6.37123e06  # radius of earth
        assert dyndel == 6 or dyndel == 4, "invalid dyndel %d" % dyndel
        # DPHI = dlat * Pi / 180.
        D2Q = 0.25 * (RADIUS * RADIUS) * (DPHI * DPHI)

        if inverse:
            diff_time = (dyndiff / D2Q) ** (dyndel / 2) * timestep
            diff_time = -1 / math.log(1 - diff_time)
            diff_time = diff_time * timestep / 3600.0
            return diff_time
        else:
            dampn = dyndiff * 3600. / timestep
            EN = 1 - math.exp(-1. / dampn)
            ENDT = EN / timestep
            return D2Q * ENDT ** (1.0 / int(dyndel / 2))

    def metaDIFFS(self, dyndiff=12., dyndel=6):
        assert dyndel == 6 or dyndel == 4, "metaDIFFS: invalid dyndel %d" % dyndel
        NLEV = self.nlev()
        DPHI = self.dx_dy()[1]
        timestep = self.atmos_time_step()
        val = self.diff_fn(dyndiff, dyndel)
        tmp1 = dyndel / 2  # integral
        DIFF_COEFF = np.repeat(val, NLEV)
        DIFF_COEFF[-1] = 4e06
        DIFF_COEFF_Q = np.repeat(val, NLEV)
        DIFF_COEFF_Q[-1] = 4e06
        DIFF_EXP = np.repeat(tmp1, NLEV)
        DIFF_EXP[-1] = 1
        DIFF_EXP_Q = np.repeat(tmp1, NLEV)
        DIFF_EXP_Q[-1] = 1
        DIFF_COEFF_Q[13:NLEV - 1] = 1.5e08
        DIFF_EXP_Q[13:NLEV - 1] = 2
        # R gives 7 sig figs, which is managed at the writing to file of these
        # lists.
        return DIFF_COEFF, DIFF_COEFF_Q, DIFF_EXP, DIFF_EXP_Q

    @register_param("DYNDIFF")
    def diffusion(self, diff_time):
        """
        Compute arrays of diffusion co-efficients for all   levels.
        :param diff_time: time in hours for diffusion. If None inverse calculation will be done
        :return: if inverse set compute diffusion timescale otherwise compute diffusion coefficients.
             returns them as a list
             diff_coeff -- diffusion coefficient, diff_coeff_q -- diffusion coefficient for water vapour
             diff_power -- power of diffusion, diff_power_q -- power for water vapour diffusion.
        """
        # TODO -- if necessary allow levels to be specified or read from namelists. Also read other info on model!
        # NAMELIST information.
        diff_coeff_nl, diff_coeff_q_nl, diff_exp_nl, diff_exp_q_nl = (
            namelist_var(nl_var=var, namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
            for var in ['DIFF_COEFF', 'DIFF_COEFF_Q', 'DIFF_EXP', 'DIFF_EXP_Q']
        )
        timestep = self.atmos_time_step()
        DPHI = self.dx_dy()[1]
        inverse = (diff_time is None)
        if inverse:
            powerDiff = diff_exp_nl.read_value(dirpath=self.model_dir)[0] * 2
            diff_time_hrs = self.diff_fn(diff_coeff_nl.read_value(dirpath=self.model_dir)[0], dyndel=powerDiff,
                                         inverse=inverse)
            # round to 3 dps
            diff_time_hrs = round(diff_time_hrs, 3)
            return diff_time_hrs
        else:
            diff_pwr = 6  # assuming 3rd order.
            diff_coeff, diff_coeff_q, diff_exp, diff_exp_q = \
                self.metaDIFFS(dyndiff=diff_time, dyndel=diff_pwr)
            return [(diff_coeff_nl, diff_coeff), (diff_coeff_q_nl, diff_coeff_q),
                    (diff_exp_nl, diff_exp), (diff_exp_q_nl, diff_exp_q)]

    @register_param("RHCRIT")
    def cloudRHcrit(self, rhcrit):
        """
        Compute rhcrit on multiple model levels
        :param rhcrit: meta parameter for rhcrit. If None inverse calculation  will be done.
        :return: namelist, value.
        """
        self.nlev(expect=19)
        rhcrit_nl = namelist_var(nl_var='RHCRIT', namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'), default=0.7)
        inverse = (rhcrit is None)
        if inverse:
            cloud_rh_crit = rhcrit_nl.read_value(dirpath=self.model_dir)
            rhcrit = cloud_rh_crit[3]
            expected = 19 * [rhcrit]
            for it, v in enumerate([0.95, 0.9, 0.85]):
                expected[it] = max(v, rhcrit)
            if expected != cloud_rh_crit:
                raise ValueError(f"Expected rhcrit {np.array(expected)} but got {np.array(cloud_rh_crit)}")
            return rhcrit
        else:
            cloud_rh_crit = 19 * [rhcrit]
            cloud_rh_crit[0] = max(0.95, rhcrit)
            cloud_rh_crit[1] = max(0.9, rhcrit)
            cloud_rh_crit[2] = max(0.85, rhcrit)
            return rhcrit_nl, cloud_rh_crit

    @register_param("ICE_DIFF")
    def iceDiff(self, OcnIceDiff):
        """
        Generate namelist for NH and SH ocean ice diffusion coefficients.
        :param OcnIceDiff: Value wanted for ocean ice diffusion coefficient
            Same value will be used for northern and southern hemispheres. If None inverse calculation will be done.
        :return: (by default) the namelist/value information as list
        """
        iceDiff_nlNH, iceDiff_nlSH = (namelist_var(nl_var=var, namelist='SEAICENL',
                                                   filepath=pathlib.Path('CNTLOCN'),default=2.5e-5)
                                      for var in ['EDDYDIFFN', 'EDDYDIFFS'])
        inverse = (OcnIceDiff is None)
        if inverse:
            v = iceDiff_nlNH.read_value(dirpath=self.model_dir)
            v_sh = iceDiff_nlSH.read_value(dirpath=self.model_dir)
            if v != v_sh:
                raise ValueError(
                    "Ocean ice diffusion coefficient is not the same for northern and southern hemispheres")
            return v
        else:  # set values
            return [(iceDiff_nlNH, OcnIceDiff), (iceDiff_nlSH, OcnIceDiff)]

    @register_param("MAX_ICE")
    def ice_max_conc(self, iceMaxConc):
        """
        Generate namelist for NH and SH ocean ice maximum concentration
        :param iceMaxConc: Value wanted for ocean max concentration
            Same value will be used for northern and southern hemispheres with SH maximum being 0.98.
            If None inverse condition will be done
        :return: nl/values to set the values, if inverse return the NH value.
        """
        iceMax_nlNH, iceMax_nlSH = (namelist_var(nl_var=var, namelist='SEAICENL',
                                                 filepath=pathlib.Path('CNTLOCN'), default=0.995)
                                    for var in ['AMXNORTH', 'AMXSOUTH'])
        inverse = (iceMaxConc is None)
        if inverse:
            v = iceMax_nlNH.read_value(dirpath=self.model_dir)
            v2 = iceMax_nlSH.read_value(dirpath=self.model_dir)
            if min(0.98, v) != v2:
                raise ValueError(f"SH Ocean ice maximum concentration = {v2} not {min(0.98, v)}")
            return v
        else:  # set values
            # SH value limited to 0.98
            return [(iceMax_nlNH, iceMaxConc), (iceMax_nlSH, min(0.98, iceMaxConc))]

    @register_param("OCN_ISODIFF")
    def ocnIsoDiff(self, ocnIsoDiff):
        """
        Generate namelist for changes to Ocean isopycnal diffusion.
        :param ocnIsoDiff: Value for ocean ice diffusion. Will set two values AM0_SI & AM1_SI
           Note these picked by examining Lettie Roach's generated files. If None then inverse calculation will be done
        :return: namelist/values to be set or the namelist value.
        """
        ocnDiff_AM0, ocnDiff_AM1 = (namelist_var(nl_var=var, namelist='EDDY',
                                                 filepath=pathlib.Path('CNTLOCN'), default=1e3)
                                    for var in ['AM0_SI', 'AM1_SI'])
        inverse = (ocnIsoDiff is None)
        if inverse:
            v = ocnDiff_AM0.read_value(dirpath=self.model_dir)
            v2 = ocnDiff_AM1.read_value(dirpath=self.model_dir)
            if v != v2:
                raise ValueError(f"Ocean isopycnal diffusion coefficients differ")
            return v
        else:  # set values -- both  to same value
            return [(ocnDiff_AM0, ocnIsoDiff), (ocnDiff_AM1, ocnIsoDiff)]

    ## class methods now!
    # @classmethod
    # def parse_isoduration(cls, s: str | typing.List) -> typing.List|str:
    #
    #     """ Parse a str ISO-8601 Duration: https://en.wikipedia.org/wiki/ISO_8601#Durations
    #       OR convert a 6 element list (y m, d, h m s) into a ISO duration.
    #     Originally copied from:
    #     https://stackoverflow.com/questions/36976138/is-there-an-easy-way-to-convert-iso-8601-duration-to-timedelta
    #     Though could use isodate library but trying to avoid dependencies and isodate does not look maintained.
    #     :param s: str to be parsed. If not a string starting with "P" then ValueError will be raised.
    #     :return: 6 element list [YYYY,MM,DD,HH,mm,SS.ss] which is suitable for the UM namelists
    #     """
    #
    #     def get_isosplit(s, split):
    #         if split in s:
    #             n, s = s.split(split, 1)
    #         else:
    #             n = '0'
    #         return n.replace(',', '.'), s  # to handle like "P0,5Y"
    #
    #     if isinstance(s, str):
    #         logging.debug("Parsing {str}")
    #         if s[0] != 'P':
    #             raise ValueError("ISO 8061 demands durations start with P")
    #         s = s.split('P', 1)[-1]  # Remove prefix
    #
    #         split = s.split('T')
    #         if len(split) == 1:
    #             sYMD, sHMS = split[0], ''
    #         else:
    #             sYMD, sHMS = split  # pull them out
    #
    #         durn = []
    #         for split_let in ['Y', 'M', 'D']:  # Step through letter dividers
    #             d, sYMD = get_isosplit(sYMD, split_let)
    #             durn.append(float(d))
    #
    #         for split_let in ['H', 'M', 'S']:  # Step through letter dividers
    #             d, sHMS = get_isosplit(sHMS, split_let)
    #             durn.append(float(d))
    #     elif isinstance(s, list) and len(s) == 6:  # invert list
    #         durn = 'P'
    #         logging.debug("Converting {s} to string")
    #         for element, chars in zip(s, ['Y', 'M', 'D', 'H', 'M', 'S']):
    #             if element != 0:
    #                 if isinstance(element, float) and element.is_integer():
    #                     element = int(element)
    #                 durn += f"{element}{chars}"
    #             if chars == 'D':  # days want to add T as into the H, M, S cpt.
    #                 if np.any(np.array(s[3:]) != 0):
    #                     durn += 'T'
    #         if durn == 'P':  # everything = 0
    #             durn += '0S'
    #     else:
    #         raise ValueError(f"Do not know what to do with {s} of type {type(s)}")
    #
    #     return durn

    ## generic function for ASTART/AINITIAL/OSTART/OINITIAl/
    def initHist_nlcfiles(self, value: str | None, nl_var: str = None):
        """

        :param value: value to be set -- typically the name of a file. If None then the inverse calculation will be done.
        :param nl_var: namelist variable to be set.
        :return:namlelist/value tupple to be set or whatever "value" is in the model namelist.
        """
        nl = namelist_var(nl_var=nl_var, namelist='NLCFILES', filepath=pathlib.Path('INITHIS'))

        inverse = (value is None)
        if inverse:
            value = nl.read_value(dirpath=self.model_dir)
            return value.split(':')[1].lstrip()  # extract value from passed in name list removing space when doing so.
        else:
            # read existing value to get the parameter name and then re-order
            value_nml = nl.read_value(dirpath=self.model_dir)

            parameter = value_nml.split(':')[0]
            st = f"{parameter}: {value}"
            return (nl, st)

    @register_param("ASTART")
    def astart(self, astartV):
        """

        :param astartV: value of ASTART
        :return: nl/tuple (if astartV is not None) to be set or whatever astart is in the models namelist.
        """
        return self.initHist_nlcfiles(astartV, nl_var='ASTART')

    @register_param("AINITIAL")
    def ainitial(self, ainitialV):
        """

        :param ainitialV: value of ASTART
        :return: nl/tuple (if ainitialV is not None) to be set or whatever ainitial is in the models namelist.
        """
        return self.initHist_nlcfiles(ainitialV, nl_var='AINITIAL')

    @register_param("OSTART")
    def ostart(self, ostartV):
        """

        :param ostartV: value of OSTART
        :return: nl/tuple (if ostartV is not None) to be set or whatever ostart is in the models namelist.
        """
        return self.initHist_nlcfiles(ostartV, nl_var='OSTART')

    @register_param("OINITIAL")
    def oinitial(self, oinitialV):
        """

        :param oinitialV: value of OSTART
        :return: nl/tuple (if oinitialV is not None) to be set or whatever oinitial is in the models namelist.
        """
        return self.initHist_nlcfiles(oinitialV, nl_var='OINITIAL')

    @register_param("ensembleMember")
    def ens_member(self, ensMember: typing.Optional[int]) -> None:
        """
        Do nothing as for HadCM3 random perturb is from name.
        :param ensMember: ensemble member. The ensemble member wanted.
        :return: None (for now) as nothing done.
        """

        inverse = (ensMember is None)
        if inverse:
            logging.warning("Can not invert ensMember")
            return None

        return None


traverse = importlib.resources.files("Models")
with importlib.resources.as_file(traverse.joinpath("parameter_config/HadCM3_Parameters.csv")) as pth:
    HadCM3.update_from_file(pth, duplicate=True)

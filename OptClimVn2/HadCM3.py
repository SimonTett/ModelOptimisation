"""
Class to support HadCM3 in optimisation (and other approaches) work

"""
import fileinput
# TODO -- now have version 1.0X of f90nml may not need to patch here.
import functools  # std functools.
import glob
import math
import os
import re
import shutil

import \
    f90nml  # NEEDED because f90nml.patch (as used in ModelSimulation) fails with RECONA. For the moment dealing with this here.
import numpy as  np

import ModelSimulation
from ModelSimulation import _namedTupClass

# functions to generate parameter values given meta-parameters.
# could   be class methods or functions..


# variables below are used in diffusion computations and are hardwired for HadAM3 N48 resolution
# A change might be to pass them through with defaults or pull them out of the namelist??

dlat = 2.5  # for N96 this should be 1.25
timestep = 1800.  # for N96 should be 900 ??
diff_pwr = 6  # 6th order diffusion
Pi = 3.1415927  # value of pi
RADIUS = 6.37123e06  # radius of earth
NLEV = 19  # how many levels in atmosphere


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


# functions below are used to convert meta parameters to namelist variables.
# All should have an inverse optional argument. This is used when reading parameters back.
# TODO -- convert functions below to methods. Then cam have access to resolution etc.

def diff_fn(dyndiff, dyndel=6, inverse=False):
    """
    Support Function to compute diff  co-efficient
    :param dyndiff: diffusion value (fowrd should be in hours; backwards in what ever model wants!)
    :param dyndel: order of diffusion. 6 is default corresponding to 3rd order.
    :param inverse (default False) If True return inverse calculation.
    :return:
    """

    assert dyndel == 6 or dyndel == 4, "invalid dyndel %d" % dyndel
    DPHI = dlat * Pi / 180.
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


def metaDIFFS(dyndiff=12., dyndel=6):
    assert dyndel == 6 or dyndel == 4, "metaDIFFS: invalid dyndel %d" % dyndel
    val = diff_fn(dyndiff, dyndel)
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


# functions for name-lists

def sph_ice(sphIce=True, inverse=False, namelist=False):
    """
    Setup namelist for non-spherical ice.
    :param sphIce: True if want spherical ice (which is default)
    :param inverse: Return (from namelists) if spherical ice.
    :param namelist:  Return namelist info
    :return:  returns values or namelist.
    """
    nl = [_namedTupClass(var='I_CNV_ICE_LW', namelist='R2LWCLNL', file='CNTLATM'),  # default 1, perturb 7
          _namedTupClass(var='I_ST_ICE_LW', namelist='R2LWCLNL', file='CNTLATM'),  # default 1, perturb 7
          _namedTupClass(var='I_CNV_ICE_SW', namelist='R2SWCLNL', file='CNTLATM'),  # defaault 3, perturb 7
          _namedTupClass(var='I_ST_ICE_SW', namelist='R2SWCLNL', file='CNTLATM'),  # default 2, perturb 7
          ]
    values_sph = [1, 1, 3, 2]
    values_nonSph = [7, 7, 7, 7]
    if namelist:  # just return the namelist info
        return nl
    elif inverse:  # inverse -- so  extract value and test for consistency
        # check all is OK
        sph = sphIce[nl[0]] == 1
        if sph:
            values = values_sph
        else:
            values = values_nonSph

        for n, v in zip(nl, values):
            if sphIce[n] != v:
                raise Exception("Got %d when expected %d" % (sphIce[n], v))
        return sph

    else:  # setting values
        if sphIce:
            values = values_sph
        else:
            values = values_nonSph
        result = {n: v for n, v in zip(nl, values)}  # return namelist info.
        return result


def initHist_nlcfiles(value='NoSuchFile.txt', inverse=False, namelist=False, parameter=None):
    """

    :param value: value to be set -- typically the name of a file
    :param parameter: string for parameter -- must be provided.
    :param inverse: inverst string (remove bit before the : and return the string after that. )
    :return:string (or inverse) to return.
    """
    nl = _namedTupClass(var=parameter, namelist='NLCFILES', file='INITHIS')

    if namelist:
        return [nl]

    if inverse:
        return value[nl].split(':')[1].lstrip()  # extract value from passed in name list removing space when doing so.
    else:
        # need to generate 8 blanks then fill in with parameter.
        st = "%-8.8s: %s" % (
            parameter, value)  # Alas HadCM3 lives in the 8 character MSDOS world... I wonder why that limit chosen?
        return {nl: st}


def startTime(time=[1965, 7, 4], inverse=False, namelist=False):
    """

    :param time:  start time as 3 to 6 element list. Spcifiying in order year, month, day of month, hour, minute and second
                    if hour, minute and second not specified they are set to zero.
    :param inverse: Do inverse calculation and return 6 element list as [YYYY, MM, DD, HH, MM, SS]
    :param namelist: No computation just return the namelist info. 
    :return:  namelist info and values as array. (or if inverse set return the start time)
        Need to set MODEL_BASIS_TIME in CNTLALL/&NLSTCALL & CONTCNTL/&NLSTCALL but only the year part 
    -- which reguires something clever. No make the specifiy the basis_time and then write it out. 
    fixhd(21) & fixhd(28) in RECONA/&HEADERS  -- this resets the year. (Presumably we could reset the month etc too.) 

    
    Just need to reset year => adding to tupple an indicator for which part of an array to change?? (Breaks existing stuff?)
    ALas f90nml patch seems to fail for RECONA so will need to write it out explicitly. 
    """

    # verify that len of target is >= 3 and <= 6 and if not raise error.
    if not (1 <= len(time) <= 6):
        raise Exception("start time  should have at >= 1 members and <= 6")

    # set up all the var/namelist/file info.
    namelistData = [_namedTupClass(var='MODEL_BASIS_TIME', namelist=nl, file=file) for
                    file, nl in zip(['CNTLALL', 'CONTCNTL'],
                                    ['NLSTCALL', 'NLSTCALL'])]

    if namelist:
        return namelistData  # return the namelist info
    elif inverse:
        return time[namelistData[0]]  # jsut return the value
    else:
        t = time[:]
        t.extend([0] * (len(t) - 6))  # add trailing zeros
        result = {nl: t for nl in namelistData}
        return result  # return the namelist info.


def runTarget(target=[0, 0, 0], inverse=False, namelist=False):
    """
    
    :param target:  run target time as [YYYY, MM, DD, HH, MM, SS] as used by the UM.   
                             If list contains less than 6 elements then remaining values are set to 0.
    :param inverse: Do inverse calculation and return  6 element list as [YYYY, MM, DD, HH, MM, SS]
    :param namelist: Do no computation just return the namelist info 
    :return:  namelist info and values as array. (or if inverse set return the run target )
    """
    # verify that len of target is >= 3 and <= 6 and if not raise error.
    if not (1 <= len(target) <= 6):
        raise Exception("target should have at >= 1 members and <= 6")

    # set up all the var/namelist/file info. 
    namelistData = [_namedTupClass(var='RUN_TARGET_END', namelist=nl, file=file) for
                    file, nl in zip(['CNTLALL', 'CONTCNTL', 'RECONA', 'SIZES'],
                                    ['NLSTCALL', 'NLSTCALL', 'STSHCOMP', 'STSHCOMP'])]
    if namelist:
        return namelistData  # return the namelist info
    elif inverse:
        return target[namelistData[0]]  # jsut return the value
    else:
        tgt = target[:]
        tgt.extend([0] * (len(tgt) - 6))  # add trailing zeros
        result = {nl: tgt for nl in namelistData}
        return result  # return the namelist info.


def runName(name='abcdz', inverse=False, namelist=False):
    """
    Compute experiment and job id.
    :param name: 5-character UM name
    :param inverse: Default value False. If True compute name from input directory.
    :param namelist: return the namelist used and do no computation.
    :return: experiment and job id from name or name from name['experID']+name['jobID']+name
    """
    # make namelist information.
    jobname_nl = _namedTupClass(var='RUN_JOB_NAME', namelist='NLCHISTO', file='INITHIS')
    exper_nl = _namedTupClass(var='EXPT_ID', namelist='NLSTCALL', file='CNTLALL')
    jobid_nl = _namedTupClass(var='JOB_ID', namelist='NLSTCALL', file='CNTLALL')
    exper2_nl = _namedTupClass(var='EXPT_ID', namelist='NLSTCALL', file='CONTCNTL')
    jobid2_nl = _namedTupClass(var='JOB_ID', namelist='NLSTCALL', file='CONTCNTL')
    if namelist:
        return [jobname_nl, exper_nl, jobid_nl, exper2_nl, jobid2_nl]
    elif inverse:
        return name[exper_nl] + name[jobid_nl]  # wrap experID & jobID to make name
    else:
        sname = str(name)  # convert to string
        return {exper_nl: sname[0:4], jobid_nl: sname[4],
                exper2_nl: sname[0:4], jobid2_nl: sname[4],
                jobname_nl: sname + "000"}  # split name to make experID, jobID and jobnaame


def gravityWave(kay=2e4, inverse=False, namelist=False):
    """
    Compute gravity wave parameters given kay parameter.
    :param kay: value of kay (default is 2x10^4)
    :param inverse:  invert calculation
    :param namelist: return the namelist used and do no computation.
    :return: dict containing kay_gwave and kay_lee_gwave parameters. (or value of kay if inverse set)
    """

    # name list info
    gwave = _namedTupClass(var='KAY_GWAVE', namelist='RUNCNST', file='CNTLATM')
    lee_gwave = _namedTupClass(var='KAY_LEE_GWAVE', namelist='RUNCNST', file='CNTLATM')

    if namelist:
        return [gwave, lee_gwave]  # return the namelist only
    # do the computations.
    elif inverse:
        return kay[gwave]  # extract kay from the info.
    else:  # from MIke's code (in turn from R code)
        gwd_pt = [1e04, 1.5e04, 2e04]
        lee_pt = [1.5e05, 2.25e05, 3e05]
        lee = IDLinterpol(lee_pt, gwd_pt, kay)
        return {gwave: kay, lee_gwave: lee}


def iceAlbedo(alpham=0.5, inverse=False, namelist=False):
    """
    Compute ice albedo values given alpham
    :param alpham: value of alpham (if forward) or dict containing parameters for inverse
    :param inverse: default is False -- compute alpham
    :param namelist -- return namelist used.
    :return:
    """
    # namelist information
    alpham_nl = _namedTupClass(var='ALPHAM', namelist='RUNCNST', file='CNTLATM')
    dtice_nl = _namedTupClass(var='DTICE', namelist='RUNCNST', file='CNTLATM')
    if namelist:
        return [alpham_nl, dtice_nl]
    elif inverse:
        return alpham[alpham_nl]
    else:
        mins = [0.5, 0.57, 0.65]  # alpham values
        maxs = [10., 5., 2.]  # corresponding dtice values
        dtice = IDLinterpol(maxs, mins, alpham)  # from Mike's code.
        return {dtice_nl: dtice, alpham_nl: alpham}


def cloudWater(cw_land=2e-4, inverse=False, namelist=False):
    """
    Compute cw_sea from  cw_land
    :param cw_land: value of cw_land
    :param inverse -- if true do inverse calculation and return cw_land
    :return: returns dict containing cloud_cw_land and cloud_cw_sea values.
    """
    cw_land_nl = _namedTupClass(var='CW_LAND', namelist='RUNCNST', file='CNTLATM')
    cw_sea_nl = _namedTupClass(var='CW_SEA', namelist='RUNCNST', file='CNTLATM')
    if namelist:
        return [cw_land_nl, cw_sea_nl]
    elif inverse:
        return cw_land[cw_land_nl]
    else:
        cwl = [1e-04, 2e-04, 2e-03]
        cws = [2e-05, 5e-05, 5e-04]
        cw_sea = IDLinterpol(cws, cwl, cw_land)
        return {cw_land_nl: cw_land, cw_sea_nl: cw_sea}


def cloudRHcrit(rhcrit=0.7, inverse=False, namelist=False):
    """
    Compute rhcrit on multiple model levels
    :param rhcrit: meta parameter for rhcrit
    :param inverse: default False. If True invert the relationship
    :param namelist: default False. If True return only the namelist information.
    :return: (value of meta parameter if inverse set otherwise
       a dict with cloud_rh_crit containing a list of rh_crit on model levels

    """
    # TODO if necessary -- allow number of levels to be specified??
    rhcrit_nl = _namedTupClass(var='RHCRIT', namelist='RUNCNST', file='CNTLATM')
    if namelist:
        return {rhcrit_nl: None}
    elif inverse:
        return rhcrit[rhcrit_nl][3]
    else:
        cloud_rh_crit = 19 * [rhcrit]
        cloud_rh_crit[0] = max(0.95, rhcrit)
        cloud_rh_crit[1] = max(0.9, rhcrit)
        cloud_rh_crit[2] = max(0.85, rhcrit)
        return {rhcrit_nl: cloud_rh_crit}


def cloudEACF(eacf=0.5, inverse=False, namelist=False):
    """
    Compute array of eacf values for each model level. If inverse set work out what meta-parameter is from array.
    :param eacf: meta-pramater value for eacf
    :param inverse: compute inverse relationship
    :param namelist: default False. If True return only the namelist information.
    :return: either a dict with cloud_eacf being a list of values on each level.
    """
    # TODO if necessary -- allow number of levels to be specified or at least pull out of the model.
    eacf_nl = _namedTupClass(var='EACF', namelist='SLBC21', file='CNTLATM')
    if namelist:
        return [eacf_nl]
    elif inverse:
        return eacf[eacf_nl][0]
    else:
        assert eacf >= 0.5, "eacf seed must be ge 0.5, according to R code but it is %f\n" % eacf
        eacfbl = [0.5, 0.7, 0.8]
        eacftrp = [0.5, 0.6, 0.65]
        five = 5 * [eacf]
        eacf1 = IDLinterpol(eacftrp, eacfbl, eacf)
        cloud_eacf = 19 * [eacf1]
        cloud_eacf[0:5] = five
        cloud_eacf[5] = (2. * eacf + eacf1) / 3.
        cloud_eacf[6] = (eacf + 2. * eacf1) / 3.
        return {eacf_nl: cloud_eacf}


def diffusion(diff_time=12.0, namelist=False, inverse=False):
    """
    Compute arrays of diffusion co-efficients for all   levels.
    :param diff_time: time in hours for diffusion
    :param inverse: optional -- default = False. invert relationship to work our diffusion time
    :param namelist: default False. If True return only the namelist information.
    :return: if inverse set compute diffusion timescale otherwise compute diffusion coefficients.
         returns them as a dict with entries
         diff_coeff -- diffusion coefficient, diff_coeff_q -- diffusion coefficient for water vapour
         diff_power -- power of diffusion, diff_power_q -- power for diffusion.
    """
    # TODO -- if necessary allow levels to be specified or read from namelists. Also read other info on model!
    # NAMELIST information.
    diff_coeff_nl = _namedTupClass(var='DIFF_COEFF', namelist='RUNCNST', file='CNTLATM')
    diff_coeff_q_nl = _namedTupClass(var='DIFF_COEFF_Q', namelist='RUNCNST', file='CNTLATM')
    diff_exp_nl = _namedTupClass(var='DIFF_EXP', namelist='RUNCNST', file='CNTLATM')
    diff_exp_q_nl = _namedTupClass(var='DIFF_EXP_Q', namelist='RUNCNST', file='CNTLATM')
    if namelist:
        return [diff_coeff_nl, diff_coeff_q_nl, diff_exp_nl, diff_exp_q_nl]
    elif inverse:
        powerDiff = diff_time[diff_exp_nl][0] * 2
        diff_time_hrs = diff_fn(diff_time[diff_coeff_nl][0], dyndel=powerDiff, inverse=inverse)
        return diff_time_hrs
    else:
        diff_coeff, diff_coeff_q, diff_exp, diff_exp_q = \
            metaDIFFS(dyndiff=diff_time, dyndel=diff_pwr)
        return {diff_coeff_nl: diff_coeff, diff_coeff_q_nl: diff_coeff_q, diff_exp_nl: diff_exp,
                diff_exp_q_nl: diff_exp_q}


def iceDiff(OcnIceDiff=2.5e-5,namelist=False,inverse=False):
    """
    Generate namelist for NH and SH ocean ice diffusion co-efficient.
    :param OcnIceDiff: Value wanted for ocean ice diffusion coefficient
        Same value will be used for norther and southern hemispheres
        or dict containing parameters for inverse
    :param namelist: If True return only the name list information.
    :param inverse:  If True return the diffusion coefficient in the namelist.
    :return: (by default) the namelist information as a dict.
    """
    iceDiff_nlNH = _namedTupClass(var='EDDYDIFFN', namelist='SEAICENL', file='CNTLOCN')
    iceDiff_nlSH = _namedTupClass(var='EDDYDIFFS', namelist='SEAICENL', file='CNTLOCN')
    if namelist:
        return [iceDiff_nlNH,iceDiff_nlSH]
    elif inverse:
        return OcnIceDiff[iceDiff_nlNH]
    else: # set values
        return {iceDiff_nlNH:OcnIceDiff,iceDiff_nlSH:OcnIceDiff}

def iceMaxConc(iceMaxConc=0.995, namelist=False, inverse=False):
    """
    Generate namelist for NH and SH ocean ice maximum concentration
    :param iceMaxConc: Value wanted for ocean max concentration
        Same value will be used for northerh and southern hemispheres with SH maximum being 0.98
        or dict containing parameters for inverse
    :param namelist: If True return only the name list information.
    :param inverse:  If True return the diffusion coefficient in the namelist.
    :return: (by default) the namelist information as a dict.
    """
    iceMax_nlNH = _namedTupClass(var='AMXNORTH', namelist='SEAICENL', file='CNTLOCN')
    iceMax_nlSH = _namedTupClass(var='AMXSOUTH', namelist='SEAICENL', file='CNTLOCN')
    if namelist:
        return [iceMax_nlNH, iceMax_nlSH]
    elif inverse:
        return iceMaxConc[iceMax_nlNH]
    else:  # set values
        # SH value limited to 0.98
        return {iceMax_nlNH: iceMaxConc, iceMax_nlSH: min(0.98,iceMaxConc)}


def ocnIsoDiff(ocnIsoDiff=1e3,namelist=False,inverse=False):
    """
    Generate namelist for changes to Ocean isopynical diffusion.
    :param ocnIsoDiff: Value for ocean ice diffusion. Will set two values AM0_SI & AM1_SI
       Note these picked by examinign Lettie Roach's generated files.
    :param namelist: If True return the name list information
    :param inverse:  If True invert the relationship from the supplied namelist
    :return: Namelist info as a dict
    """

    ocnDiff_AM0= _namedTupClass(var='AM0_SI', namelist='EDDY', file='CNTLOCN')
    ocnDiff_AM1 = _namedTupClass(var='AM1_SI', namelist='EDDY', file='CNTLOCN')
    if namelist:
        return [ocnDiff_AM0, ocnDiff_AM1]
    elif inverse:
        return ocnIsoDiff[ocnDiff_AM0]
    else:  # set values -- both  to same value
        return {ocnDiff_AM0: ocnIsoDiff, ocnDiff_AM1: ocnIsoDiff}

class HadCM3(ModelSimulation.ModelSimulation):
    """
    HadCM3 class. Sub-class of ModelSimulation.
    It overrides createModelSimulation & setParams.

    """

    ## functions below are for meta-parameters.
    ## I wonder if they should embed information about which namelists they want to change
    ## then they return this information as a dict indexed by a named tupple. The named tupple provides
    ## the information about the name list variable, name list name and file while the value is what gets set.
    ## disadvantage is that functions then are very tightly bound into namelist functionality making it more difficult
    ## to modify it. Advantage is that meta param fns deal with namelist complexity.

    # set of fns below are for converting parameters to multiple namelist values.

    def __init__(self, dirPath, obsNames=None,
                 create=False, refDirPath=None, name=None, ppExePath=None,
                 ppOutputFile=None, runTime=None, runCode=None,  # options for creating new study
                 update=False,  # options for updating existing study
                 verbose=False, parameters={}):
        """
        TODO remove runCode and runTime -- they are really properties of the submission rather than the model itself.
         However, how they are done depends on the model... So a bit tricky!


        Create an instance of HadCM3 class. Default behaviour is to read from dirPath and prohibit updates.
        :param dirPath -- path to directory where model simulation exists or is to be created
        :param create (optional with default False). If True create new directory and populate it.
            Afterwards the ModelSimulation will be readOnly.
            These options should be specified when creating a new study otherwise they are optional and ignored
            :param refDirPath -- reference directory. Copy all files from here into dirPath
            :param name -- name of the model simulation. If not provided will be taken from dirPath
            :param ppExePath --  path to post proessing executable
            :param ppOutputFile -- File name of output of post processing executable. Default is observations.nc
            :param obsNames -- list of observations to be readin. (see readObs())
            :param runTime -- run time in seconds for UM job. If set to None nothing is changed. 
            :param runCode -- code to be used by Job.
        :param update -- allow updates to the simulation information.
        :param verbose -- provide  verbose output. (See individual methods). Default is False.
        : kwargs -- these are parameter and values. 
        :returns initialised object.
        """

        # no parameters should be provided unless create or update provided
        if (len(parameters) > 0) and not (create or update):
            raise ValueError("Provided parameters but not specified create or update")
        # do HadCM3 specific cross checks.

        # should not specifiy ASTART & AINITIAL
        if 'AINITIAL' in parameters and 'ASTART' in parameters:
            raise NameError("DO not specify both AINITIAL and ASTART.")
        # and OINITIAL and OSTART
        if 'OINITIAL' in parameters and 'OSTART' in parameters:
            raise NameError("DO not specify both OINITIAL and OSTART.")
        # call superclass init
        super(HadCM3, self).__init__(dirPath,
                                     obsNames=obsNames, create=create, refDirPath=refDirPath, name=name,
                                     ppExePath=ppExePath,
                                     ppOutputFile=ppOutputFile, parameters=parameters,  # options for creating new study
                                     update=update,  # options for updating existing study
                                     verbose=verbose)
        if create:  # want to create model instance so do creation.
            self.fixClimFCG()  # fix the ClimFGC
            self.modifySubmit(runTime=runTime, runCode=runCode)  # modify the Submit script
            self.modifyScript()  # modify the script
            self.createWorkDir(refDirPath)  # create the work dirctory (and fill it in)

        ## Set up namelist mappings. #TODO add documentation to parameters and have way of model instance reporting on known params.
        # easy case all variables in SLBC21 and which just set the values.
        for var in ['VF1', 'ICE_SIZE', 'ENTCOEF', 'CT', 'ASYM_LAMBDA', 'CHARNOCK', 'G0', 'Z0FSEA',  # atmos stuff
                    'N_DROP_MIN', 'IA_AERO_POWER', 'IA_AERO_SCALE',  # indirect aerosol stuff
                    # model here is n_drop = IA_AERO_SCALE*(1-exp(IA_AERO_POWER*N_AERO)) where N_AERO is no of aerosol drops and n_drop is no of cld drolets.
                    # default values are 'N_DROP_MIN': 3.5E7, 'IA_AERO_POWER': -2.5e-9, 'IA_AERO_SCALE': 3.75E8
                    'CLOUDTAU', 'NUM_STAR', 'L0', 'L1', 'OHSCA', 'VOLSCA', 'ANTHSCA', 'RAD_AIT', 'RAD_ACC'
                    # sulphate params.  CLOUDTAU (1.08E4)  air parcel lifetime in cloud, NUM_STAR (1.0E6) threshold concn of accu mode particles
                    # L0=6.5E-5, Scavenging parameter when S < S_threshold
                    # L1=2.955E-5, Scavenging parameter when S > S_threshold
                    # OHSCA=1.0, -- scaling parameter for the OH field.
                    # VOLSCA=1.0, -- scaling parameter for volcanic SO2 emissions
                    # ANTHSCA=1.0, -- scaling parameter for anthropogenic emissions
                    # RAD_AIT -- radius of aitkin mode droplets -- 24e-9
                    # RAD_ACC -- radius of accum mode doplets -- 95E-9
                    ]:
            self.simpleNamelist(var)
            # ensembleMember has no code to do anything with...
            # current reconfig code pertubs based on runid anyhow.
            # TODO add functionality to UM to deal with ensembleMember

        # Hard case ones where we have a function to run or perturb multiple variables are more complex.
        # for meta-parameter we register function. Functions should return dict indexed by namelist info.
        # I think these should be class methods to save computing them every time we create a model.
        # but overhead is likely small
        # TODO: Convert MetaFn and Namelists to class methods.

        self.registerMetaFn('RUNID', runName, verbose=verbose)  # runid
        self.registerMetaFn('KAY_GWAVE', gravityWave, verbose=verbose)  # gravity wave
        self.registerMetaFn('ALPHAM', iceAlbedo, verbose=verbose)  # ice albedo
        self.registerMetaFn('CW_LAND', cloudWater, verbose=verbose)  # CW_LAND
        self.registerMetaFn('RHCRIT', cloudRHcrit, verbose=verbose)  # RHCRIT meta-param generates array
        self.registerMetaFn('EACF', cloudEACF, verbose=verbose)  # EACF meta-param generates array
        self.registerMetaFn('DYNDIFF', diffusion, verbose=verbose)  # Dynamics Diffusion generates lots of arrays
        self.registerMetaFn('RUN_TARGET', runTarget,
                            verbose=verbose)  # length of simulation -- modifies several namelist vars
        self.registerMetaFn('START_TIME', startTime,
                            verbose=verbose)  # start_Time for run -- modifies several namelist vars
        self.registerMetaFn("SPHERICAL_ICE", sph_ice, verbose=verbose)  # Spherical ice (or not)
        self.registerMetaFn("OcnIceDiff",iceDiff, verbose=verbose) # Ocean-Ice heat diffusion.
        self.registerMetaFn("IceMaxConc", iceMaxConc, verbose=verbose)  # Maximum ice concentration.
        self.registerMetaFn("OcnIsoDiff", ocnIsoDiff, verbose=verbose)  # Maximum ice concentration.
        # add AINITIAL, OINITIAL, ASTART & OSTART in inithist
        for var in ['AINITIAL', 'OINITIAL', 'ASTART', 'OSTART']:  # need slightly special code for these variables.
            # ideas is to use the same basic function which has as argument the parameter.
            # Then use functools.partial to generate function which gets registered.
            fn = functools.partial(initHist_nlcfiles, parameter=var)
            fn.__name__ = var.lower() + '_initHist_nlcfiles'
            self.registerMetaFn(var, fn, verbose=verbose)
        # got some parameters and either creating or updating -- update namelist.
        if len(parameters) > 0 and (create or update):
            self.setReadOnly(False)  # allow modification
            self.setParams(parameters, verbose=verbose, fail=True)  # apply namelist etc
            # deal with START_TIME -- f90nml.patch  fails... and f90nml.read doesn't work as it should
            # need a temp files etc -- TODO: If need this functionality again then warp in a subroutine and try and fix f90nml!
            #
            if 'START_TIME' in parameters:
                RECONA_FILE = os.path.join(self.dirPath, 'RECONA')
                recona = f90nml.read(RECONA_FILE)
                # if f90nml.__version__ != '0.21': # version should be 0.21
                #    raise Exception("Can only work with f90nml version 0.21. Version is %s"%(f90nml.__version__))
                # f90nml is .20 then it appears to truncate the array
                # so fill it in, # general problem here is that array length varies in rather arbitrary ways.
                # the only obvious reference point we have is 405 at fixhs[11] -- which means vn4.5 TODO fix this.
                fixhd = recona['headers']['FIXHD']
                # f90nml (even at vn 0.21) has problems reading in nml in the following format
                # fixhd(12)=xxx -- instead putting the value at posn 0.
                offset = fixhd.index(405) - 11  # if 405 not found will trigger an error.
                # print("patching fixhd in %s is " % RECONA_FILE, repr(fixhd))
                # all this is done because f90nml seems to remember it starts at some posn.
                fixhd[20 + offset] = parameters['START_TIME'][0]
                fixhd[27 + offset] = parameters['START_TIME'][0]
                recona.write(RECONA_FILE, force=True)  # overwrite the existing file

            self.setReadOnly(True)  # make it read only

    def createWorkDir(self, refDirPath, verbose=False):
        """
        Create the workdir and if refDirPath has .astart a& .ostart copy those into created workDir
        :param refDirPath -- name of reference directory
        :param (optional) verbobse -- default False. If True be verbose.
        :return: nada
        """

        workDir = os.path.join(self.dirPath, 'W')
        if not os.path.isdir(workDir): os.makedirs(workDir)  # create the directory
        for f in ['*.astart', '*.ostart']:  # possible start files
            p = glob.glob(os.path.join(refDirPath, f))  # glob them
            if p is not None and len(p) == 1:  # got one
                p = p[0]  # copy it.
                try:
                    shutil.copy(p, workDir)
                    if verbose: print("Copied %s to %s" % (p, workDir))
                except IOError:
                    pass

    def modifyScript(self):
        """
        modify script. 
         set ARCHIVE_DIR to runid/A -- not in SCRIPT??? WOnder where it comes from. Will look at modified script..
         set EXPTID to runid  --  first ^EXPTID=xhdi
         set JOBID to jobid  -- first ^JOBID
         add hook for post-process jobs -- will have script as postProcess.sh # this should be generated by the module that
           handles submission. HadCM3 module, in theory, doesn't really care what machine it is being submitted for. 
        :param name -- the name of the run. 
        :return: 
        """
        # TODO seperate out name and runid.
        runid = self.getParams().get('RUNID',
                                     self.name())  # TODO fix this so that if params passed a strong it does sensible thing with it...
        experID = runid[0:4]
        jobID = runid[4]
        modifystr = '## modified'
        with fileinput.input(os.path.join(self.dirPath, 'SCRIPT'), inplace=1, backup='.bak') as f:
            for line in f:
                # TODO -- resubmission of crun seems to fail. Attempt to fix this vai removing line with
                # . submitchk with ./SUBMIT -- just run the script again. (But
                if re.search(modifystr, line):
                    raise Exception("Already modified Script")
                elif re.match('^EXPTID=', line):
                    print("EXPTID=%s %s" % (experID, modifystr))
                elif re.match('^JOBID=', line):
                    print("JOBID=%s %s" % (jobID, modifystr))
                elif re.match('MESSAGE="Run .* finished. Time=`date`"', line):
                    print('MESSAGE="Run %s#%s finished. Time=`date`"' % (experID, jobID))
                elif re.match('MY_DATADIR=', line):  # fix MY_DATADIR
                    print("MY_DATADIR=%s %s" % (self.dirPath, modifystr))
                # now find and replace all the DATADIR/$RUNID stuff.
                elif r'DATADIR/$RUNID/' in line:
                    line = line.replace('DATADIR/$RUNID/', 'DATADIR/')
                    print(line[0:-1], modifystr)

                # put marker in for post process script
                # this currently set up to go right at the end
                # but really want it to go when run has finished -- which is sensitive to NRUN/CRUN.
                #  SOmewhere near line 598
                elif re.match(r'exit \$RCMASTER', line) and f.filelineno() > 500:  # near the end...
                    ppModifyMark = '# =insert post-process script here= # '
                    print("""
                       if [ $RCMASTER -eq 0 ] {mark}
                          then {mark}
                          echo 'Worked so doing stuff..' {mark}
                          {modifyMark} {mark} 
                        fi # close block {mark} 
                        """.format(mark=modifystr, modifyMark=ppModifyMark))
                    print(line[0:-1])
                    # store the mark and the file for use by later processing
                    self.postProcessMark = ppModifyMark
                    self.postProcessFile = os.path.join(self.dirPath, 'SCRIPT')
                else:
                    print(line[0:-1])  # remove newline

    def modifySubmit(self, runTime=None, runCode=None):
        """
        Modify the Submit script for HadCM3
        Change to SUBMIT
          set RUNID to runid in SUBMIT   == first ^export RUNID=
          set JOBDIR to dirPath   -- first ^JOBDIR=
          set CJOBN to runid in SUBMIT  -- first ^CJOBN=
          set MY_DATADIR to dirPath
          set DATAW and DATAM to $MY_DATADIR/W & $MY_DATADIR/M respectively.
        :param runTime: default None -- if not None then specify time in seconds for model job. 
        :param runCode: default None -- if not Note then this is the project code for the model job.
        
        :return: nada
        """
        # first work out runID
        runid = self.getParams().get('RUNID',
                                     self.name())  # TODO fix this so that if params passed a string it does sensible thing with it...
        # modify submit
        maxLineNo = 75  # maximum lines to modify
        # modify SUBMIT
        modifyStr = '## modified'
        with fileinput.input(os.path.join(self.dirPath, 'SUBMIT'), inplace=1, backup='.bakR') as f:
            for line in f:
                # need to make these changes only once...
                if re.search(modifyStr, line):
                    raise Exception("Already modified SUBMIT")
                elif re.match('export RUNID=', line) and f.filelineno() < maxLineNo:
                    print("export RUNID=%s %s" % (runid, modifyStr))
                elif re.match('MY_DATADIR=', line) and f.filelineno() < maxLineNo:
                    print("MY_DATADIR=%s %s" % (self.dirPath, modifyStr))
                # now find and replace all the DATADIR/$RUNID stuff.
                elif r'DATADIR/$RUNID/' in line:
                    line = line.replace('DATADIR/$RUNID/', 'DATADIR/')
                    print(line[0:-1], modifyStr)
                elif re.match('JOBDIR=', line) and f.filelineno() < maxLineNo:
                    print("JOBDIR=%s %s" % (self.dirPath, modifyStr))
                elif re.match('CJOBN=', line) and f.filelineno() < maxLineNo:
                    print("CJOBN=%s %s" % (runid + '000', modifyStr))
                elif (runTime is not None) and re.match('[NC]RUN_TIME_LIMIT=', line):  # got a time specified
                    l2 = line.split('=')[0]  # split line at =
                    print(l2 + '=%d ' % (runTime) + modifyStr)  # add on the time
                elif (runCode is not None) and re.match('ACCOUNT=', line):  # got a project code specified
                    l2 = line.split('=')[0]  # split line at =
                    print(l2 + '=%s ' % (runCode) + modifyStr)  # add on the time
                else:
                    print(line[0:-1])  # remove trailing newline

    def simpleNamelist(self, var, nl='SLBC21', nlFile='CNTLATM'):
        """
        set up single variable with name var.lower() in namelist nl in file nlFile
        :param var: Name of variable
        :param (optional) nl -- name of namelist. Default is 'slbc21'
        :param (optional) nlFile -- name of file for namelist. Default is 'CNTLATM'
        :return: none
        """
        self.genVarToNameList(var, nameListVar=var.upper(), nameListName=nl, nameListFile=nlFile)

    def fixClimFCG(self):
        """
        Fix problems with the CLIM_FCG_* namelists so they can be parsed by f90nml. 
        Generates CNTLATM.bak and modifies CNTLATM
        :return: None
        """
        with fileinput.input(os.path.join(self.dirPath, 'CNTLATM'), inplace=1, backup='.bak') as f:
            for line in f:
                if re.search(r'CLIM_FCG_.*\(1,', line):
                    line = line.replace('(1,', '(:,')  # replace the leading 1 with :
                print(line[0:-1])  # remove trailing newline.

    #

    def setParams(self, params, addParam=True, write=True, verbose=False, fail=True):
        """
        Set the parameter values and write them to the configuration file
        and modify the parameters in the current directory. Calls the superclass to do standard stuff first then
         uses existing code to modify parameters in HadCM3 namelists.
        :param params -- dictionary (or ordered dict or pd.Series) of the parameter values
        :param addParam (default False) -- if True add to existing parameters
        :param write (default True) -- if True update configuration file.
        :param verbose (default False) -- if True provide more verbose output
        :param fail (default True) -- if True fail if parameter not defined.
        :return:
        """

        if addParam == False:
            # need to find a way of resetting namelist files.
            # one option would be to copy the namelist files from the refdir. That would require working out all the files
            # that is moderately tricky and not yet needed. So raise anNotImplementedError if tried.
            raise NotImplementedError("Not yet implemented addParam")

        super(HadCM3, self).setParams(params, addParam=addParam, write=write,
                                      verbose=verbose)  # use super classs setParams
        # remove ensembleMember from the params -- we have no
        #  namelist for it. writeNameList complains if parameter provided 
        # and no translation function.
        try:
            eM = params.pop('ensembleMember')
        except KeyError:
            pass

        self.writeNameList(verbose=verbose, fail=fail, **params)  # generate/update namelists.

    def submit(self):
        """
        
        :return: path to script to run to submit the model 
        """

        return os.path.join(self.dirPath, 'SUBMIT')

    def scriptPostprocess(self,modifyStr = None):
        """
        add post process info to the script
        :return:
        """

        # Modify SCRIPT and remove some text near bottom..code comes from modifyScript.
        replaceRCMASTER = True
        nskip = 0
        with fileinput.input(os.path.join(self.dirPath, 'SCRIPT'), inplace=1, backup='.bakR') as f:
            for line in f:
                line = line[0:-1]  # remove tr
                # deal with need to put marker in for post process script -- see modifyScript
                # where this code comes from.
                if ('$RCMASTER -eq 0 ] ##' in line) and (
                        f.filelineno() > 500) and replaceRCMASTER:  # near the end. Code VERY fragile
                    replaceRCMASTER = False
                    ppModifyMark = '# =insert post-process script here= # '
                    print("""
                                   if [ $RCMASTER -eq 0 ] {mark}
                                      then {mark}
                                      echo 'Worked so doing stuff..' {mark}
                                      {modifyMark} {mark} 
                                    fi # close block {mark} 
                                    """.format(mark=modifyStr, modifyMark=ppModifyMark))
                    # print above seems to replace multiple lines
                    # store the mark and the file for use by later processing
                    self.postProcessMark = ppModifyMark
                    self.postProcessFile = os.path.join(self.dirPath, 'SCRIPT')
                    nskip = 5  # want to skip some lines
                elif nskip > 0:  # got some lines to skip so don't write them out
                    nskip -= 1  # but decrement nskip.
                else:
                    print(line)  # print line without newline

    def continueSimulation(self,minimal=False):
        """
        Update model so that it is a CRUN and STEP=4
          also modify post-processing info
        :param minimal -- do not **modify** post-processing info if True.
        :return:
        """
        # modify submit
        # modify SUBMIT

        # call superclass continue method
        super(HadCM3, self).continueSimulation(minimal=minimal)
        modifyStr = '## modifiedContinue'
        with fileinput.input(os.path.join(self.dirPath, 'SUBMIT'), inplace=1, backup='.bakR') as f:
            for line in f:
                line = line[0:-1]  # remove trailing newline
                if re.match('^TYPE=NRUN', line):  # DEAL with NRUN
                    line = line.replace('NRUN', 'CRUN', 1) + modifyStr
                    print(line)
                elif re.match('^STEP=1', line):  # Deal with STEP=1
                    print(line.replace('STEP=1', 'STEP=4', 1) + modifyStr)
                else:
                    print(line)  # remove trailing newline

        if not minimal:
           self.scriptPostprocess(modifyStr=modifyStr)

    def restartSimulation(self,minimal=False):
        """
        Update model so that it is a NRUN and STEP=4
          also modify post-processing info
        :param minimal -- do not **modify** post-processing info if True.
        :return:
        """
        # modify submit
        # modify SUBMIT

        # call superclass restart method
        super(HadCM3, self).restartSimulation(minimal=minimal) # superclass does some stuff with config
        modifyStr = '## modifiedRestart'
        with fileinput.input(os.path.join(self.dirPath, 'SUBMIT'), inplace=1, backup='.bakR') as f:
            for line in f:
                line = line[0:-1]  # remove trailing newline
                if re.match('^TYPE=CRUN', line):  # DEAL with NRUN
                    line = line.replace('CRUN', 'NRUN', 1) + modifyStr
                    print(line)
                elif re.match('^STEP=1', line):  # Deal with STEP=1
                    print(line.replace('STEP=1', 'STEP=4', 1) + modifyStr)
                else:
                    print(line)  # remove trailing newline

        if not minimal:
            self.scriptPostprocess(modifyStr=modifyStr)


    def perturb(self, verbose=False, params=None):
        """
        Perturb the model **namelist** so as to generate a small perturbation.
        :param verbose -- If True be verbose
        :params -- if not None then set those parameters in the namelist.
        """



        perturbList = super(HadCM3, self).perturb(params=params, verbose=verbose)
        # store that we actually perturbed something. Note that superclass runs restart
        if params is not None: # got some parameters to perturb
            self.setReadOnly(readOnly=False)
            self.writeNameList(verbose=verbose, fail=True, **params)  # generate/update namelists.
            self.setReadOnly(readOnly=False)
        return perturbList

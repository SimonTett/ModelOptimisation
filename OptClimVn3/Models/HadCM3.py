from __future__ import annotations

import logging

import numpy as np

from Models.Model import Model, register_param
import importlib.resources
from Models.namelist_var import namelist_var
import pathlib
import datetime


def parse_isoduration(s):
    """ Parse a str ISO-8601 Duration: https://en.wikipedia.org/wiki/ISO_8601#Durations
    Originally copied from:
    https://stackoverflow.com/questions/36976138/is-there-an-easy-way-to-convert-iso-8601-duration-to-timedelta
    Though could use isodate library but trying to avoid dependencies and isodate does not look maintained.
    :param s: str to be parsed. If not a string starting with "P" then ValueError will be raised
    :return: 6 element list [YYYY,MM,DD,HH,mm,SS.ss] which is suitable for the UM namelists
    """

    def get_isosplit(s, split):
        if split in s:
            n, s = s.split(split, 1)
        else:
            n = '0'
        return n.replace(',', '.'), s  # to handle like "P0,5Y"

    if not isinstance(s, str):
        raise ValueError("ISO 8061 demands a string")
    if s[0] != 'P':
        raise ValueError("ISO 8061 demands durations start with P")
    s = s.split('P', 1)[-1]  # Remove prefix

    split = s.split('T')
    if len(split) == 1:
        sYMD, sHMS = split[0], ''
    else:
        sYMD, sHMS = split  # pull them out

    durn = []
    for split_let in ['Y', 'M', 'D']:  # Step through letter dividers
        d, sYMD = get_isosplit(sYMD, split_let)
        durn.append(float(d))

    for split_let in ['H', 'M', 'S']:  # Step through letter dividers
        d, sHMS = get_isosplit(sHMS, split_let)
        durn.append(float(d))

    return durn


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
      Which, by definition, are specific to HadCM3
    """


    ### Stuff for Namelists.

    def atmos_time_step(self):
        """
        Get timestep info and return it in seconds. (Model internally uses steps per day (I think))
        :return: timestep in seconds
        """
        timestep_nl = namelist_var(filepath=pathlib.Path('CNTLATM'), namelist='NLSTCATM', nl_var='A_ENERGYSTEPS')
        steps_per_day = timestep_nl.read_value(dirpath=self.model_dir)
        timestep = 3600.0*24/steps_per_day
        return timestep

    def dx_dy(self, expect=None,units=None):
        """
        Work out the dx and dy values for the model (in radians).
        :param expect: two element np.array of expected values. First element is dx, and second element is dy
        :param units: units to return dx and dy in. None (radians), 'degrees' or 'meters'.
        :return: dx and dy values in specified units.
        """
        dx_nl,dy_nl = (namelist_var(filepath=pathlib.Path('SIZES'), nl_var=var,namelist='NLSIZES')
                       for var in ['ROW_LENGTH', 'P_ROWS'])
        result = np.array([2/dx_nl.read_value(dirpath=self.model_dir),
                           1/(dy_nl.read_value(dirpath=self.model_dir)-1)])*np.pi

        # do some unit conversions
        RADIUS = 6.37123e06  # radius of earth
        if units is None:
            pass
        elif units == 'degrees':
            result=np.rad2deg(result)
        elif units == 'meters':
            result *= RADIUS
        else:
            raise ValueError(f"Unknown units {units}")
        if (expect is not None) and np.any(result != expect):
            raise ValueError(f"Expected {expect} values but got {result}")
        return result

    def nlev(self, expect=None):
        """
        Work out the number of levels by reading the p_levels
        :param expect If not None if nlev does not equal expect an error will be raised
        :return: the number of levels.
        """

        lev_nl = namelist_var(filepath=pathlib.Path('SIZES'), nl_var='P_LEVELS', namelist='NLSIZES')
        no_levels = lev_nl.read_value(dirpath=self.model_dir)
        if (expect is not None) and (no_levels != expect):
            raise ValueError(f"Expected {expect} levels but got {no_levels}")

        return no_levels

    @register_param('EACF')
    def cloudEACF(self, eacf, inverse=False):
        """
        Compute array of eacf values for each model level. If inverse set work out what meta-parameter is from array.
        :param eacf: meta-parameter value for eacf. If None then inverse will be assumed regardless of actual value
        :param inverse: compute inverse relationship.  If namelist value not found then default value of 0.5 will be returned.
        """

        self.nlev(expect=19)  # test have 19 levels
        eacf_nl = namelist_var(filepath=pathlib.Path('CNTLATM'), namelist='SLBC21', nl_var='EACF')
        if (inverse or (eacf is None)):
            eacf_val = eacf_nl.read_value(dirpath=self.model_dir, default=0.5)
            return eacf_val[0]
        else:
            # add check for number of levels here.
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
    def sph_ice(self, sphIce=True, inverse=False):
        """
        Set/read namelists for spherical/non-spherical ice.
        :param sphIce: True if want spherical ice (which is default)
        :param inverse: Return (from namelists) if spherical ice.SPAHERICAL
        :return:  returns nl/tuple pair or, if inverse, the value of sphIce
        """
        nl = [namelist_var(nl_var='I_CNV_ICE_LW', namelist='R2LWCLNL', filepath=pathlib.Path('CNTLATM')),
              # default 1, perturb 7
              namelist_var(nl_var='I_ST_ICE_LW', namelist='R2LWCLNL', filepath=pathlib.Path('CNTLATM')),
              # default 1, perturb 7
              namelist_var(nl_var='I_CNV_ICE_SW', namelist='R2SWCLNL', filepath=pathlib.Path('CNTLATM')),
              # defaault 3, perturb 7
              namelist_var(nl_var='I_ST_ICE_SW', namelist='R2SWCLNL', filepath=pathlib.Path('CNTLATM')),
              # default 2, perturb 7
              ]
        values_sph = [1, 1, 3, 2]
        values_nonSph = [7, 7, 7, 7]

        if inverse:  # inverse -- so  extract value and test for consistency
            # check all is OK
            sph = (nl[0].read_value(dirpath=self.model_dir, default=1) == 1)
            if sph:
                values = values_sph
            else:
                values = values_nonSph
            # check  namelist values are consistent
            for n, v in zip(nl, values):
                if n.read_value(dirpath=self.model_dir, default=v) != v:
                    raise Exception("Got %d when expected %d" % (sphIce[n], v))
            return sph

        # set values
        if sphIce:
            values = values_sph
        else:
            values = values_nonSph
        result = list(zip(nl, values))  # return namelist info.
        return result

    @register_param("START_TIME")
    def start_time(self, time_input: str | list, inverse=False):
        """
        :param time:  start time as ISO format  string.
        :param inverse: Do inverse calculation and return as ISO format string
        :return:  namelist info and values as array. (or if inverse set return the start time)
        """
        # It would be really nice to use cftime rather than datetime
        # but I don't think there is a from isoformat method for cftime..

        # set up all the var/namelist/file info.
        # Need to set MODEL_BASIS_TIME in CNTLALL/&NLSTCALL & CONTCNTL/&NLSTCALL
        namelistData = [namelist_var(nl_var='MODEL_BASIS_TIME', namelist=nl, filepath=pathlib.Path(file)) for
                        file, nl in zip(['CNTLALL', 'CONTCNTL'],
                                        ['NLSTCALL', 'NLSTCALL'])]
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

    def time_delta(self, duration, inverse=False, runTarget=True):
        """
        :param input:  Duration as string in ISO 8061 format (PnYnMnDTnHnMn.nnS)
        :param inverse: Do inverse calculation and return  Duration as string in ISO 8061 format (PnYnMnDTnHnMn.nnS)
        :param runTarget: namelist info suitable for runTarget. (default). If False namelist info suitable for resubInterval
        :return: namelist info and values as array. (or if inverse set return the run target )
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

        if inverse:
            durn = namelistData[0].read_value(dirpath=self.model_dir)
            logging.warning("No inverse conversion to ISO timedelta notation")
            # check rest are OK
            for nl in namelistData[1:]:
                d2 = nl.read_value(dirpath=self.model_dir)
                if d2 != durn:
                    raise ValueError(f"Durations differ between {nl} and {namelistData[0]}")
            return durn  # just return the value as a 6 element list.
        else:
            durn = parse_isoduration(duration)
            # verify that len of target is >= 1 and <= 6 and if not raise error.
            if len(durn) != 6:
                raise Exception(f"Durn {durn} should have 6 elements. Computed from {duration} ")
            result = [(nl, durn) for nl in namelistData]
            return result  # return the namelist info.

    @register_param("RUN_TARGET")
    def run_target(self, target_input, inverse=False):
        """
        set runTarget -- see timeDelta for documentation.
        :param target_input: target length of run as ISO string
        :param inverse: Return run_target
        :return: Return list of nl,value pairs or run_target
        """
        return self.time_delta(target_input, inverse=inverse)

    @register_param('RESUB_INTERVAL')
    def resub_interval(self, interval_input, inverse=False):
        """
        Set resubmit durations -- see timeDelta for documentation
        :param interval_input:
        :param inverse:
        """
        return self.time_delta(interval_input, inverse=inverse, runTarget=False)

    @register_param("NAME")
    def run_name(self, name, inverse=False):
        """
        Compute experiment and job id.
        :param name: 5-character UM name
        :param inverse: Default value False. If True compute name from input directory.
        :param namelist: return the namelist used and do no computation.
        :return: experiment and job id from name or name from name['experID']+name['jobID']+name
        """
        # make namelist information.
        jobname_nl = namelist_var(nl_var='RUN_JOB_NAME', namelist='NLCHISTO', filepath=pathlib.Path('INITHIS'))
        exper_nl = namelist_var(nl_var='EXPT_ID', namelist='NLSTCALL', filepath=pathlib.Path('CNTLALL'))
        jobid_nl = namelist_var(nl_var='JOB_ID', namelist='NLSTCALL', filepath=pathlib.Path('CNTLALL'))
        exper2_nl = namelist_var(nl_var='EXPT_ID', namelist='NLSTCALL', filepath=pathlib.Path('CONTCNTL'))
        jobid2_nl = namelist_var(nl_var='JOB_ID', namelist='NLSTCALL', filepath=pathlib.Path('CONTCNTL'))
        if inverse:
            return exper_nl.read_value(self.model_dir) + jobid_nl.read_value(
                self.model_dir)  # wrap experID & jobID to make name
        else:
            sname = str(name)  # convert to string
            if len(sname) != 5:
                raise ValueError("HadCM3 expects 5 character names not {sname}")
            result = [(exper_nl, sname[0:4]), (jobid_nl, sname[4]),
                      (exper2_nl, sname[0:4]), (jobid2_nl, sname[4]),
                      (jobname_nl, sname + "000")]  # split name to make experID, jobID and jobname
            return result

    @register_param("CW")
    def cloud_water(self, cw_land, inverse=False):
        """
        Compute cw_sea from  cw_land
        :param cw_land: value of cw_land
        :param inverse -- if true do inverse calculation and return cw_land
        :return: returns dict containing cloud_cw_land and cloud_cw_sea values.
        """
        cw_land_nl = namelist_var(nl_var='CW_LAND', namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
        cw_sea_nl = namelist_var(nl_var='CW_SEA', namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
        if inverse:
            return cw_land_nl.read_value(dirpath=self.model_dir)
        else:
            cwl = [1e-04, 2e-04, 2e-03]
            cws = [2e-05, 5e-05, 5e-04]
            cw_sea = IDLinterpol(cws, cwl, cw_land)
            return [(cw_land_nl, cw_land), (cw_sea_nl, cw_sea)]

    @register_param("KAY_GWAVE")
    def gravity_wave(self, kay, inverse=False):
        """
        Compute gravity wave parameters given kay parameter.
        :param kay: value of kay
        :param inverse:  invert calculation
        :return: list  containing kay_gwave and kay_lee_gwave parameters. (or value of kay if inverse set)
        """

        # name list info
        gwave, lee_gwave = (namelist_var(nl_var=var, namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
                            for var in ['KAY_GWAVE', 'KAY_LEE_GWAVE'])

        if inverse:
            v = gwave.read_value(dirpath=self.model_dir)
            return v
        else:  # from MIke's code (in turn from R code)
            gwd_pt = [1e04, 1.5e04, 2e04]
            lee_pt = [1.5e05, 2.25e05, 3e05]
            lee = IDLinterpol(lee_pt, gwd_pt, kay)
            return [(gwave, kay), (lee_gwave, lee)]

    @register_param("ALPHAM")
    def ice_albedo(self, alpham, inverse=False):
        """
        Compute ice albedo values given alpham
        :param alpham: value of alpham (if forward)
        :param inverse: default is False -- compute alpham
        :return:
        """
        # namelist information
        alpham_nl, dtice_nl = (namelist_var(nl_var=var, namelist='RUNCNST', filepath=pathlib.Path('CNTLATM')) for var in
                               ['ALPHAM', 'DTICE'])
        if inverse:
            return alpham_nl.read_value(dirpath=self.model_dir)
        else:
            mins = [0.5, 0.57, 0.65]  # alpham values
            maxs = [10., 5., 2.]  # corresponding dtice values
            dtice = IDLinterpol(maxs, mins, alpham)  # from Mike's code.
            return [(dtice_nl, dtice), (alpham_nl, alpham)]

    ## STuff for diffusion calculation. Rather ugly code. Probably not worth doing much with as diffusion does
    ## not seem that important!
    def diff_fn(self,dyndiff, dyndel=6,  inverse=False):
        """
        Support Function to compute diff  coefficient
        :param dyndiff: diffusion value in hours.
        :param dyndel: order of diffusion. 6 is default corresponding to 3rd order.
        :param inverse (default False) If True return inverse calculation.
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
    def diffusion(self, diff_time, inverse=False):
        """
        Compute arrays of diffusion co-efficients for all   levels.
        :param diff_time: time in hours for diffusion
        :param inverse: If True invert relationship to work our diffusion time
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
        if inverse:
            powerDiff = diff_exp_nl.read_value(dirpath=self.model_dir)[0] * 2
            diff_time_hrs = self.diff_fn(diff_coeff_nl.read_value(dirpath=self.model_dir)[0], dyndel=powerDiff,
                                    inverse=inverse)
            return diff_time_hrs
        else:
            diff_pwr = 6 # assuming 3rd order.
            diff_coeff, diff_coeff_q, diff_exp, diff_exp_q = \
                self.metaDIFFS(dyndiff=diff_time, dyndel=diff_pwr)
            return [(diff_coeff_nl, diff_coeff), (diff_coeff_q_nl, diff_coeff_q),
                    (diff_exp_nl, diff_exp), (diff_exp_q_nl, diff_exp_q)]

    @register_param("RHCRIT")
    def cloudRHcrit(self, rhcrit, inverse=False):
        """
        Compute rhcrit on multiple model levels
        :param rhcrit: meta parameter for rhcrit
        :param inverse: default False. If True return current value of rhcrit
        :return: namelist, value.

        """

        self.nlev(expect=19)
        rhcrit_nl = namelist_var(nl_var='RHCRIT', namelist='RUNCNST', filepath=pathlib.Path('CNTLATM'))
        if inverse:
            cloud_rh_crit = rhcrit_nl.read_value(dirpath=self.model_dir, default=0.7)
            rhcrit= cloud_rh_crit[3]
            expected = 19 * [rhcrit]
            for it,v in enumerate([0.95,0.9,0.85]):
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
    def iceDiff(self, OcnIceDiff, inverse=False):
        """
        Generate namelist for NH and SH ocean ice diffusion coefficients.
        :param OcnIceDiff: Value wanted for ocean ice diffusion coefficient
            Same value will be used for northern and southern hemispheres
        :param inverse:  If True return the diffusion coefficient in the namelist.
        :return: (by default) the namelist/value information as list
        """
        iceDiff_nlNH, iceDiff_nlSH = (namelist_var(nl_var=var, namelist='SEAICENL', filepath=pathlib.Path('CNTLOCN'))
                                      for var in ['EDDYDIFFN', 'EDDYDIFFS'])
        if inverse:
            v = iceDiff_nlNH.read_value(dirpath=self.model_dir, default=2.5e-5)
            v_sh = iceDiff_nlSH.read_value(dirpath=self.model_dir, default=2.5e-5)
            if v != v_sh:
                raise ValueError(
                    "Ocean ice diffusion coefficient is not the same for northern and southern hemispheres")
            return v
        else:  # set values
            return [(iceDiff_nlNH, OcnIceDiff), (iceDiff_nlSH, OcnIceDiff)]

    @register_param("MAX_ICE")
    def ice_max_conc(self, iceMaxConc, inverse=False):
        """
        Generate namelist for NH and SH ocean ice maximum concentration
        :param iceMaxConc: Value wanted for ocean max concentration
            Same value will be used for northern and southern hemispheres with SH maximum being 0.98
        :param inverse:  If True return the diffusion coefficient in the namelist.
        :return: nl/values to set the values, if inverse return the NH value.
        """
        iceMax_nlNH, iceMax_nlSH = (namelist_var(nl_var=var, namelist='SEAICENL', filepath=pathlib.Path('CNTLOCN'))
                                    for var in ['AMXNORTH', 'AMXSOUTH'])
        if inverse:
            v = iceMax_nlNH.read_value(dirpath=self.model_dir, default=0.995)
            v2 = iceMax_nlSH.read_value(dirpath=self.model_dir, default=0.995)
            if min(0.98, v) != v2:
                raise ValueError(f"SH Ocean ice maximum concentration = {v2} not {min(0.98, v)}")
            return v
        else:  # set values
            # SH value limited to 0.98
            return [(iceMax_nlNH, iceMaxConc), (iceMax_nlSH, min(0.98, iceMaxConc))]

    @register_param("OCN_ISODIFF")
    def ocnIsoDiff(self, ocnIsoDiff, inverse=False):
        """
        Generate namelist for changes to Ocean isopycnal diffusion.
        :param ocnIsoDiff: Value for ocean ice diffusion. Will set two values AM0_SI & AM1_SI
           Note these picked by examining Lettie Roach's generated files.
        :param inverse:  If True invert the relationship from the supplied namelist
        :return: namelist/values to be set or the namelist value.
        """
        ocnDiff_AM0, ocnDiff_AM1 = (namelist_var(nl_var=var, namelist='EDDY', filepath=pathlib.Path('CNTLOCN'))
                                    for var in ['AM0_SI', 'AM1_SI'])
        if inverse:
            v = ocnDiff_AM0.read_value(dirpath=self.model_dir, default=1e3)
            v2 = ocnDiff_AM1.read_value(dirpath=self.model_dir, default=1e3)
            if v != v2:
                raise ValueError(f"Ocean isopycnal diffusion coefficients differ")
            return v
        else:  # set values -- both  to same value
            return [(ocnDiff_AM0, ocnIsoDiff), (ocnDiff_AM1, ocnIsoDiff)]


traverse = importlib.resources.files("Models")
with importlib.resources.as_file(traverse.joinpath("parameter_config/HadCM3_Parameters.csv")) as pth:
    HadCM3.update_from_file(pth, duplicate=True)

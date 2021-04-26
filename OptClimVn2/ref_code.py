# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:41:18 2015
Reference code -- old versions from Kuniko. Kept to allow checking...
@author: stett2
"""
import sys
from datetime import datetime
import numpy as np
from numpy import zeros, array, ones, dot, linalg, identity, diag, amax, amin
from numpy import math, sqrt, fabs, transpose, save, savetxt
from scipy.stats import chi2

version = '$Id: ref_code.py 671 2018-07-29 13:34:32Z stett2 $'  # subversion magic


# see http://svnbook.red-bean.com/en/1.4/svn.advanced.props.special.keywords.html


def rangeAwarePerturbations(baseVals, parLimits, steps, trace=False):
    """
    Applies desired perturbations (steps) to the initial set of parameter
    values to return the parameter sets required for the GaussNewton phase
    of an iteration.

    The perturbations are towards the centre of the valid range for each
    parameter

    Inputs

         baseVals: array of parameters for the first of the GN runs (unperturbed)

         parLimits: max,min as read from JSON into 2-D array,
                    and passed through to here unaltered

         steps: array of stepsizess for each parameter.

         trace: Optional argument with default value False.
                Boolean -- True to get output, False no output

    Note inputs are all arrays, with data in order of the parameter list.

    Returns
          y: array defining the next set of parameter values
             with dimensions (nparameters, nparameters+1)
    """
    if trace:
        print("in rangeAwarePerturbations from baseVals")
        print(baseVals)

    npar = len(baseVals)
    nruns = npar + 1

    y = np.zeros((npar, nruns))

    # derive the centre-of-valid-range

    centres = (parLimits[:, 0] + parLimits[:, 1]) * 0.5

    if trace:
        print("steps", steps)
        print("parLimits", parLimits)
        print("centres", centres)

    # set "base run" values - those being perturbed one per run,after run first run.

    for j in range(nruns):  # Think this could be done with y=np.array(baseVals)
        y[:, j] = baseVals[:]

    # apply step so it goes in direction of the mid-range for the relevant parameter

    for j in range(npar):
        # parameter j, in run j+1
        if y[j, j + 1] <= centres[j]:
            y[j, j + 1] += abs(float(steps[j]))
        else:
            y[j, j + 1] -= abs(float(steps[j]))

    if trace:
        print("leaving rangeAwarePerturbations with derived perturbed runs:")
        print(y)

    return y


def regularize_cov_ref(covariance, cond_number, initial_scale=1e-4, trace=False):
    '''
    Applys Tikinhov regularisation to covariance matrix so its inverse can be sensibly computed. This sometimes called ridge regression.
    Params: covariance - a NxN symmetric matrix whose diagonal values will be modified.
            cond_number -- the target condition number relative to the condition number of the diagonal matrix
            initial_scale -- initial scaling on the diagonal elements (default value 1e-4)
            trace -- provide some diagnostics when True (default is False)

   Algorithm: The scaling is doubled each iteration until the condition
   number is less than target value. Target value is cond_number*cond(diagonal covariance)

    '''
    diag_cov = np.diag(covariance)
    con = np.linalg.cond(covariance)
    tgt_cond = cond_number * np.linalg.cond(
        np.diag(diag_cov))  # how much more than pure diagonal is target condition number

    scale_diag = initial_scale  ## initial scaling#
    reg_cov = covariance  # make sure got a regularised covariance to return.
    ## AS python does ptr copy this doesn't cost much..
    while (con > tgt_cond):  ## iterate until condition number <= tgt condition number
        scale_diag = scale_diag * 2.0
        reg_cov = covariance + np.diag(scale_diag * diag_cov)
        con = np.linalg.cond(reg_cov)
        if (scale_diag > 10):
            print("failed to regularize matrix. scale_diag,con,tgt_cond = ", scale_diag, con, tgt_cond)
            exit()
    ##  end of regularisation loop.
    if trace:
        print("Used Tikinhov regularisation with scale = ", scale_diag)
        print("Condition #  = ", con)

    return reg_cov


### end of regularize_cov

def calcErr_ref(s_UM_valueT, s_obs, covT, use_constraint, coef1, coef2, nsize, nloop, ioffset):
    """"
    calculate error for simulated observed values given targets
    and add anything else needed.
    NOTE from SFBT: PLEASE DOCUMENT ALL INPUTS & OUTPUTS TO FUNCTION and
    how this routine works. REFERENCE routine corresponding to Kuniko's
    original code. New code will replace this but keeping this version for
    testing purposes.
    """

    f = zeros(nsize)  # nsize=na

    # constraint
    f_constraint = zeros(nsize)
    nObs = s_UM_valueT.shape[0]
    for i in range(nloop):  # nloop=k-n-1
        UMNEW = s_UM_valueT[:, ioffset + i]  # ioffset=n+1
        FNEW = UMNEW - s_obs
        if len(covT) == 0:
            f[i] = sum(FNEW ** 2)
        else:
            FNEWT = transpose(FNEW)
            FNEWTcovT = dot(FNEWT, covT)
            f[i] = dot(FNEWTcovT, FNEW)
        # constraint
        f_constraint[i] = coef1 * f[i] + coef2 * use_constraint[ioffset + i] * use_constraint[ioffset + i]

    ##   err = sqrt(coef1*f) # in error
    err = np.sqrt(f / nObs)  # corrected value
    err_constraint = np.sqrt(f_constraint)

    #   err=sqrt(array)

    return err, err_constraint


def doGaussNewton_ref(param_value, param_range, UM_value, obs, cov, scalings, olist,
                      constraint, studyJSON, trace=False):
    """ 
    Reference version of doGaaussNewton -- Kuniko's orginal version. This allows 
    Easy comparision between Kuniko's original code and re-enginnered code.
    
    the function doGaussNewton_ref does the  calculation for the n+1 initial runs in an iteration,
    and can be invoked from (usually) the class GaussNewton, using data from studies, or from
    test python scripts for arbitrary inputs
 
    :param param_value: array of parameter values
    :param param_range: array of max/min of range for each parameter in order
    :param UM_value: array of simulated observables
    :param obs: array of target values
    :param cov: covariance array (header defines the observables in use) TODO inconsistent with use of constraint_target in JSON file?
    :param scalings: observables are related arrays get scaled before statistical analysis
    :param olist: name of the observables - probabaly unused except perhaps in diagnostic
    :param constraint: TODO what is this?? 
    :param studyJSON The complete dictionary from the study's JSON of which
                     this function might lift simple values to give extensibility. 
    Other arguments have had some manipulation in reading the JSON (scalings, slist... to give correspondence across arrays)
    :param trace: turn on/off trace of what happening 
    :returns: yyT: array defining the next set of parameter values ## change the name of this to somethign sensible...
    :returns: err: error values associated with input set of observables, one err per run. 
    :returns outdict: dictionary of data destined for json file
    """
    #     Through out code use the following is used:
    #    n - no. parameters # change to nParam
    #    m - no. observables # change to nObs
    # TODO:  replace has_key(x) throughout with x in dict

    # get constants from JSON's directory:

    alphas = studyJSON['alphas']
    terminus = studyJSON['terminus']
    # optional ones....
    if "sigma" in studyJSON:
        sigma = studyJSON['sigma']
    else:
        sigma = 0

    if (sigma != 0):
        ## need constraint_target  and if not fail.
        if "constraint_target" in studyJSON:
            constraint_target = studyJSON['constraint_target']
        else:
            sys.exit("Please specifiy constraint_target when sigma != 0.")  ## exit with error
    ## get the mu value
    if "mu" in studyJSON:
        mu = studyJSON['mu']
    else:
        mu = 1

    m = len(obs)
    n = len(param_value[0, :])

    # include constraint
    coef1 = 1. / (m + sigma)
    coef2 = sigma / ((m + 1) * (2. * mu))

    if sigma == 0:
        constraint = zeros(n + 1)  # TODO make np.zeros
        constraint[:] = 0.  # todo Unnesc
    else:  # sigma != 0 so have constraint..
        use_constraint = constraint.copy() - constraint_target  # TODO what this doing?

    ## print(out some info if trace on)
    if trace:
        print("Version is ", version)
        #        print('constraint.shape()=',constraint)
        #        print('constraint_target=',constraint_target)
        print(" n %d m %d \n" % (n, m))
        print("param_range")
        print(param_range)

    na = len(alphas)  # TODO na->nAlphas or nLineSearchRuns

    optStatus = "Continue"  # default optStatus

    # transpose arrays TODO -remove transpose and just do calculations directly
    param_valueT = transpose(param_value)

    param_rangeT = transpose(param_range)
    UM_valueT = transpose(UM_value)

    if trace:
        print("UM_valueT", UM_valueT)

    # scale observables, observations and covariance matrix
    # TODO replace with UM_value=UM_value*scalings taking advantage of broadcasting.
    # and 
    s_UM_valueT = UM_valueT.copy()
    for i in range(n + 1):  # CHANGED FROM m
        s_UM_valueT[:, i] = UM_valueT[:, i] * scalings

    s_obs = obs * scalings  # leave

    if trace:
        print("obs")
        print(zip(olist, zip(obs, scalings)))
        print(zip(olist, s_obs))

    # TOD cov=cov*np.outer(scalings,scalings)
    for i in range(m):
        cov[:, i] = cov[:, i] * scalings  # scale rows

    cov = cov * scalings  # using broadcasting to scale columns?

    ## now want to regularize s_cov though should be done after constraint included.
    if 'covar_cond' in studyJSON and studyJSON[
        'covar_cond'] is not None:  # specified a condition number for the covariance matrix
        cov = regularize_cov_ref(cov, studyJSON['covar_cond'], trace=trace)

    # constraint C_bar
    s_cov_constraint = zeros((m + 1, m + 1))
    covT = linalg.inv(cov)  # TODO replace with pseudo inverse and call it covInv
    # TODO  move all stuff with constraint together and check theory too.
    # TODO merge constraint into simulated and observed vector/matrix
    s_cov_constraint[0:m, 0:m] = coef1 * covT
    s_cov_constraint[m, m] = coef2

    if trace:
        print("Scaled and regularized cov")
        print(cov)

    UM = s_UM_valueT[:, 0]
    # commented out by SFBT
    #    param_min = param_rangeT[ 1, : ]
    #
    #
    #    param_max = param_rangeT[ 0, : ]
    ## ADDED by SFBT 
    param_min = param_rangeT[0, :]
    param_max = param_rangeT[1, :]
    if trace:
        print("param_rangeT")
        print(param_rangeT)
        print("param_min")
        print(param_min)
        print("param_max")
        print(param_max)

    ## TODO -- scale from 1 to 10 based on param_min and param_max
    ## param_scale=(param-param_min)/(param_max-param_min)*9+1
    ## NB this will change the results so do after all tests passed.
    param_scale = ones((n))

    for i in range(n):
        tmp = fabs(param_valueT[i, 0]) * param_scale[i]
        ##tmp = fabs(param_value[ 0, i ]) * param_scale[i]
        while tmp < 1:
            param_scale[i] = param_scale[i] * 10
            tmp = tmp * 10

    ## TODO clean JACOBIAN calculation
    ## DO NOT TRANSPOSE JACOBIAN!!!
    Jacobian = zeros((m, n))
    ##Jacobian = zeros( (n,m) ) # TODO -- remove commented code.
    ##Jacobian = zeros( (m,n) )

    # constraint Jacobian_bar
    Jacobian_constraint = zeros((m + 1, n))

    for i in range(n):
        ##Jacobian[ :, i ] = (UM_valueT[ :,i+1 ] - UM) \
        Jacobian[:, i] = (s_UM_valueT[:, i + 1] - UM) \
                         / ((param_valueT[i, i + 1] - param_valueT[i, 0]) * param_scale[i])
        ##Jacobian[ : , i ] = (UM_value[ i+1, : ] - UM) \
    ##/((param_value[ i+1, i ]-param_value[ 0, i ])*param_scale[ i ])

    # constraint Jacobian_bar
    Jacobian_constraint[0:m, :] = Jacobian
    for i in range(n):
        # 02/02/2015 changed below to incorporate TOA imbalance of 0.5 W/m^2
        # Jacobian_constraint[m,i] = (constraint[i+1]-constraint[0]) \
        Jacobian_constraint[m, i] = (use_constraint[i + 1] - use_constraint[0]) \
                                    / ((param_valueT[i, i + 1] - param_valueT[i, 0]) * param_scale[i])

    F = UM - s_obs  # This is difference of previous best case from obs
    ##F = UM - obs
    # TODO understand what this is doing.
    # constraint r_bar
    F_constraint = zeros(m + 1)
    F_constraint[0:m] = F
    F_constraint[m] = use_constraint[0]

    if trace:
        print("Jacobian before transpose")
        print(Jacobian)

    ##    save("Jacobian",Jacobian)
    ## read with j= np.load("Jacobian.npy")
    ##TODO figure out what happens if no constraint
    JacobianT = Jacobian.transpose()

    # constraint J_bar T
    Jacobian_constraintT = Jacobian_constraint.transpose()

    ## 29/09/2014
    ## modified  below to take account of cov in r & henceforth Jacobian
    if len(cov) == 0:
        J = dot(JacobianT, Jacobian)
    else:
        # constraint approximate Hessian of f(x)
        dum = dot(Jacobian_constraintT, s_cov_constraint)
        J = dot(dum, Jacobian_constraint)

    if trace:
        print("after dot, J is ")
        print(J)

    ##    save("dottedJ",J)
    ## TOD clean regularisation up, rename J to hessian
    perJ = J

    if trace:
        print('m, linalg.matrix_rank(perJ)', m, linalg.matrix_rank(perJ))

    con = linalg.cond(J)
    k = 8
    eye = identity(n)

    if trace:  ## SFBT added to reference doGaussNewton
        print("con %e k %d" % (con, k))
    while con > 10e10:
        if k >= 4:
            k = k - 1
            perJ = J + math.pow(10, -k) * eye
            con = linalg.cond(perJ)
            if trace:
                print("conREF %e k %d" % (con, k))
        # 11/05/2015 if con does not ge small enough quit w/ message
        # 22/04/2015 if con does not ge small enough
        else:
            print("regularisation insufficient, stopping: con %e k %d" % (con, k))
            optStatus = "Fatal"
            dictionary = {"jacobian": Jacobian, "hessian": J, "condnum": con}
            return optStatus, None, None, None, dictionary
            ## calculate eval_min, eval_max, eval_star
            # eval_max, evec_max = eigs(J, k=1, which='LM')
            # eval_min, evec_min = eigs(J, k=1, sigma=0, which='LM')
            # eval_star = eval_max*10.**(-10) - eval_min
            # perJ = J + eval_star*eye
            # con = linalg.cond(perJ)
            # print "con %e eval_min %e eval_star %e k %d"%(con, eval_min, eval_star, k)
            # 22/04/2015 break out of the while loop
            # break

    J = perJ

    ## 29/09/2014
    ## modified below to take account of cov in r & henceforth Jacobian
    # tmp = dot(JTcovT,F)
    # constraint approximate Jacobian of f(x)
    tmp = dot(dum, F_constraint)  # TODO look at maths and understand this line.
    ##tmp = dot(JacobianT,F)

    s = -linalg.solve(J, tmp)  # work out direction to go in.

    if trace:
        fn_label = 'DOGN_REF'
        print(fn_label + ": hessian =", np.round(J, 2))
        print(fn_label + ": J^T C^-1 F =  ", np.round(Jacobian_constraintT.dot(s_cov_constraint).dot(F_constraint), 2))
        print(fn_label + ": s=", s)

    x = param_scale * param_valueT[:, 0]

    # err & constrained err
    nsize = n + 1
    nloop = n + 1
    ioffset = 0
    err, err_constraint = calcErr_ref(s_UM_valueT, s_obs, covT, use_constraint, coef1, coef2, nsize, nloop, ioffset)
    ## TODO work with scaled parameters throughout
    test_max = param_max * param_scale
    test_min = param_min * param_scale
    y = zeros((na, n))
    ##y = zeros( (n,na) )
    ## deal with boundaries TODO use numpy stuff to avoid loops.
    for i in range(na):
        ##y[ i,: ] = x + math.pow(0.1,i)*s
        y[i, :] = x + alphas[i] * s
        ##y[ :, i ] = x + math.pow(0.1,i)*s
        for j in range(n):
            if y[i, j] > test_max[j]:
                y[i, j] = test_max[j]
            elif y[i, j] < test_min[j]:
                y[i, j] = test_min[j]
            else:
                y[i, j] = y[i, j]
            ##if y[ j, i] > test_max[j]:
            ##    y[ j, i] = test_max[j]
            ##elif y[ j, i] < test_min[j]:
            ##    y[ j, i] = test_min[j]
            ##else:
            ##    y[ j, i] = y[ j, i]

    yy = y
    for i in range(na):
        ##for i in range(m):
        yy[i, :] = y[i, :] / param_scale
        ##yy[ : , i] = y[ :, i]/param_scale

    # transpose backward
    yyT = transpose(yy)

    ################################ snip ################################
    dictionary = {"jacobian": Jacobian, "hessian": J, "condnum": con,
                  "objfn_GN": err, 'InvCov': covT}
    dictionary['software'] = version

    # hack for time being: output in files
    date = datetime.now()
    sdate = date.strftime('%d-%m-%Y_%H:%M:%S')
    outdir = './'  # with /exports/work/geos_cesd_workspace/OptClim/Data/kytest/
    # the framework fails.
    # This puts files in the iteration directory,
    # or if you are runnign standalone tests, to the directory you are in.

    #    outfile = outdir+'jacob_'+sdate+'.dat'
    #    save(outfile,Jacobian)
    #
    #    outfile = outdir+'hessi_'+sdate+'.dat'
    #    save(outfile,J)
    #
    #    outfile = outdir+'parav_'+sdate+'.dat'
    #    save(outfile,param_value)
    #
    #    outfile = outdir+'condn_'+sdate+'.txt'
    #    f = open(outfile,'w')
    #    f.write(str(con)+'\n')
    #    f.close()
    ################################ snip ################################

    # return yyT,err,dictionary
    # constraint err
    return optStatus, yyT, err, err_constraint, dictionary


### end of doGaussNewton

## reference code for doLineSearch
def doLineSearch_ref(param_value, param_range, UM_value, obs, defCov, scalings, olist, constraint, studyJSON,
                     trace=False):
    # def doLineSearch(param_value,param_range, UM_value, obs,cov, alphas, terminus, scalings, olist, constraint, mu, sigma, trace=False):
    """
    the function doGaussNewton does the calculataion, and can be invoked from
    the class GaussNewton, using data from studies, or from test python scripts
    for arbitrary inputs
    :param param_value: array of parameter values
    :param  param_range: array of max/min of range for each parameter in order
    :param UM_value: array of simulated observables
    :param obs: array of target values
    :param defCov: DEFAULT covariance array (header defines the observables in use) # TODO rename this
    :param scalings: observables are related arrays get scaled before statistical analysis
    :param olist: name of the observables - probably unused except perhaps in diagnostic
    :param studyJSON to allow simple parameters to be extensible.
    :param trace: little used at present - scope to turn on/off some trace of what happening
    :returns: status: "Continue", "Covnerged", "Stalled"
    :returns: err: error values associated with input set of observables, one err per run.
    :returns: newParam: array defining the next set of parameter values
    :returns: index: index identifying the best of the runs from along the line searched.
    :returns outdict: dictionary of data destined for json file
    """
    ## TODO rename these to sensible things
    ##        n - no. parameters
    ##        m - no. observables

    ## TODO generally clean up constraint calculation

    ##TODO replace has_key(x) with x in dict
    # get constants from JSON's directory:

    alphas = studyJSON['alphas']
    terminus = studyJSON['terminus']

    # optional ones....
    if "sigma" in studyJSON:
        sigma = studyJSON['sigma']
    else:
        sigma = 0

    if (sigma != 0):
        ## need constraint_target
        if "constraint_target" in studyJSON:
            constraint_target = studyJSON['constraint_target']
        else:
            sys.exit("Please specifiy constraint_target when sigma != 0.")  ## exit with error

    if "mu" in studyJSON:
        mu = studyJSON['mu']
    else:
        mu = 0

    if "prob_int" in studyJSON:
        prob_int = studyJSON['prob_int']
    else:
        prob_int = 0.5
    if "prob_obs" in studyJSON:
        prob_obs = studyJSON['prob_obs']
    else:
        prob_obs = 0.5
    # if key "covLS" exists then treat this as the covariance file to use.
    # TODO make this non optional...
    cov_iv = studyJSON['covIV']  # modification to reference code to get covariance from studyJSON file.
    #        if studyJSON['study'].has_key('covLS'):
    #             ls_slist, cov_iv=fwk.readCovariances(studyJSON, 'covLS')
    #             if  ls_slist != olist:
    #                  print "doLineSearch: error in cov file with keyword covLS"
    #                  print "ls_slist"
    #                  print ls_slist
    #                  print "olist"
    #                  print olist
    #                  exit()
    # else:
    cov = defCov  # default covariance
    # initialise dictionary for output data

    outdict = dict()
    outdict['software'] = version

    m = len(obs)
    n = len(param_value[0, :])

    # constraint
    coef1 = 1. / (m + sigma)
    coef2 = sigma / ((m + 1) * (2. * mu))
    # 02/02/2015 added line below to incorporate TOA imbalance of 0.5 W/m^2 # TODO clean
    use_constraint = constraint - constraint_target

    na = len(param_value[:, 0]) - n - 1
    # verify have the same number of model values as alpha values. If not die!
    assert na == len(alphas), "length of line confused %d or %d" % (na, len(alphas))
    if trace:
        print("Version is ", version)
        print('use_constraint.shape', use_constraint.shape)
        print(" n %d m %d \n" % (n, m))
        print("na = %d" % (na))

    # transpose arrays # TODO remove transposes
    param_valueT = transpose(param_value)
    UM_valueT = transpose(UM_value)

    # scale observables and covariance matrix # TODO replace with same apporach as doGaussNewton
    s_UM_valueT = UM_valueT.copy()
    for i in range(n + 1 + na):
        s_UM_valueT[:, i] = UM_valueT[:, i] * scalings

    s_obs = obs * scalings

    s_cov = cov.copy()
    for i in range(m):
        s_cov[:, i] = cov[:, i] * scalings

    s_cov = s_cov * scalings  # scale rows using broadcasting??

    ## now want to regularize s_cov
    ## THIS CODE SHOULD GO INTO UNIFIED ERROR FUNCTION AS IT WILL BE INCLUDED IN doLS too.
    if 'covar_cond' in studyJSON and studyJSON[
        'covar_cond'] is not None:  # specified a condition number for the covariance matrix?
        s_cov = regularize_cov_ref(s_cov, studyJSON['covar_cond'], trace=trace)

    UM = s_UM_valueT[:, 0]

    F = UM - s_obs
    if len(cov) == 0:
        fcurrent = sum(F ** 2)
        if trace:
            print("no covariance set")
    else:
        covT = linalg.inv(s_cov)
        FT = transpose(F)
        FTcovT = dot(FT, covT)
        fcurrent = dot(FTcovT, F)

        # constraint
    fcurrent_constraint = coef1 * fcurrent + coef2 * use_constraint[0] * use_constraint[0]

    if trace:
        print("s_cov")
        print(s_cov)
        print("covT")
        # print(covT)

    tmp = param_valueT[1, :]
    k = tmp.size
    # err & constrained err for current iteration.
    # If no constraint, constraint_err_current=err_current by definition.

    nsize = 1  # TODO figure out what these numbers are. Suspiciously magic...  best done by passing slices into calcErr
    nloop = 1
    ioffset = 0
    # TODO move and simplify calcErr so can be reused by other routines. CLarify with MIKE need. i.e. write fn documentation first!
    err_current, err_constraint_current = calcErr_ref(s_UM_valueT, s_obs, covT, use_constraint, coef1, coef2, nsize,
                                                      nloop, ioffset)
    tmp = param_valueT[1, :]
    k = tmp.size
    # TODO clean up code.
    # err & constraied err for new iteration.
    # If no constraint, constraint_err=err by definition.
    nsize = na
    nloop = k - n - 1
    ioffset = n + 1
    err, err_constraint = calcErr_ref(s_UM_valueT, s_obs, covT, use_constraint, coef1, coef2, nsize, nloop, ioffset)

    # index = f.argmin()
    # fmin = f.min()
    # constraint
    index = err_constraint.argmin()
    err_constraint_min = err_constraint.min()

    # errmin=err[index]

    y = param_valueT[:, n + 1 + index]
    NewUMT = s_UM_valueT[:, n + 1 + index]

    # NewUMT=UM_valueT[ :, n+1+index ]
    # y = param_value[ n+1+index, : ]
    # NewUM=UM_value[ n+1+index, : ]

    NewParam = []  # define it

    status = "Stalled"  # reset this if we have more to do
    # For now, it says - no more.

    step = zeros(n)  # establish array for the stepsize
    ## TODO verify this doing what expected and that return stat is used.
    ## TODO clean i.e. just do outdict['status']="Stalled"
    ## TODO simplify output to just pass outdict? Clarify with MIKE!
    ## TODO clean termination codes.
    # evaluate 3 criteria

    # ---------------------------------------------------------
    # if constrained err at t=n <= minimum constrained err at t=n+1, exit
    # ---------------------------------------------------------
    if err_constraint_current <= err_constraint_min:
        status = "Stalled"
        substatus = "error worse"
        outdict['status'] = status
        outdict['substatus'] = substatus
        outdict['bestrun'] = index
        return status, err, err_constraint, NewParam, index, outdict

    # ---------------------------------------------------------
    # No progress
    #   read in internal covariance matrix "cov_iv"
    #   calc & use inverse internal covariance matrix "cov_ivT"
    #   Which may also need regularising.
    # ---------------------------------------------------------
    # internal covariance matrix

    # scale and invert covariance matrix
    s_cov_iv = cov_iv.copy()
    for i in range(m):
        s_cov_iv[:, i] = cov_iv[:, i] * scalings
    s_cov_iv = s_cov_iv * scalings  # scale columns?
    ## regularize.
    if 'covar_cond' in studyJSON and studyJSON[
        'covar_cond'] is not None:  # specified a condition number for the covariance matrix?
        s_cov_iv = regularize_cov_ref(s_cov_iv, studyJSON['covar_cond'], trace=trace)

    # check whether the scaled covariance matrices are invertible ~ TODO why this being done here and not earlier?
    # TODO (after checking with Cora) replace all inverse with pinv No COra says stick to inverse. Make it an option for latter use.
    rank_cov_internal_variability = linalg.matrix_rank(s_cov_iv)
    rank_cov_total = linalg.matrix_rank(s_cov)
    if rank_cov_internal_variability != m:
        print("Warning: internal variability covariance matrix not invertible w/ rank:", rank_cov_internal_variability)
    if rank_cov_total != m:
        print("Warning: total covariance matrix not invertible w/ rank:", rank_cov_total)
    if trace:
        print('Rank & cond. no. for int. var. cov:', rank_cov_internal_variability, linalg.cond(s_cov_iv))
        print('Rank & cond. no. for total cov:    ', rank_cov_total, linalg.cond(s_cov))

    cov_ivT = linalg.inv(s_cov_iv)

    # simulated observables to compare
    xn = NewUMT.copy()
    xnm1 = UM.copy()

    dum = xn - xnm1

    diag_ele = zeros(m)
    for i in range(m):
        diag_ele[i] = s_cov_iv[i, i]

    if trace:
        print('xn[i],xnm1[i],dum[i],diag_ele[m],dum[i]*dum[i]/diag_ele[i]')
        for i in range(m):
            print(xn[i], xnm1[i], dum[i], diag_ele[i], dum[i] * dum[i] / diag_ele[i])

    dumT = transpose(dum)
    dumTcov_ivT = dot(dumT, cov_ivT)
    test_stat_no_progress = 2. * dot(dumTcov_ivT, dum)
    # confidence interval
    chisq_val_no_progress = chi2.ppf(prob_int, m)
    # print 'xn-xnm1' , dum

    # statistically the same
    if test_stat_no_progress <= chisq_val_no_progress:
        # status = "Stalled" # changed below by SFBT
        status = "No State Change"
        substatus = "no state change"
        outdict['status'] = status
        outdict['substatus'] = substatus
        outdict['bestrun'] = index
        outdict['test_stat_no_progress'] = test_stat_no_progress
        outdict['chisq_val_no_progress'] = chisq_val_no_progress
        return status, err, err_constraint, NewParam, index, outdict

    # -----------------------------------------
    # In agreement with target observations
    #   use total covariance matrix
    # -----------------------------------------
    # compare simulated observables with targets
    xn = NewUMT.copy()
    xobs = s_obs.copy()

    dum = xn - xobs
    dumT = transpose(dum)
    dumTcovT = dot(dumT, covT)
    test_stat_agree_obs = dot(dumTcovT, dum)
    # confidence interval
    chisq_val_agree_obs = chi2.ppf(prob_obs, m)

    # statistically the same
    if test_stat_agree_obs <= chisq_val_agree_obs:
        status = "Converged"
        outdict['status'] = status
        outdict['bestrun'] = index
        outdict['test_stat_agree_obs'] = test_stat_agree_obs
        outdict['chisq_val_agree_obs'] = chisq_val_agree_obs
        return status, err, err_constraint, NewParam, index, outdict

    #        print 'Test statistic for No Progress: ', test_stat_no_progress
    #        print 'Test statistic for Agree w/ obs:', test_stat_agree_obs
    #        print 'Chi-square value:               ', chisq_val
    #        print ''

    for i in range(n):
        step[i] = param_valueT[i, i + 1] - param_valueT[i, 0]
        NewParam = rangeAwarePerturbations(y, param_range, step)
        status = "Continue"  # have more runs to do

    # transpose
    #        NewParamT = transpose(NewParam)
    # forgotten why this is needed!        index+=n+1
    outdict['status'] = status
    outdict['bestrun'] = index
    outdict['test_stat_no_progress'] = test_stat_no_progress
    outdict['test_stat_agree_obs'] = test_stat_agree_obs
    outdict['chisq_val_no_progress'] = chisq_val_no_progress
    outdict['chisq_val_agree_obs'] = chisq_val_agree_obs

    return status, err, err_constraint, NewParam, index, outdict

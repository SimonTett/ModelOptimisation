"""
Module that provides optimisation functions. Functions here are Scientific which means they
 only get to use data in the "optimise" block of the configuration file. For everything else the framework provides data
 as numpy arrays and lists. Exception: Currently needs access to minmax, step and covariance blocks
  Currently provides:
    doGaussNewton: Does Gauss Newton calculation working out LineSearch values
    doLineSearch: Decides to terminate or not and returns next set of doGaussNewton cases

   And a bunch of support routines
See individual functions for documentation. 
Test cases for this module can be found in test_Optimise.py
"""

## issues for Mike.
#  arrays returned should be in the same "order" as inputs. Both  appear to be transposed.

import numpy as np
import logging
import xarray
from scipy.stats import chi2
import typing

my_logger = logging.getLogger(f"OPTCLIM.{__name__}")
def get_default(dct, key, default):
    """
    :param dct: dictionary to read value from
    :param key: key to use
    :param default: default value to use if not set or None
    :return: value from dct if provided, default if not provided or None
    """
    value = dct.get(key, default)
    if value is None:  # value is none then use default
        value = default
    return value  # and return the value


def scale_data(scales, obs, simObs, *covariances):
    """
    :param scales: Scalings to apply to data
    :param obs: Observations to be scales
    :param simObs: Simulated Observations to be scaled
    :*param covariances: Covariances to be scaled
    :return:  returns scaled obs, simObs and covariances
    """
    cov_scale = np.outer(scales, scales)
    scaled_cov = []
    for cov in covariances:  # iterate over covariances scaling them
        scaled_cov.append(cov * cov_scale)  # scale covariances

    # make up result array
    result = [obs * scales, simObs * scales[np.newaxis, :]]
    result.extend(scaled_cov)
    #logging.warning("Calling scale_data. Probably should not! ")
    return result


def rangeAwarePerturbations(baseVals, parLimits, steps, nrandom=None,
                            deterministic=True, trace=False):
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

         steps: array of stepsizes for each parameter.

         nrandom: perturb nrandom parameters rather than the full set.
         deterministic: optional argument with default value True -- make random choice of parameters deterministic.
            deterministic will be added to the RNG seed. So to modify what you get change the value of deterministic.
            It is a bit of a hack though...

         trace: Optional argument with default value False.
                Boolean -- True to get output, False no output

    Note inputs are all arrays, with data in order of the parameter list.

    Returns
          y: array defining the next set of parameter values
             with dimensions (nparameters+1, nparameters) with each row corresponding to a
             single simulation.
    """
    if trace:
        print("in rangeAwarePerturbations from baseVals")
        print(baseVals)

    npar = len(baseVals)
    nruns = npar + 1

    y = np.zeros((nruns, npar))

    # derive the centre-of-valid-range

    centres = (parLimits[:, 0] + parLimits[:, 1]) * 0.5

    if trace:
        print("steps", steps)
        print("parLimits", parLimits)
        print("centres", centres)

    # set "base run" values - those being perturbed one per run,after run first run.

    for j in range(nruns):  # Think this could be done with y=np.array(baseVals)
        y[j, :] = baseVals[:]

    # apply step so it goes in direction of the mid-range for the relevant parameter

    for j in range(npar):
        # parameter j, in run j+1
        if y[j + 1, j] <= centres[j]:
            y[j + 1, j] += abs(float(steps[j]))
        else:
            y[j + 1, j] -= abs(float(steps[j]))
    # check in range and fail if not
    bad = (y < parLimits[:,0]) | (y > parLimits[:,1])
    if np.any(bad):
        ybad = np.where(bad,y,np.nan)
        print("Bad y =:\n",ybad,"\n")
        print("Param range\n",parLimits,"\n")
        breakpoint()
        raise ValueError("Parameters out of range")
    # possibly perturb the array
    if nrandom is not None:
        y, randIndx = randSelect(y, nrandom, deterministic=deterministic, trace=trace)
    else:
        randIndx = None

    # check in range and fail if not
    bad = (y < parLimits[:,0]) | (y > parLimits[:,1])
    if np.any(bad):
        raise ValueError("Parameters out of range")

    if trace:
        print("leaving rangeAwarePerturbations with derived perturbed runs:")
        print(y)

    return y, randIndx


def regularize_hessian(hessian, reg_crit_cond, reg_pow_range, trace=False):
    """ Regularize hessian matrix """
    fn_label = 'regHes'
    perJ = hessian.copy()  # make sure we copy values.
    con = np.linalg.cond(hessian)
    eye = np.identity(hessian.shape[0])
    for k in range(*reg_pow_range):  # *(list) breaks up list when argument
        if con < reg_crit_cond:
            break  # exit the loop if our condition number small enough

        perJ = hessian + eye * 10 ** k  # add scaled Identity matrix to diagonal.
        con = np.linalg.cond(perJ)  # compute condition number
        if trace:
            print(fn_label + ": con %e k %d" % (con, k))
    # end loop over powers of 10.
    if (con >= reg_crit_cond):  # failed to regularize matrix
        print("regularisation insufficient, stopping: con %e k %d" % (con, k))
        return None, con
    else:  # managed to regularize hessian
        # TODO also return the value of k actually used. Noting that we might not
        # need to do any regularisation.
        return perJ, con


def regularize_cov(covariance, cond_number=None, initial_scale=1e-4, trace=False):
    """
    Applys Tikinhov regularisation to covariance matrix so its inverse can be sensibly computed.
    This is sometimes called ridge regression.
    Params: covariance - a NxN symmetric matrix whose diagonal values will be modified.
            cond_number -- the target condition number relative to the condition number of the diagonal matrix
            initial_scale -- initial scaling on the diagonal elements (default value 1e-4)
            trace -- provide some diagnostics when True (default is False)

    Returns: Regularised covariance matrix

    Algorithm: The scaling is doubled each iteration until the condition
    number is less than target value. Target value is cond_number*cond(diagonal covariance)


    """
    if cond_number is None:
        return covariance.copy()  # just return the covariance number if cond_number is None.

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
            raise ValueError()
    ##  end of regularisation loop.
    if trace:
        print("Used Tikinhov regularisation with scale = ", scale_diag)
        print("Condition #  = ", con, " Cond No is ", cond_number)

    return reg_cov


### end of regularize_cov


def calcErr(simulated, observations, cov=None):
    """
    calculate error for simulated observed values given targets and covariance.

    param: simulated: A numpy array of simulated observations. Each row
      corresponds to a different simulation and each column to a
      different observations.

    param: observations: A numpy 1D array of observations. 

    opt-param: cov: Covariance matrix. Default value is the identity matrix.

    """

    nobs = len(observations)

    if cov is None:
        cov = np.identity(nobs)  # make covariance a unit matrix if not defined

    ## compute number of simulations.
    if simulated.ndim == 1:  # might break if we only have one obs...
        nsimulations = 1
        lsimulated = np.reshape(simulated, (1, simulated.shape[0]))  # maek it 2d
    else:
        nsimulations = simulated.shape[0]  # number of simulations
        lsimulated = simulated  # ptr so free!

    err = np.zeros(nsimulations)  # set up error vector
    inv_cov = np.linalg.inv(cov)  # compute inverse covariance

    for i in range(nsimulations):
        delta = lsimulated[i,] - observations
        err[i] = (delta.dot(inv_cov)).dot(delta)

    err = np.sqrt(err / float(nobs))  # root mean square.
    return err


def doGaussNewton(param_value, param_range, UM_value, obs, cov=None,
                  scalings=None, olist=None,
                  constraint=None, constraint_target=None, studyJSON={}, trace=False):
    """
    the function doGaussNewton does the  calculation for the n+1 initial runs in an iteration,
    and can be invoked from (usually) makeNewRuns, using data from studies, or from
    test python scripts for arbitrary inputs. 

 
    :param param_value: numpy array of parameter values. 
    :param param_range: 2D numpy  array of min/max of range for each parameter.
      param_range[:,0] is the min value, param_range[:,1] the maximum value.
      Q: Should this be passed through studyJSON as property of optimisation rather than framework?
    :param UM_value: array of simulated observables
    :param obs: array of target values -- should contain same number of elements as columns in UM_value

     param_value & UM_value are ordered such that rows correspond to different simulations and 
     columns to parameters and simulated observations respectively. 
 
    :opt-param cov: covariance array (header defines the observables in use) 
                    Default is identity matrix
    :opt-param scalings: observables are related arrays get scaled before statistical analysis
                Default value is 1
    :opt-param olist: name of the observables. Default values are obs1..obsN where N is number of oBs.
    :opt-param constraint: Value of simulated constraint.
    :opt-param constraint_target: Value of observed constraint. BOTH constraint and constraint_target should be   present if studyJSON has sigma True
    :opt-param studyJSON The optimisation control specified as a dict. Contains the following values:
        alphas        -- alpha values used in linesearch. Default values are 0.3, 0.7 and 1.
        sigma         -- If true then constraint is used. Default is False
        mu            -- weighting on constraint. (wt is sqrt(1/2mu)). Default is 1.
        covar_cond    -- if specified used by regularize_cov (qv) to regularize covariance
        reg_crit_cond -- critical condition number for regularisation of Hessian. Default is 10e10.
        reg_pow_range -- range of powers used to generate list of powers in regularisation. Default is (-7,-2)
    
    :opt-param trace: turn on/off trace of what happening. Default is False. Set True to get some tracing.

    :returns: linesearch: array defining the next set of parameter values. Each row corresponds to parameter set. 
    :returns: err: Error values for each simulation.
    :returns: err_constraint: Constrained error for each simulation
    :returns info: dictionary of data destined for json file.
       hessian: An array of the regularised hessian matrix.
       jacobian: An array of the Jacobian matrix
       InvCov: An array of the Inverse Covariance matrix (after possible regularisation)
       condnum: The Condition Number of the regularized hessian matrix.
       software: A string with info on the software
       scalings: Scalings applied to data.
       olist: names of variables 
       params: A dictionaary of the  actual parameters (or defaults used) from studyJSON
      
    """
    fn_label = 'doGN'  # name of function for tracing purpose
    nObs = len(obs)
    nParam = len(param_value[0, :])
    ## Set up default values
    if trace is True:
        breakpoint()
    if scalings is None:
        use_scalings = np.repeat(1.0, nObs)
    else:
        use_scalings = scalings.copy()

    if cov is None:
        use_cov = np.identity(nObs)
    else:
        use_cov = cov

    if olist is None:
        use_olist = ['obs' + str(x) for x in range(nObs)]
    else:
        use_olist = olist

    # get constants from JSON's directory giving them a default value if not present.

    alphas = get_default(studyJSON, 'alphas', [0.3, 0.7, 1])
    sigma = get_default(studyJSON, 'sigma', False)  # default is no constraint
    mu = get_default(studyJSON, 'mu', 1)  # mu if we have it otherwise default value of 1.
    covar_cond = get_default(studyJSON, 'covar_cond', None)
    reg_crit_cond = get_default(studyJSON, 'reg_crit_covar',
                                10e10)  # condition number above which we regularise hessian
    reg_pow_range = get_default(studyJSON, 'reg_pow_range', [-7, -2])  # range of powers (normal Python convection)
    # we use for regularisation

    # store values used in params
    params = {'alphas': alphas, 'sigma': sigma,
              'constraint_target': constraint_target, 'mu': mu,
              'reg_crit_cond': reg_crit_cond, 'reg_pow_range': reg_pow_range}  # parameters as actually used.

    use_obs, use_UM_value, use_cov = scale_data(use_scalings, obs, UM_value, use_cov)  # scale

    # compute scaling on parameters in order to keep Jacobain "well behaved"
    # will eventually allow multiple ways of doing this. 
    ## TODO -- scale from 1 to 10 based on param_min and param_max
    ## param_scale=(param-param_min)/(param_max-param_min)*9+1
    ## NB this will change the results so do after all tests passed.
    pwr = np.floor(np.log10(np.fabs(param_value[0, :] + 1e-17)))  # compute power of 10 for params
    param_scale = np.ones((nParam))  # default scaling is one for all parameters
    indx = (pwr < 0) & (np.fabs(param_value[0,
                                :]) > 1e-10)  # find those parameters that need to be scaled up because they are less than 1 and are not tiny!
    param_scale[indx] = 10 ** (-pwr[indx])  # compute scaling.
    if trace: print("Param_scale is ", param_scale)

    # Deal with constraint
    if sigma:
        # got constraint so check have constraint_target and constraint not none
        if constraint_target is None or constraint is None:
            print("Sigma true but constraint_target not provided or constraint not set")
            raise ValueError()
            return "Fail"  # return failure

        if constraint.ndim == 2:
            use_UM_value = np.hstack([use_UM_value, constraint])  # add constraint value to end of UM value
        else:
            use_UM_value = np.hstack(
                [use_UM_value, constraint[:, np.newaxis]])  # add constraint value to end of UM value

        use_obs = np.hstack((use_obs, constraint_target))  # add constraint target to end of obs
        use_cov2 = np.zeros((nObs + 1, nObs + 1))  # generate bigger matrix to put modified covariance in,
        use_cov2[0:-1, 0:-1] = use_cov[:, :]  # std one
        use_cov2[-1, -1] = 2 * mu  ## include constraint in the covariance.
        use_cov = use_cov2  # and use_cov is now use_cov2
        del use_cov2  # delete the use_cov2
        nObs += 1  # one more observation
        if trace:
            print("Using Constrained optimisation ")

    optStatus = "Continue"  # default optStatus

    # Possibly regularize covariance matrix
    if covar_cond is not None:  # specified a condition number for the covariance matrix?
        use_cov = regularize_cov(use_cov, covar_cond, trace=trace)

    InvCov = np.linalg.inv(use_cov)  # Inverse of covariance matrix. Might be better as pseudo-inv
    if trace > 2: print(fn_label + ": Scaled and regularized cov = ", cov)

    # verify parameters in sensible range
    if np.any(param_range[:, 0] > param_range[:, 1]):
        print("minimum param > maximum param")
        raise ValueError()
    # Compute Jacobian
    UM = use_UM_value[0, :]

    nruns = param_value.shape[0]  # how many runs were done
    Jacobian = np.zeros((nruns - 1, nObs))
    paramIndex = np.zeros(nruns - 1, dtype=int)  # the index to the parameters actually changed.
    for i in range(nruns - 1):
        # Jacobian calculation ASSUMES only one parameter changed.
        # work out which one  was changed
        deltaParam = param_value[i + 1, :] - param_value[0, :]
        non_zero = (deltaParam != 0.0)  # logical array True where non-zero. Probably need to be FP aware...
        assert np.sum(non_zero) == 1  # should only have one parameter changed..
        paramIndex[i] = np.where(non_zero)[0]  # work out which parameter was actually changed
        dParam = deltaParam[non_zero] * param_scale[non_zero]  # change in param after scaling.
        Jacobian[i, :] = (use_UM_value[i + 1, :] - UM) / dParam[0]

    F = UM - use_obs  # Difference of previous best case from obs
    hessian = (Jacobian.dot(InvCov)).dot(Jacobian.T) / float(nObs)  # = $\del^2f(x) or hessian
    info = {"jacobian": Jacobian, "hessian": hessian, "condnum": None}  # store for the moment
    hessian, con = regularize_hessian(hessian, reg_crit_cond, reg_pow_range, trace=trace)  # regularize hessian
    if hessian is None:
        # regularization failed
        my_logger.warning("Failed to regularise hessian matrix. You probably need to fix your covariance matrix.")
        info['condnum'] = con  # update info on the condition number
        raise ValueError("Failed because hessian could not be regularised")
        return "Fatal", None, None, None, info

    # solving linear problem $\nabla^2f(x) s = \nabla f(x) $
    # Note that if not perturbing all parameters s is in the sub-space (and ordering) of the perturbed parameters
    # undoing this is a bit tricky involving some reverse indexing.
    s = -np.linalg.solve(hessian, Jacobian.dot(InvCov).dot(F) / float(nObs))
    if trace:
        print(fn_label + ": parameters perturbed", paramIndex)  # no names for parameters just indices.)
        if trace > 2:
            print(fn_label + ": hessian =", np.round(hessian, 2))
            print(fn_label + ": J^T C^-1 F =  ", np.round(Jacobian.dot(InvCov).dot(F), 2))
        print(fn_label + ": s=", s)
        print(fn_label + "; con=", con)

    # err & constrained err
    err_constraint = calcErr(use_UM_value, use_obs, cov=use_cov)  # compute error (which might include constraint)
    # now deal with constraint (in hacky way)
    if sigma:  # Need to compute unconstrained error.
        err = calcErr(use_UM_value[:, 0:-1], use_obs[0:-1], cov=use_cov[0:-1, 0:-1])  # without constraint
    else:
        err = err_constraint  # error and constrained error the same by defn.

    #############################################################
    # compute linesearch values.

    ##linesearch = np.broadcast_to(param_value[0,:],[len(alphas), nParam]).copy()# fill linesearch with base param value alas needs numpy 1.10 or greater and eddie at 1.9
    linesearch = np.zeros([len(alphas), nParam])  # make empty array
    linesearch[:, :] = param_value[0, :]  # use broadcasting to fill array
    # compute alphas*s_scale for all combinations. Order determines array size.
    # note (as above) that s is in sub-space of perturbed parameters
    # and so we need to do some tricky array indexing to get it back to the full parameter set.
    searchVect = s / param_scale[paramIndex]  # rescale search vector
    linesearch[:, paramIndex] += np.outer(alphas, searchVect)
    linesearch = np.maximum(param_range[:, 0], np.minimum(linesearch, param_range[:, 1]))  # deal with boundaries
    bad = (linesearch < param_range[:,0]) | (linesearch > param_range[:,1]) # outside range
    if np.any(bad):
        raise ValueError("Have linesearch params outside param_range")

    # wrap diagnostics up.
    info = dict(jacobian=Jacobian, hessian=hessian, condnum=con, searchVect=searchVect,
                InvCov=InvCov,
                #software_vn=version, revision=revision, SvnURL=svnURL,
                scalings=use_scalings, params=params, paramIndex=paramIndex)
    # TODO include in the info dict the parameter names and their values. That might need to
    # happen in the framework. This will make subsequent data processing much easier.
    # TODO include some information about the regularisation
    return optStatus, linesearch, err, err_constraint, info


# end of doGaussNewton


def randSelect(paramValues, nrandom, deterministic=True, trace=False):
    """
    Randomly selects nrandom subset of perturbations from full set. Should be called from  doLineSearch
      if required. Will then modify next set of runs to be ran. Should also modify optClimBegin

    :param paramValues: parameter values for next iteration -- each case corresponds to a row with 0th row being
              the base value. The base value should always be included in the set.
    :param nrandom: number of random cases to select
    :param deterministic: optional with default True -- if True RNG initialised with first set of param values
              deterministic will be added to the values in order to allow a little user control over the RNG
    :param trace: optional with default False -- if true more diagnostic information provided
    :return: sub-set of nrandom+1 param_values to actually run and  indx into initial parameter array.
    """

    fn = "randSelect"
    # check  < nrandom < len(paramNames)
    nparams = paramValues.shape[1]
    assert paramValues.shape == (nparams + 1, nparams)
    assert 0 < nrandom < nparams  # nrandom between 0 and nrandom.
    if trace:
        print(fn, ": All tests passed. Shuffling.")

    if deterministic:  # initialise RNG based on param_values[0,:]
        # this is tricky because we need to convert them to integer arrays...
        scale = 10.0 ** np.floor(np.log10(np.abs(paramValues[0, :])) - 3)  # allow a thousand different values
        int_v = np.int_(np.floor(abs(paramValues[0, :]) / scale))  # now got it as integer
        int_v[0] += deterministic  # increment a bit.
        np.random.seed(int_v)  # set the seed up
        if trace: print(fn, ": Seed set to ", int_v)

    indx = np.arange(nparams)  # index to parameters
    np.random.shuffle(indx)  # shuffle the index
    indx = indx[0:nrandom]  # get the first nrandom elements
    # sort indx to make comparison for humans easier.
    indx = np.sort(indx)
    indx = np.hstack((0, indx + 1))  # put the zeroth row back in.
    pvalues = paramValues[indx, :]  # extract the array values wanted.

    if trace:
        print(fn, ": Indx is ", indx)
        print(fn, ": Original Values are ", paramValues)
        print(fn, ": Shuffled array is", pvalues)

    return pvalues, indx  # that is us all done.


def doLineSearch(param_value, param_range, UM_value, obs, step, cov=None, cov_iv=None,
                 scalings=None, olist=None,
                 constraint=None, constraint_target=None, studyJSON={}, trace=False):
    """ Run Linesearch Algorithm and decide when to stop.
     :param param_value: numpy array of parameter values.
     !param param_range: numpy array of parameter ranges.
     :param UM_value: numpy array of simulated observables for the entire iteration. The last nalpha will be used.
     :param obs: 1D array of target observations
     :param step: 1D array of perturbation steps.

      param_value & UM_value are ordered such that rows correspond to different simulations and
      columns to parameters and simulated observations respectively.

     :opt-param cov: covariance array used to compute error with default the identity matrix
     :opt-param cov_iv: covariance array used to decide if states are different -- depends on internal var.
       Default is 1
     :opt-param scalings: observables and related arrays get scaled before statistical analysis
                 Default value is 1
     :opt-param olist: name of the observables. Default values are obs1..obsN where N is number of oBs.
     :opt-param constraint: Value of simulated constraint.
     :opt-param constraint_target -- observed value for constraint.

     BOTH constraint and constraint_target should be present if studyJSON has simga set to True

     :opt-param studyJSON The optimisation control specified as a dict. Uses the following values:
         alphas        -- alpha values used by doGaussNewton -- only number used. Default is 3.
         sigma         -- If True then constraint is used. Default is False
         mu            -- weighting on constraint. (wt is sqrt(1/2mu)). Default is 1.
         covar_cond    -- if specified used by regularize_cov (qv) to regularize covariance.
                 Termination Conditions
         prob_int      -- critical probability that previous and current best states are different. Default is 0.5
         prob_obs      -- critical probability that current state is different from observations. Default is 0.5

         nrandom        -- if not none then chose (at random) nrandom permutations rather than full set. Default is None
         deterministicPertub -- if True then make random choice is deterministic function of parameters.
         minImprovement -- the minimum value by which the error should be reduced to continue iteration.

     :opt-param trace: turn on/off trace of what happening. Default is False. Set True to get some tracing.
     :returns StatusInfo -- status of optimisation. 'continue' means keep going
     :returns: err: Error values for each simulation.
     :returns: err_constraint: Constrained error for each simulation
     :returns: GaussNewton: nparam+1, nparam  array defining the next set of parameter values.
               Or None if LineSearch has terminated.
     :returns index  -- which one of the linesearch values is the best value.
     :returns: bestParam -- the best parameters in the current iteration.
     :returns info: dictionary of data destined for json file.
        cov: An array of the  Covariance matrix (after possible regularisation)
        cov_iv: An array of the covariance matrix of int var (after possible regularisation)
        software: A string with info on the software
        scalings: Scalings applied to data.
        olist: names of variables
        params: A dictionary of the  actual parameters (or defaults used) from studyJSON
        BestIndex: Best index of the linesearch values
        StatusInfo: Additional information on the status

    """

    # There is quite a bit of common code in do_GaussNewton and doLineSearch
    ## code is largely to do with setup/default values. Wonder if that code could be
    ## pulled into some common code? Probably by having class for GN and LS which share initialisation code..
    ## But I think that such a large redisgn that not worth doing..
    ## Setup default values
    fn_label = 'doLS'  # name of function for tracing purpose
    nObs = len(obs)
    nParam = len(param_value[0, :])
    # Set up default values being careful not to change inputs
    if scalings is None:
        use_scalings = np.repeat(1.0, nObs)
    else:
        use_scalings = scalings.copy()
    if cov is None:
        use_cov = np.identity(nObs)
    else:
        use_cov = cov.copy()
    if cov_iv is None:
        use_cov_iv = np.identity(nObs)
    else:
        use_cov_iv = cov_iv.copy()
    if olist is None:
        use_olist = ['obs' + str(x + 1) for x in range(nObs)]
    else:
        use_olist = olist[:]  # can#'t copy lists! Just slice them..

    # scale obs, simulated obs and covariances
    use_obs, use_UM_value, use_cov, use_cov_iv = scale_data(use_scalings, obs, UM_value, use_cov, use_cov_iv)
    # extract values from studyJSON providing default values if wanted.
    alphas = get_default(studyJSON, 'alphas', [1.0, 0.7, 0.3])
    nalphas = len(alphas)  # only want the number of alpha values.
    sigma = get_default(studyJSON, 'sigma', False)  # default is no constraint
    mu = get_default(studyJSON, 'mu', 1)  # mu if we have it otherwise default value of 1.
    covar_cond = get_default(studyJSON, 'covar_cond', None)
    prob_int = get_default(studyJSON, 'prob_int', 0.5)
    prob_obs = get_default(studyJSON, 'prob_obs', 0.5)
    # compute derived critical values for termination tests
    chisq_no_progress = chi2.ppf(prob_int, nObs)
    chisq_obs_agree = chi2.ppf(prob_obs, nObs)
    # Potential random choice.
    nrandom = get_default(studyJSON, 'nrandom', None)
    deterministicPerturb = get_default(studyJSON, 'deterministicPertub', True)
    # reduce error
    minImprovement = get_default(studyJSON, 'minImprovement', 0.0)
    # store optional (and derived) variables from studyJSON in params
    params = {'nalphas': nalphas, 'alphas': alphas,
              'sigma': sigma, 'prob_int': prob_int, 'prob_obs': prob_obs,
              'chisq_no_progress': chisq_no_progress, 'chisq_obs_agree': chisq_obs_agree,
              'constraint_target': constraint_target, 'mu': mu,
              'nrandom': nrandom, 'deterministicPerturb': deterministicPerturb,
              'minImprovement': minImprovement}  # parameters as actually used.
    if trace:
        print("----------- doGN params --------------------- ")
        print(params)
        print("------------------------------------------------------------")

    # augment vectors with constraint -- but have tricky problem of internal variability and constraint..
    # going on Kuniko's code looks like constraint ignored here. One way of doing that is to give it
    # zero wt in the inverse covariance matrix. Perhaps "augment" that matrix??

    if sigma:
        # got constraint so check have constraint_target and constraint not none
        if constraint_target is None or constraint is None:
            print("Sigma True but constraint_target not provided or constraint not set")
            raise ValueError()
            return "Fail"  # return failure

        if constraint.ndim == 2:
            use_UM_value = np.hstack([use_UM_value, constraint])  # add constraint value to end of UM value
        else:
            use_UM_value = np.hstack(
                [use_UM_value, constraint[:, np.newaxis]])  # add constraint value to end of UM value

        use_obs = np.hstack((use_obs, constraint_target))  # add constraint target to end of obs
        use_cov2 = np.zeros((nObs + 1, nObs + 1))  # generate bigger matrix to put modified covariance in,
        use_cov2[0:-1, 0:-1] = use_cov[:, :]  # std one
        use_cov2[-1, -1] = 2 * mu  ## include constraint in the covariance.
        use_cov = use_cov2
        del use_cov2  # remove the temp covariance so we can't accidentally do anything to it

    #  potentially regularize and invert covariance matrices.
    if covar_cond is not None:  # got requirement to regularize cov matrix?
        use_cov_iv = regularize_cov(use_cov_iv, covar_cond, trace=trace)
        use_cov = regularize_cov(use_cov, covar_cond, trace=trace)

    InvCov_iv = np.linalg.inv(use_cov_iv)  # invert it.
    InvCov = np.linalg.inv(use_cov)  # invert covariance matrix

    ## Compute constrained_error (and error) Code is lift from doGaussNewton
    err_constraint = calcErr(use_UM_value, use_obs, cov=use_cov)
    # compute error (which might include constraint)
    # now deal with constraint (in hacky way)
    if sigma:  # Need to compute unconstrained error.
        err = calcErr(use_UM_value[:, 0:-1], use_obs[0:-1], cov=use_cov[0:-1, 0:-1])  # without constraint
    else:
        err = err_constraint  # error and constrained error the same by defn.
    ## Compute best case for next iteration.
    # sub-sample only the linesearch values as UM_value contains all values from this iteration
    last_UM_value = use_UM_value[0, :]  # best previous state
    last_err_constraint = err_constraint[0]
    use_UM_value = use_UM_value[-nalphas:, :]  # last nalpbha values
    use_param_value = param_value[-nalphas:, :]
    err_constraint = err_constraint[-nalphas:]
    err = err[-nalphas:]

    if (nalphas) != use_UM_value.shape[0]:
        print("Expected ", nalphas, " columns got ", use_UM_value.shape[0])
        raise ValueError  # trigger error.
    index = np.argmin(err_constraint)
    nextIterValue = use_UM_value[index, :]
    nextIterParam = use_param_value[index, :]
    info = dict(bestrun=index, cov=use_cov, cov_iv=use_cov_iv, scalings=scalings,
                params=params)#, software_vn=version, revision=revision, SvnURL=svnURL)
    # decide if continue (or not)
    # Will go through tests computing various diagnostics
    # then we test to see if we should go on and if not why not.

    ## state no different? Note does not use constraint.
    delta = nextIterValue[0:nObs] - last_UM_value[0:nObs]  # Obs difference from last best est
    # ignoring constraint. Not sure if this appropriate...
    test_stat_state = 2. * delta.dot(InvCov_iv).dot(delta.T)
    info['test_no_progress'] = test_stat_state
    info['chisq_no_progress'] = chisq_no_progress

    ## Close to observations -- can use err_constraint  or err??
    delta = nextIterValue[0:nObs] - obs[0:nObs]  # difference from Obs est
    test_stat_obs = delta.dot(InvCov[0:nObs, 0:nObs]).dot(delta.T)
    info['test_obs_agree'] = test_stat_obs
    info['chisq_obs_agree'] = chisq_obs_agree
    NewParam = None  ## make it none by default.

    ## Now to decide if we continue
    pp = param_range[:,:] # copy it!

    if (err_constraint[index] + minImprovement > last_err_constraint):
        StatusInfo = 'Stalled'  # worse than previous state
    elif test_stat_state <= chisq_no_progress:
        StatusInfo = 'No State Change'  # close to previous state
    elif test_stat_obs <= chisq_obs_agree:  # close to target obs
        StatusInfo = 'Converged'
    else:  ## generate new paramters as continuing
        NewParam, randIndex = rangeAwarePerturbations(nextIterParam, param_range, step,
                                                      nrandom, deterministic=deterministicPerturb)
        # check new param are in range
        bad = (NewParam < param_range[:,0]) | (NewParam > param_range[:,1])  # outside range
        if np.any(bad):
            raise ValueError("Have NewParm params outside param_range")
        info['randIndex'] = randIndex  # store the random index if we had them.
        StatusInfo = 'Continue'

    info['StatusInfo'] = StatusInfo
    import numpy.testing as nptest
    nptest.assert_array_equal(param_range,pp)

    return StatusInfo, err, err_constraint, NewParam, index, nextIterParam, info


## slightly more logical would be NewParam, index,StatusInfo, err_constraint,info
# end of doLinesearch

def run_fn(function, params, npt, constraint_target=None):
    """
    Support function to run function in a way that all cases are ran before error happens..
    :param function: function to run
    :param params: parameter set for function
    :param npt -- number of obs values but not actually used.
    :param constraint_target: if not none then have a constraint
    :return: array of obs values and constraint.
    """
    if params is None:
        raise ValueError("Params are none.")

    # obsValues = np.zeros([params.shape[0], npt])
    obsValues = function(params)
    if np.any(np.isnan(obsValues)):
        # got some nan -- trigger error.
        raise ValueError
    return obsValues


def gaussNewton(function: typing.Callable,
                startParam: np.ndarray,
                paramRange: np.ndarray,
                paramStep: np.ndarray,
                target: np.ndarray,
                optimise: typing.Mapping,
                cov: typing.Optional[np.ndarray] = None,
                cov_iv: typing.Optional[np.ndarray] = None,
                scalings: typing.Optional[np.ndarray] = None, constraint_target: typing.Optional[float] = None,
                trace: bool = False):
    """
    Apply guassNewton/Linesearch algorithm to specified function.
    :param function: function to be optimised. Should take a numpy array of N values and return a M length array of observations
    :param startParam: a length N numpy array of the starting parameters for the optimisation
    :param paramRange: a Nx2 numpy array of the minimum parameter values [*,0] and maximum parameter values [*,1]
    :param paramStep: The perturbation to be made to each parameter -- note that algorithm development could work this out automatically.
    :param target: The target for the optimisation. A len M numpy array.
    :param optimise: A dict of information used by optimisation.
    :param cov: (default None)  Covariance matrix for scaling cost function. Default values sets by doGaussNewton and doLineSearch
    :param cov_iv: (default None) Covariance  matrix used in doLinesearch to determine if values changed enough.
    :param scalings : (default is 1) Scalings to apply to simulated observations and targets. A len M numpy array
    :param constraint_target : (Optional -- default is None) If provided the target value for the constraint
    :param trace: provide more trace information
    :return: Returns minimal error param values,  status of termination & information on GN/LS cpts of algorithm
    """

    # stage 0 -- setup
    maxIterations = optimise.get("maxIterations")
    if maxIterations is not None and trace:
        print("Max Iterations is ", maxIterations)
    nrandom = optimise.get('nrandom', None)
    deterministicPerturb = optimise.get('deterministicPertub', True)
    statusInfo = 'Continue'
    npt = len(target)  # how many points we expect.
    if constraint_target is not None: npt += 1  # increment number of points to deal with constraint
    iterCount = 0  # how many iterations we done.
    nFail = 0  # how many cases have failed since last sucess
    totalFail = 0  # total number of failure we have had.
    prevBestParam = startParam[:]  # copy startParam so have a prevBestParam if needed.

    # stage 1 -- Work out parameters for first iteration
    paramsGN, randIndx = rangeAwarePerturbations(startParam, paramRange, paramStep,
                                                 nrandom=nrandom, deterministic=deterministicPerturb, trace=trace)
    statusList = []  # a list of the status
    while statusInfo == 'Continue':
        # obsValuesGN, constraintGN = run_fn(function, paramsGN, npt,
        #                                   constraint_target=constraint_target)  # run the functions.
        obsValuesGN = run_fn(function, paramsGN, npt,
                             constraint_target=constraint_target)  # run the functions.
        optStatus, paramsLS, err, err_constraint, infoGN = \
            doGaussNewton(paramsGN, paramRange, obsValuesGN, target, cov=cov, scalings=scalings,
                          # constraint=constraintGN, constraint_target=constraint_target,
                          studyJSON=optimise, trace=trace)  # run GN
        # add some more information to the info dict.
        infoGN['err_constraint'] = err_constraint
        infoGN['obsValues'] = obsValuesGN
        infoGN['paramValues'] = paramsGN
        if trace:  # print(out some information)
            print("GN: paramValues: \n", paramsLS)  # , " err_constraint", err_constraint[0])

        # run the functions on the linesearch values
        if paramsLS is None: # failed in some way.
            break # exit the loop.

        obsValuesLS = run_fn(function, paramsLS, npt, constraint_target=constraint_target)
        # need to merge paramsGS and paramsLS, obsValesGN & obsValuesGN & constraintGN and constraintLS
        params = np.vstack((paramsGN, paramsLS))
        obsValues = np.vstack((obsValuesGN, obsValuesLS))
        # constraint = np.hstack((constraintGN, constraintLS))
        statusInfo, err, err_constraint, paramsGN, index, bestParam, infoLS = \
            doLineSearch(params, paramRange, obsValues, target, paramStep, cov=cov, cov_iv=cov_iv,
                         scalings=scalings,
                         studyJSON=optimise, trace=trace)  # run LS

        # add some information to the LineSearch info dict.
        infoLS['err_constraint'] = err_constraint
        infoLS['paramValues'] = paramsLS
        infoLS['obsValues'] = obsValuesLS
        statusList.append({'gaussNewton': infoGN, 'lineSearch': infoLS})
        iterCount += 1  # increase iteration count
        if (maxIterations is not None) and (iterCount >= maxIterations):
            if trace:
                print(f"Done {iterCount} iterations which is > {maxIterations}. Stopping.")
            if statusInfo == 'Continue':
                if trace:
                    print(" Ran out of iterations")
                statusInfo = 'Failed'
                break
        if trace:
            print("LS: statusInfo %s Iter: %d Err_constraint" % (statusInfo, iterCount), err_constraint)

        if statusInfo == 'Continue' or statusInfo == 'Converged':
            nFail == 0  # reset failure count as we are ok
            prevBestParam = bestParam[:]  # update prevBestParam in case we restart
        else:  # we've failed...
            nFail += 1  # increment failure count
            totalFail += 1  # and total fail count
            if (nrandom is not None) and (nFail < optimise.get('maxFails', 0)):  # random perturbation so allow retry
                # generate list of perturbed parameters.
                # the tricky issue here is that if we are running deterministically then
                # we will always get the same parameters perturbed...
                # Will hack this by passing a number to deterministic
                # then using that to increment the RNG.
                params, randIndx = rangeAwarePerturbations(prevBestParam, paramRange, paramStep, nrandom=nrandom,
                                                           deterministic=totalFail + 1, trace=trace)
                statusInfo = 'continue'  # keep going.

        if trace:
            print("prevBestParam on iter %i is " % iterCount, prevBestParam)

    # end of iterative loop running doGaussNewton and doLineSearch.

    # rearrange the info array
    # start with the err_constraint from lineSearch
    if nrandom is not None:
        raise NotImplementedError(
            "Need to make info code work with random algorithm...If you don't care switch this off")

    jacobian = []
    hessian = []
    alpha = []
    err_constraint = [statusList[0]['gaussNewton']['err_constraint'][0]]
    bestParams = [startParam]
    iter = np.arange(len(statusList))
    for iterInfo in statusList:
        jacobian.append(iterInfo['gaussNewton']['jacobian'])
        hessian.append(iterInfo['gaussNewton']['hessian'])
        bestAlpha = iterInfo['lineSearch']['bestrun']
        err_constraint.append(iterInfo['lineSearch']['err_constraint'][bestAlpha])
        bestParams.append(iterInfo['lineSearch']['paramValues'][bestAlpha])
        alpha.append(iterInfo['lineSearch']['params']['alphas'][bestAlpha])

    # err_constraint=err_constraint[:-1]
    ## we don't want the last linesearch values...
    bestParams = bestParams[:-1]
    # for var in (jacobian,hessian,bestAlpha,err_constraint,bestParams,alpha):
    #    var=np.asscalar(var)

    jacobian = np.asarray(jacobian)
    hessian = np.asarray(hessian)
    err_constraint = np.asarray(err_constraint)
    bestParams = np.asarray(bestParams)
    alpha = np.asarray(alpha)
    statusList = {'jacobian': jacobian, 'hessian': hessian, 'alpha': alpha, 'err_constraint': err_constraint,
                  'iter': iter, 'bestParams': bestParams}

    return prevBestParam, statusInfo, statusList  # would also like to return a bunch of info to help trace the performance of the algorithm.


# new Jacobian support stuff.
def rangeAwarePerturbations2(baseVals, parLimits, steps, nrandom=None,
                             deterministic=True, trace=False):
    """
    Generate perturbations towards the centre of the valid range for each
    parameter

    Inputs

         baseVals: array of parameters for the first of the GN runs (unperturbed)

         parLimits: max,min as read from JSON into 2-D array,
                    and passed through to here unaltered

         steps: array of stepsizes for each parameter.

         nrandom: perturb nrandom parameters rather than the full set.
         deterministic: optional argument with default value True -- make random choice of parameters deterministic.
            deterministic will be added to the RNG seed. So to modify what you get change the value of deterministic.
            It is a bit of a hack though...

         trace: Optional argument with default value False.
                Boolean -- True to get output, False no output

    Note inputs are all arrays, with data in order of the parameter list.

    Returns
          y: array defining the pertrubed parameter  parameter values
             with dimension nparameters
    """
    if trace:
        print("in rangeAwarePerturbations from baseVals")
        print(baseVals)

    # derive the centre-of-valid-range

    centres = (parLimits[:, 0] + parLimits[:, 1]) * 0.5

    deltaParam = np.abs(steps)
    sgn = np.sign((centres - baseVals))
    L = (sgn < 0)
    deltaParam[L] *= -1
    # possibly perturb the array
    if nrandom is not None:
        deltaParam, randIndx = randSelect(deltaParam, nrandom, deterministic=deterministic, trace=trace)
    else:
        randIndx = None

    if trace:
        print("leaving rangeAwarePerturbations with derived perturbed runs:")
        print(deltaParam)

    return deltaParam, randIndx


def runJacobian(function, startParam, deltaParam, paramRange, *args,
                obsNames=None, nEnsemble=1, verbose=False, returnVar=False, **kwargs):
    """
    compute Jacobian from given function. Really a fairly thin layer to allow optimise dict to be unpicked.
    :param function: function to be ran -- should call appropriate model simulations
    :param startParam:  start parameter as a pandas Series
    :param deltaParam:  perturbations as a pandas Series
    :param paramRange: paramRange as a pandas dataFrame

    Other unnamed arguments are passed to jacobian (and then to function)

    :keyword args

    :param obsNames (default None). Names of observations -- used to label the xarray. If not passed then will be 0,1,2...
    :param nEnsemble (default 1). Jacobian will be computed multiple times with parameter ensembleMember=X being passed to function.
    :param verbose (default False). If set then more verbose information
    :param returnVar (default False). If true then return the variance.

    other keyword arguments are passed to function.

    :return: jacobian (wrapped as an xarray)
    """

    extraArgs = kwargs.copy()
    dp, indx = rangeAwarePerturbations2(startParam.values, paramRange.loc[['minParam', 'maxParam']].values.T,
                                        deltaParam.values)  # compute the perturbations.

    j = []  # empty list. At the end will convert it.
    for ensembleMember in range(0, nEnsemble):
        extraArgs.update(ensembleMember=ensembleMember)
        jac = GNjacobian(function, startParam.values, dp, verbose=True, **extraArgs)
        j.append(jac)

    if obsNames is not None:
        use_obsNames = obsNames
    else:
        use_obsNames = ['Obs' + str(i) for i in np.arange(0, j[0].shape[1])]

    coords = (('parameter', startParam.index), ('Observation', use_obsNames))  # co-ord info for xarray
    if nEnsemble > 1:
        result = np.array(j).mean(0)  # average across ensembles.
        if returnVar:
            var = np.array(j).var(0)  # compute variance.
            var = xarray.DataArray(var, coords)  # convert to xarray
    else:
        result = j[0]  # only one ensemble
    result = xarray.DataArray(result, coords)  # convert to xarray

    if returnVar:
        result = xarray.Dataset({'Jacobian': result})
        if nEnsemble > 1:
            result['Jacobian_var'] = var
    return result  # return it.


def GNjacobian(function, startParam, deltaParam, *extraArgs, paramIndex=None,
               verbose=False, **kwargs):
    """
    Compute Jacobian for specified function.
    :param function: function to be used. Should take a numpy array of NxM values and return a N*Y Jacobian array
    :param startParam: a length M numpy array of the reference parameters for the Jacobian calculation
    :param deltaParam: a length N array of parameter perturbations.

    Optional parameters

    :param verbose: provide more trace. Default False
    :param extraArgs: List of extra arguments -- passed through to function call.

    :return: Returns Jacobian as a NxY array
    """

    # stage 0 -- setup

    if paramIndex is None:  # not random selection..
        paramIndex = np.arange(0, len(startParam))  # will just select everything!

    params = startParam + np.vstack((np.zeros(startParam.size), np.diag(deltaParam)[paramIndex, :]))
    # logging.debug("Jacobian: parameters for Jacobian are " + repr(params)) # this coming out when not wanted.

    simValues = function(params, *extraArgs, **kwargs)  # run the function on the parameters
    jacobian = simValues[1:, :] - simValues[0, :]
    #
    # scales = kwargs.get('scales',None)
    # if scales is not None:
    #     jacobian = jacobian*scales.reshape(1,-1) # scale jacobian
    # left commented for now. scaling is really to help with regularisation and plotting.
    # so will leave it out. StudyConfig will (eventually)  apply scaling if present.

    jacobian = jacobian / deltaParam[paramIndex].reshape(-1, 1)
    if np.any(np.isnan(jacobian)):
        logging.info('Jacobian: Found NaN and raising ValueError')
        raise ValueError
    return jacobian

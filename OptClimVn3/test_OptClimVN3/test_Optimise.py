import unittest

import numpy as np
import numpy.testing as nptest

from Optimise import doGaussNewton, calcErr, doLineSearch,  randSelect, gaussNewton, runJacobian, \
    GNjacobian
from ref_code import doGaussNewton_ref, doLineSearch_ref  ## import reference code.

__author__ = 'stett2'


## tests for calcErr

def optfunction(params, *extraArgs, randomScale=1e-9, **kwargs):
    """
    Test fn for optimising...
    :param params: np array of parameter values.
    :*extraArgs -- extra arguments. Not used in this
    :**kwargs -- keyword arguments not actually used either.
    :return:
    """
    import numpy.random as random
    if (params.ndim == 1):
        result = np.array([np.sum(params ** 3), np.sum(params ** 2), np.sum(np.sqrt(np.abs(params))), np.sum(params)])
    else:
        result = np.array(
            [np.sum(params ** 3, 1), np.sum(params ** 2, 1), np.sum(np.sqrt(np.abs(params)), 1), np.sum(params, 1)])
        result = result.T

    if kwargs.get('ensembleMember') is not None:  # got an ensemble member
        # initialise RNG and add a small random perturbation to result.
        # probably should be done by each param vector separately.
        # so then result would be deterministic.
        perturb = np.zeros(result.shape)
        maxSeed = 2 ** (32 - 8) - 1  # allow up to 256 ensembleMembers -- might work beyond this...
        for x in range(0, params.shape[0]):  # iterate over params.
            seed = 0
            seed += int(np.product(params[x, :]).view(np.uint64))
            seed += int(np.sum(params[x, :]).view(np.uint64))
            while (seed > maxSeed):
                seed = seed // 2
            seed = seed * 256 + kwargs.get('ensembleMember')  # add in the ensemble member,..
            rng = random.RandomState(seed)  # get a RNG class Seed is comb of parameters and ens member.
            perturb[x, :] = rng.normal(0.0, randomScale, result.shape[1])  # random small perturbations.

        result += perturb

    return result

class TestcalcErr(unittest.TestCase):

    def test_single_sim_rms(self):
        """ Test get expected result for no covariance supplied with single simuln. Result should be RMS diff"""
        npt = 10
        delta = np.ones(npt)
        obs = np.random.uniform(size=(npt))
        sim = obs + delta
        err = calcErr(sim, obs)
        expect = np.sqrt(np.mean(delta ** 2))
        nptest.assert_allclose(expect, err)

    def test_rms(self):
        """ Test get expected result for unit  covariance supplied with multiple simulations.
        Result should be RMS diff"""

        npt = 10
        cov = np.diag(np.ones(npt))
        dv = np.arange(1, 4.)
        delta = np.outer(dv, np.ones(npt))
        obs = np.random.uniform(size=(npt))
        sim = np.reshape(obs, (1, npt)) + delta
        err = calcErr(sim, obs, cov)
        expect = np.sqrt(np.mean(delta ** 2, axis=1))
        nptest.assert_allclose(expect, err)

    ## Tests for doGaussNewton & doLineSearch


class TestGNLS(unittest.TestCase):
    # to debug an individual test do the following:
    # TestGaussNewton('test_transformed').debug()
    # or just use PyCharm which seems to automatically run tests...
    def setUp(self):
        """ setup for each test """

        self.studyJSON = {"some info": 'fred', 'alphas': [1.0, 0.7, 0.3], 'terminus': 0.5,
                          'constraint_target': 1.0, 'sigma': 1, 'mu': 0.1}
        #               'covar_cond':None,'prob_int':0.5,'prob_obs':0.5}
        self.rtol = 5e-5  # relative tolerance for comparison
        self.atol = 5e-5  # absolute tolerance
        self.nobs = 4
        self.nparam = 5
        self.nsamp = 500  # how many  random samples we should do.
        self.obs = np.zeros(self.nobs)
        self.optFunction = optfunction  # function we will be running -- should return same no of obs sa obs.
        # nparam params, nobs obns and one constraint
        # signs=np.random.permutation(np.hstack((np.repeat(1.0,self.nobs-1),np.repeat(-1.0,1))))
        self.param_range = np.vstack([np.repeat(-20.0, self.nparam), np.repeat(20.0, self.nparam)]).T
        # parameter range; 0th column is min value; 1st column max value
        # work out parameters
        self.orig_param = 0.25 * np.linspace(5, 6., num=self.nparam)  # start parameter values
        self.step = np.repeat(1.0, self.nparam)  # steps wanted in Jacobian calc.
        self.step[0] = 2.0  # change the 0th step
        self.param_value = np.vstack([np.zeros(self.nparam), np.diag(self.step)]) + self.orig_param
        # self.param_value are the synthetic parameters needed for the Jacobian calc.

        # work out synthetic observations
        self.constraint = np.zeros(self.nparam + 1);
        self.constraint[self.nparam] = 0.5  # set up constraint
        self.constraint_target = self.studyJSON['constraint_target']  # NB no longer in example JSON
        signs = np.hstack((np.repeat(1.0, self.nobs - 2), np.repeat(-1.0, 2)))  # which way obs go
        self.UM_value = np.vstack([np.zeros(self.nobs), np.diag(signs), np.zeros(self.nobs)]) + \
                        np.linspace(5, 10., num=self.nobs)  # simulated observations
        self.olist = ['obs' + str(x) for x in range(self.nobs)]  # names of obs
        self.obs = 2 * np.ones(self.nobs) + np.linspace(5, 10., num=self.nobs)  # tgt obs values
        self.cov = np.identity(self.nobs)  # unit covariance for obs uncertainty
        self.scalings = np.ones(self.nobs)

        # compute expected values for linesearch parameters given values above.
        # Jacobian, including constraint, is square and diagonal.
        # d obs/d param is [signs,0.5]/step and tgt is [2 * nobs, 1.0]+start_obs
        # parameter change is then step*[2*nobs,1]/[signs,0.5]
        tlst = []
        s_vector = self.step * np.hstack((2 * np.ones(self.nobs), 1)) / np.hstack(
            (signs, 0.5))  # how we perturb the parameters to get to zero error.
        for alpha in self.studyJSON['alphas']:
            tlst.append(self.orig_param + alpha * s_vector)
        best_case = self.orig_param + s_vector  # best case we expec

        self.expect_param = np.vstack(tlst)  # make it one big array
        # now make LS things...Perhaps put LS stuff in its own class????
        self.nalpha = len(self.studyJSON['alphas'])
        self.LSparam_value = np.vstack((self.param_value, self.expect_param))
        self.LSUM_value = np.vstack((self.UM_value, self.obs + np.outer([1, 2, 5], np.repeat(1, self.nobs))))
        self.LSconstraint = np.hstack((self.constraint, np.repeat(1.1, self.nalpha)))
        self.LSconstraint_target = self.studyJSON['constraint_target']  # NB no longer in example JSON
        self.LScov_iv = np.identity(self.nobs) / 4  # 1/4 covariance for obs uncertainty

        # all base params are greater than mid-point so perturbations move towards zero.
        sign = np.where(self.expect_param[0, :] >= 0.0, -1, 1)  # move things towards center
        self.LSexpect_param = np.vstack((self.expect_param[0, :],
                                         self.expect_param[0, :].T + sign * np.diag(np.abs(self.step))))

    def run_gauss_newton(self, ref=False, trace=False):
        """ Run gaussnewton (or ref_doGaussNewton) and pacakge results up in a dictionary """
        if ref:
            ## reference version of fn is not pure. Think that my version is...
            status, y, err, err_constraint, info = doGaussNewton_ref(
                self.param_value.copy(), self.param_range.copy(), self.UM_value.copy(), self.obs.copy(),
                self.cov.copy(), self.scalings.copy(), self.olist[:],
                self.constraint.copy(), self.studyJSON.copy(), trace=trace)
        else:
            status, y, err, err_constraint, info = doGaussNewton(
                self.param_value, self.param_range, self.UM_value, self.obs,
                self.cov.copy(), self.scalings, self.olist,
                self.constraint, self.constraint_target, self.studyJSON, trace=trace)

        return {'status': status, 'linesearch': y, 'error': err,
                'constrained_error': err_constraint, 'info': info}

    def compare_std_ref(self, std, ref):
        """ Compare output from standard and reference codes. Std code output gets some processing to make it compatable with reference code.
            Transforms are:
            1) linesearch values in std case are transposed to make them compatable with reference case
            2) Jacobian in std code has last row (corresponding to constraint) removed. """

        names = ('linesearch', 'constrained_error')
        k = 'status'
        self.assertEqual(ref[k], std[k], msg='Stats differ std=' + repr(std[k]) + " ref= " + repr(ref[k]))
        if std[k] == 'Fatal':
            return  ## parameters/values don't really make sense any more.
        for k in names:  # these should all be numpy arrays
            if type(ref[k]) == np.ndarray:  # numpy array
                if (k == 'linesearch'):
                    std[k] = std[k].T  # transpose result

                nptest.assert_allclose(ref[k], std[k],
                                       err_msg='reference and modified case not the same for ' + k,
                                       rtol=self.rtol, atol=self.atol)
            else:
                self.assertEqual(ref[k], std[k], msg='reference and modified case not the same for ' + k)
        keys = ("jacobian", "hessian", "condnum")  # keys in dictionary we want to iterate over.
        for k in keys:  # iterate over specified keys -- suspect there more pythonic way of doing this...
            ref_v = ref['info'][k]
            std_v = std['info'][k]
            if (k == 'jacobian'):
                std_v = std_v.T  # need to transpose it too..
                std_v = std_v[0:-1, :]  # remove last row corresponding to constraint

            nptest.assert_allclose(ref_v, std_v,
                                   err_msg='info[' + k + '] differs ', rtol=self.rtol, atol=self.atol)


    def run_LineSearch(self, ref=False, trace=False):
        """ Run doLineSearch (or ref_doLineSearch) and pacakge results up in a dictionary
        :param ref: if set True run reference code. Default is False
        :param trace: If set True more information is output. Default is False.
        """
        if ref:
            ## reference code modifies parameters. So copy stuff
            self.studyJSON[
                'covIV'] = self.LScov_iv.copy()  # hack as reference code expects this to be in the options dict passed in...
            status, err, err_constraint, NewParam, index, info = doLineSearch_ref(
                self.LSparam_value.copy(), self.param_range.copy(), self.LSUM_value.copy(), self.obs.copy(),
                self.cov.copy(), self.scalings.copy(), self.olist[:],
                self.LSconstraint.copy(), self.studyJSON.copy(), trace=trace)
        else:
            status, err, err_constraint, NewParam, index, bestParam, info = doLineSearch(
                self.LSparam_value, self.param_range, self.LSUM_value, self.obs,
                self.step,
                cov=self.cov, cov_iv=self.LScov_iv, scalings=self.scalings,
                olist=self.olist,
                constraint=self.LSconstraint, constraint_target=self.LSconstraint_target, studyJSON=self.studyJSON,
                trace=trace)

        return {'status': status, 'NewParam': NewParam, 'error': err,
                'constrained_error': err_constraint, 'info': info, 'index': index}

    def compare_LS_std_ref(self, std, ref):
        """ Compare output from standard and reference LineSearch codes. Std code output gets some processing to make it compatable with reference code.
            Transforms are:
            1) NewParam array is transposed.
        """
        names = ('NewParam', 'error', 'constrained_error', 'index')
        k = 'status'

        self.assertEqual(ref[k], std[k], msg='Stats differ std=' + repr(std[k]) + " ref= " + repr(ref[k]))
        for k in names:  # these should all be numpy arrays
            ## fix newparam
            if k == 'NewParam' and std[k] is None:
                std[k] = []  # make it an empty array

            if type(ref[k]) == np.ndarray:  # numpy array
                # add in any transofmrations you want in here...
                if k == "NewParam":
                    ref[k] = ref[k].T  # transpose the result
                    ##ref[k] = ref[k][-self.nalpha:,:] # get rid of the GN values.

                nptest.assert_allclose(ref[k], std[k],
                                       err_msg='reference and modified case not the same for ' + k,
                                       rtol=self.rtol, atol=self.atol)
            else:
                self.assertEqual(ref[k], std[k], msg='reference and modified case not the same for ' + k)
        for k in ['bestrun']:  # keys to compare
            self.assertEqual(ref['info'][k], std['info'][k], msg=' std and reference differ for info[' + k + ']')

    def test_compare_ref(self):
        """ Test that modified code produces same results as new code """
        std = self.run_gauss_newton()
        ref = self.run_gauss_newton(ref=True)
        self.compare_std_ref(std, ref)

    def test_val_eq_tgt(self):
        """ Test Case when param values = tgt values. Should have zero linesearch"""
        self.UM_value[0, :] = self.obs[:]
        self.constraint[0] = self.studyJSON['constraint_target']
        result = self.run_gauss_newton()
        for i in range(result['linesearch'].shape[0]):
            nptest.assert_allclose(result['linesearch'][i, :], self.param_value[0, :],
                                   err_msg='expected all LS to be same as initial')

    def test_all_same(self):
        """ Test Case when alpha values all the same. Should have same values for results"""
        self.studyJSON['alphas'] = [1.0, 1.0, 1.0]
        result = self.run_gauss_newton()
        for i in range(1, result['linesearch'].shape[0]):
            nptest.assert_allclose(result['linesearch'][i, :], result['linesearch'][0, :],
                                   err_msg='expected all param vectors to be identical')

    def test_param_error(self):
        """ Test that when min and max values swapped get an error """
        self.param_range = (np.array([self.param_range[:, 1], self.param_range[:, 0]])).T
        with self.assertRaises(ValueError):
            self.run_gauss_newton()

    def test_constraint_error(self):
        """ Check when sigma defined and constraint is none get an error """
        self.param_range = (np.array([self.param_range[:, 1], self.param_range[:, 0]])).T
        self.constraint = None
        with self.assertRaises(ValueError):
            self.run_gauss_newton()

    def test_scaleparams(self):
        """ Test case when scale one of the parameters
        which applies to param_value, param_range and expected results. """

        scale = np.ones((self.nparam, 1))
        scale[0:2] = np.reshape([1e-4, 1e-7], (2, 1))
        self.param_value *= scale.T
        self.param_range *= scale
        self.expect_param *= scale.T
        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], self.expect_param, rtol=1e-9)

    def test_obsscale(self):
        """ Test observed scaling doesn;t change result """
        ## scale the observations via the scalings parameter.
        self.scalings[0] = 10
        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], self.expect_param, rtol=1e-9)

    def test_simplecase(self):
        """ Test get expected soln when specify params and obs """

        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], self.expect_param, rtol=1e-9)  # values for LS as expected.

    def test_constraint2D(self):
        """ Test get expected soln when specify params and obs and constraint is 2D"""

        self.constraint = self.constraint[:, np.newaxis]  # add an unit dim.
        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], self.expect_param, rtol=1e-9)  # values for LS as expected.

    def test_transformed(self):
        """ Test still get expected results when obs, simulated obs and covariances linearly transformed.
        Easy to show that under the transformation
        r->Mr, J->MJ and C-> MCM^T as long as M is invertable that results don't change. """

        # Generate  a symmetric matrix (which should be invertable)
        M = np.diag(1. + np.arange(0.0, self.nobs) / float(self.nobs))
        for i in range(0, self.nobs - 1):
            # doing the following on each iteraition
            # [ X r r r r r]
            # [r  ....... ]
            # [r ......]
            # [r ........]
            M[i, i + 1:] = np.random.uniform(size=(self.nobs - 1 - i))  # above diagonal values
            M[i + 1:, i] = M[i, i + 1:]  # below diagonal values

        M = M / np.linalg.det(M)
        # transform simulated values
        for i in range(0, self.nparam + 1):
            self.UM_value[i, :] = self.UM_value[i, :].dot(M)  # transform simulated values
        self.obs = M.dot(self.obs)  # and obs
        self.cov = M.dot(self.cov).dot(M.T)  # and covariances
        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], self.expect_param, rtol=1e-6)

    def test_noconstraint(self):
        """ Results without constraint should be the same as with constraint
        given  appropriate covariance and value added to obs """
        # set up simulated observations array
        UM_value = np.zeros((self.nparam + 1, self.nobs + 1))  # empty array we fill in..
        UM_value[:, 0:-1] = self.UM_value
        UM_value[:, -1] = self.constraint
        self.UM_value = UM_value
        # Do similar trick for covariance matrin
        cov = np.zeros((self.nobs + 1, self.nobs + 1))
        cov[0:-1, 0:-1] = self.cov
        cov[-1, -1] = self.studyJSON['mu'] ** 2  # covariance is mu^2 giving wt of 1/mu^2
        self.cov = cov
        # Add extra element (corresponding to constraint)  to vectors
        self.obs = np.hstack((self.obs, self.studyJSON['constraint_target']))
        self.scalings = np.hstack((self.scalings, 1.0))
        self.olist.append('Constraint')

        # clean up control dictionary.
        del self.studyJSON['sigma']  # no constraint
        del self.studyJSON['mu']  # no constraint
        del self.studyJSON['constraint_target']  # no constraint
        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], self.expect_param, rtol=1e-9)

    def test_param_range(self):
        """ Test that if over parameter ranges that we stick at boundary """
        self.param_range[-3:-1, 0] = 0
        self.param_range[0:-3, 1] = 3
        self.expect_param = np.maximum(self.param_range[:, 0],
                                       np.minimum(self.expect_param, self.param_range[:, 1]))

        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], self.expect_param, rtol=1e-9)

    def test_regularise(self):
        """ Test that regularisation  is happening to hessian. 
        Requires generation of a dataset that produces an ill-conditioned Hessian
        Which done by generating a random Jacobian. Then SVD decompose this and adjust 
        singular values so have the right amount of ill conditioning. Compute the expected result with 
        regularisation applied. Then compute the UM_values by multiplying the Jacobian by the 
        paramaeter differences and work as before"""

        rand_jac = np.random.uniform(0.0, 2.0, (self.nparam, self.nobs))
        Jacobian = rand_jac
        hessian = Jacobian.dot(Jacobian.T) / self.nobs
        con = np.linalg.cond(hessian) / self.nobs
        crit_con = 10e10
        eye = np.identity(self.nparam)
        for k in range(-7, -2):  # this code feels like too much of a copy of code in doGaussNewton..
            if con <= crit_con:
                break  # exit loop
            perJ = hessian + eye * 10 ** k
            con = np.linalg.cond(perJ)  # compute condition number
        if (con > crit_con):  # check we managed to regularize
            print("failed to regularize")
            raise ValueError

        hessian = perJ
        F = (self.UM_value[0, :] - self.obs) / self.scalings
        s = -np.linalg.solve(hessian, Jacobian.dot(F)) / float(self.nobs)
        tlst = []
        for alpha in self.studyJSON['alphas']:
            tlst.append(self.orig_param + s * alpha)
        self.expect_param = np.vstack(tlst)
        # need to get rid of the constraint value as we are keeping that out..
        self.UM_value[0, :] = self.UM_value[0, :]  # reference value
        dparam = np.diag(self.param_value[1:, :] - self.param_value[0, :])
        for i in range(1, self.nparam + 1):
            self.UM_value[i, :] = Jacobian[i - 1, :] * dparam[i - 1] + self.UM_value[0, :]
        self.studyJSON['sigma'] = 0
        self.constraint = None
        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], self.expect_param, rtol=1e-5, atol=1e-5)

    def test_random_comp_ref(self):
        """ Generate Random set of values and then compare new and old values.
             Do multiple times (set by variable self.nsamp) """
        trace = False  # set true to give trace..
        scale = np.ones((self.nparam, 1))
        scale[1:3] = np.reshape([1e-2, 1e-3], (2, 1))

        for samp in range(0, self.nsamp):
            self.setUp()  # make sure all values setup.
            self.param_value *= scale.T
            self.param_range *= scale
            rand_case = np.random.uniform(1.0, 10.0, (self.nparam, self.nobs))
            dparam = np.diag(self.param_value[1:, :] - self.param_value[0, :])
            M = np.diag(1. + np.arange(0.0, self.nobs) / float(self.nobs))
            M += np.random.uniform(-1, 1, (self.nobs, self.nobs))
            M = (M + M.T) / 2.
            M = M / np.linalg.det(M)  # make sure it has unit determinant.
            self.obs = M.dot(self.obs)  # transform obs
            self.cov = M.dot(self.cov).dot(M.T)  # and covariances
            # compute and transform simulated values
            UM_value = np.zeros((self.nparam + 1, self.nobs))
            UM_value[0, :] = self.UM_value[0, :].dot(M)  # reference value
            for i in range(1, self.nparam + 1):
                UM_value[i, :] = rand_case[i - 1, :] * dparam[i - 1] + UM_value[0, :]
                UM_value[i, :] = UM_value[i, :].dot(M)
            self.UM_value = UM_value
            self.scalings[0] = 10  # scale one of the obs
            ref = self.run_gauss_newton(ref=True, trace=trace)
            std = self.run_gauss_newton(trace=trace)
            self.compare_std_ref(std, ref)

    def test_randSelect(self):
        """
        Test that randSelect works -- perturbing a random number
        :return:
        """

        # case 1 nrandom = 0
        with self.assertRaises(AssertionError):
            param = randSelect(self.LSexpect_param, 0, deterministic=True)
        # case 1 nrandom = self.nparam # which could be sensible...
        with self.assertRaises(AssertionError):
            param = randSelect(self.LSexpect_param, self.nparam, deterministic=True)

        # Should get expected parameters sub-sampled by indx
        nrandom = int(self.nparam / 2)
        param, indx = randSelect(self.LSexpect_param, nrandom, deterministic=True)
        self.assertEqual(len(indx), nrandom + 1)
        nptest.assert_allclose(param, self.LSexpect_param[indx, :], atol=self.atol, rtol=self.rtol,
                               err_msg='param and expected not close')
        # and check that deterministic calculation works.
        param2, indx2 = randSelect(self.LSexpect_param, nrandom, deterministic=True)  # run it again
        nptest.assert_allclose(param2, param, atol=self.atol, rtol=self.rtol,
                               err_msg='param and param2 not close')
        nptest.assert_allclose(indx2, indx, atol=self.atol, rtol=self.rtol,
                               err_msg='indx and indx2 not close')

    def test_GN_nrand(self):
        """
        tests that GN works if only given a sub-set of rows (corresponding to different params)
        :return:
        """
        indx = np.array([0, 2, 3])  # only two parameters passed in
        self.UM_value = self.UM_value[indx, :]
        self.constraint = self.constraint[indx]
        # only expect parameters in the sub-space to be modified.
        # expect=np.broadcast_to(self.param_value[0,:],self.expect_param.shape).copy() # copy first element
        expect = np.zeros(self.expect_param.shape)
        expect[:, :] = self.param_value[0, :]
        expect[:, indx[1:] - 1] = self.expect_param[:, indx[1:] - 1]  # values where we change parameters.
        self.param_value = self.param_value[indx, :]
        self.run_gauss_newton(trace=False)
        result = self.run_gauss_newton()
        nptest.assert_allclose(result['linesearch'], expect, rtol=self.rtol, atol=self.atol)

    def test_compare_LSref(self):
        """ Test that modified code produces same results as new code """
        std = self.run_LineSearch()
        ref = self.run_LineSearch(ref=True)
        self.compare_LS_std_ref(std, ref)

    def test_GN_optional_params(self):
        """ Test that optional parameters work in GN as expected..."""
        pass

    def test_LS_continue(self):
        """ Test that LS continuing works """
        # default parameters lead to continue

        std = self.run_LineSearch()
        self.assertEqual(std['status'], "Continue", msg='Status not as expected')
        nptest.assert_allclose(std['NewParam'], self.LSexpect_param,
                               rtol=self.rtol, atol=self.atol, err_msg="NewParam not as expected")
        ref = self.run_LineSearch(ref=True)
        self.compare_LS_std_ref(std, ref)

    def test_LS_states_same(self):
        """ Test that LS states are the same get detected """
        # need to modify linesearch UM_values[-3,:] and constraint[-3] to be the same as first case +0.1
        self.obs = self.LSUM_value[0, :] + 2  # make sure obs bigger so when we perturb base case we improve it.
        self.LSUM_value[-3, :] = self.LSUM_value[0, :] + 0.1  # make it a bit bigger and so closer to obs
        self.constraint[:] = self.studyJSON['constraint_target']  # make all constraints OK
        std = self.run_LineSearch()
        self.assertEqual(std['status'], "No State Change", msg='Status not as expected')
        self.assertEqual(std['NewParam'], None, msg='Expected None for NewParam')
        ref = self.run_LineSearch(ref=True)
        self.compare_LS_std_ref(std, ref)

    def test_LS_converge(self):
        """   Test that LS convergence to obs works  """
        # need to modify linesearch UM_values[-3,:] and constraint[-3] to be the same as obs_tgt +0.1
        self.LSUM_value[-2, :] = self.obs + 0.1
        self.LSconstraint[-2] = self.studyJSON['constraint_target'] + 0.1
        std = self.run_LineSearch()
        self.assertEqual(std['status'], "Converged", msg='Status not as expected')
        self.assertEqual(std['NewParam'], None, msg='Expected None for NewParam')
        ref = self.run_LineSearch(ref=True)
        self.compare_LS_std_ref(std, ref)

    def test_LS_stalled(self):
        """ Test that LS stalling is detected """
        # need to modify linesearch UM_values[-3:,:] and constraint[-3:] to be the same as first case -1, 2, -3
        self.obs = self.LSUM_value[0, :] + 2  # make sure obs bigger so when we perturb base case -we we worsen  it.
        self.LSUM_value[-3:, :] = self.LSUM_value[0, :]  # make all linesearch values the same as base case
        self.LSUM_value[-3:, :] -= np.array((1, 2, 3), ndmin=2).T  # make it a smaller and so further away from obs
        self.constraint[:] = self.studyJSON['constraint_target']  # make all constraints OK
        std = self.run_LineSearch()
        ref = self.run_LineSearch(ref=True)
        self.assertEqual(std['status'], "Stalled", msg='Status not as expected')
        self.assertEqual(std['NewParam'], None, msg='Expected None for NewParam')
        self.compare_LS_std_ref(std, ref)

    def test_LS_rand_std_ref(self):
        """ Do random tests of standard vs reference LineSearch code
        :return:None
        """
        # code below a modification of test_random_comp_ref
        trace = False  # set true to give trace..
        scale = np.ones((self.nparam, 1))
        scale[1:3] = np.reshape([1e-2, 1e-3], (2, 1))
        for samp in range(0, self.nsamp):
            self.setUp()  # make sure all values setup.
            self.param_value *= scale.T
            self.LSparam_value *= scale.T
            self.param_range *= scale
            self.step *= scale[:,0]
            M = np.diag(1. + np.arange(0.0, self.nobs) / float(self.nobs))
            M += np.random.uniform(-0.5, 0.5, (self.nobs, self.nobs))
            M = (M + M.T) / 2.
            M = M / np.linalg.det(M)  # make sure it has unit determinant.
            self.obs = M.dot(self.obs)  # transform obs
            self.cov = M.dot(self.cov).dot(M.T)  # and covariances
            self.LScov_iv = M.dot(self.LScov_iv).dot(M.T)  # and iv cov
            # compute and transform simulated values
            UM_value = np.random.uniform(-1., 1, (self.nparam + 1 + self.nalpha, self.nobs)) + self.obs
            for i in range(0, self.nparam + self.nalpha + 1):
                UM_value[i, :] = UM_value[i, :].dot(M)
            self.UM_value = UM_value
            #self.scalings[0] = 10  # scale one of the obs
            ref = self.run_LineSearch(ref=True, trace=trace)
            std = self.run_LineSearch(trace=trace) 
            # THIS FAILS with scaling  because params go out of range.
            self.compare_LS_std_ref(std, ref)


    def test_gaussNewton(self):
        """
        Test gaussNewton
        :return:  None
        """
        # feed  it a function which converges to some known value. Need to set up optimise etc...
        fn = lambda x: (x ** 2) * 20 - 5 / np.reshape(np.arange(1, x.shape[-1] + 1),
                                                      (1, -1))  # .reshape((-1,1))  # function to be optimised
        # def fn(x):
        #     top = (x ** 2) * 20 - 5
        #     bottom = np.arange(1, x.shape[1] + 1).reshape(1,-1)
        #     result = top/bottom
        #     print("r.shape",result.shape,"x.shape",x.shape)
        #     return result

        nparam = 10
        # don't start with same values..
        startParam = np.hstack((np.repeat(1.0, nparam - nparam / 2), np.repeat(0.0, nparam / 2)))  # starting values
        tgt = np.repeat(0.5, nparam) * 21  # target we want
        paramStep = np.repeat(0.01, nparam)  # parameter perturbation
        paramRange = np.vstack((np.repeat(0, nparam), np.repeat(1, nparam))).T  # param range
        cov_iv = np.diag(np.repeat(1e-12, nparam))  # import to specify this to avoid premature truncation
        cov = np.diag(np.repeat(1e-12, nparam))  # import to specify this as converge when roughly sqrt(cov) from tgt.
        optimise = {}

        best, status, info = gaussNewton(fn, startParam, paramRange, paramStep, tgt, optimise,
                                         cov=cov, cov_iv=cov_iv, trace=False)
        self.assertEqual(status, 'Converged')
        nptest.assert_allclose(np.squeeze(fn(best)), tgt, atol=1e-3)  # reached the target

    def test_jacobian(self):
        """
        Test Jacobian
        :return: None
        """
        import scipy
        # compute the Jacobian by evaluating the function and computing its derivs.
        # at params = (1,1,1,1)
        param = self.orig_param[:]
        fn = self.optFunction
        dx = np.min(np.abs(param)) / 1e6
        expect = np.zeros((self.nparam, self.nobs))
        startObs = fn(param)
        # compute 2nd order jacobian. Note different from GNjacobian but for small delta that OK!
        for indx in range(0, self.nparam):
            deltaParam = np.copy(param[:])  # copy parameter
            deltaParam[indx] += dx  # and modify.
            deltaParam2 = np.copy(param[:])  # copy parameter
            deltaParam2[indx] -= dx  # and modify.
            expect[indx, :] = (fn(deltaParam) - fn(deltaParam2)) / (
                    2 * dx)  # 2nd order calc (NB different from Jacobian w

        jac = GNjacobian(self.optFunction, self.orig_param, np.repeat(1e-6, self.nparam))
        nptest.assert_allclose(jac, expect, err_msg='Jacobian not as expected', rtol=1e-5)
        # test selection.
        paramIndex = [0, 2]  # perturbing 0th and 3rd params
        jac = GNjacobian(self.optFunction, self.orig_param, np.repeat(1e-6, self.nparam), paramIndex=paramIndex)
        expect2 = expect[paramIndex, :]
        nptest.assert_allclose(jac, expect2, err_msg='Sampled Jacobian not as expected', rtol=1e-5)
        # pass startObs in.
        jac = GNjacobian(self.optFunction, self.orig_param, np.repeat(1e-6, self.nparam), startObs=startObs)
        nptest.assert_allclose(jac, expect, err_msg=' Jacobian with startObs not as expected', rtol=1e-5)
        # pass both startObs & sub-selection.
        jac = GNjacobian(self.optFunction, self.orig_param, np.repeat(1e-6, self.nparam),
                         paramIndex=paramIndex, startObs=startObs)
        nptest.assert_allclose(jac, expect2, err_msg=' Sampled Jacobian with startObs not as expected', rtol=1e-5)
        # test extra args and kw args
        jac = GNjacobian(self.optFunction, self.orig_param, np.repeat(1e-3, self.nparam), 2,
                         paramIndex=paramIndex, startObs=startObs, ensembleMember=105)
        nptest.assert_allclose(jac, expect2, err_msg=' Sampled Jacobian with startObs and extra args not as expected',
                               rtol=1e-3)
        # check scaling works.

        scales = np.array([1.0, 1, 1, 1])
        jac = GNjacobian(self.optFunction, self.orig_param, np.repeat(1e-6, self.nparam),
                         paramIndex=paramIndex, startObs=startObs)

    def test_runJacobian(self):
        """
        Test runJacobian
        :return:
        """
        import pandas as pd
        param = self.orig_param
        paramNames = ['p' + str(i) for i in range(0, len(param))]
        param = pd.Series(self.param_range[:, 0].T, index=paramNames)
        deltaP = pd.Series(self.step, index=paramNames)
        prange = pd.DataFrame(self.param_range.T, columns=paramNames, index=['minParam', 'maxParam'])
        jac = runJacobian(self.optFunction, param, deltaP, prange)

        # comparision is to run GNjacobian and then wrap it
        expect = GNjacobian(self.optFunction, param.values, self.step)
        nptest.assert_allclose(jac.values, expect, err_msg='runJac jac not as expected', rtol=1e-5)

        # run a 10 ensemble members.
        jac = runJacobian(self.optFunction, param, deltaP, prange, nEnsemble=100,
                          returnVar=True, randomScale=1e-2)

        nptest.assert_allclose(jac.Jacobian.values, expect, err_msg='runJac jac not as expected',
                               rtol=1e-2, atol=1e-1)
        self.assertEqual(np.sum(jac.Jacobian_var.values != 0.0), jac.Jacobian_var.size, 'runJac jac not as expected')
        # self.fail("Implement tests")
        # now do at the top end.

        param = pd.Series(self.param_range[:, 1].T, index=paramNames)  # max size.
        deltaP = pd.Series(self.step, index=paramNames)
        prange = pd.DataFrame(self.param_range.T, columns=paramNames, index=['minParam', 'maxParam'])
        jac = runJacobian(self.optFunction, param, deltaP, prange)

        # comparision is to run GNjacobian and then wrap it
        expect = GNjacobian(self.optFunction, param.values, -self.step)
        nptest.assert_allclose(jac.values, expect, err_msg='runJac jac not as expected at +ve bdnry', rtol=1e-5)



if __name__ == "__main__":
    #print("Running Test Cases")
    unittest.main()  ## actually run the test cases

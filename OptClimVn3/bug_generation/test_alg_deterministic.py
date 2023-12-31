# Test that an algorithm is deterministic. Algorithm for this case is dfols
# Pattern is that objective function caches evaluations. If parameters
import numpy as np
import typing
import logging
import dfols
import random


class optObj():
    def __init__(self, target: np.ndarray, param_range: np.ndarray):
        self.cache = dict()
        self.fns_to_eval = []
        self.trace = []
        self.target = target[:]
        self.param_range = param_range[:, :]

    def reset(self):
        """
        Reset the state for another set of evaluations
        Sets self.fns_to_eval & self.trace to empty list
        :return: nada
        """
        self.fns_to_eval = []
        self.prev_trace = self.trace[:] # copy pev version
        self.trace = []

    def key(self, parameters: np.ndarray) -> bytes:
        """
        Generate key from parameters.
        :param parameters -- a np array .
        :return: bytes representing the data
        """
        key=parameters.tobytes()[:]
        return key


    def opt_fn(self, params: np.ndarray) -> typing.Optional[np.ndarray]:
        """
        return results from cache or if not in cache return None
        :param params:  Parameters
        :return: None (not in cache) or values from "model" evaluation.
        """
        if not isinstance(params,np.ndarray):
            raise ValueError("params not np.ndarray")
        key = self.key(params)
        result = self.cache.get(key, None)  # have we got the key already?
        self.trace += [params] # and parameters to trace.
        if result is None:
            result = np.repeat(np.nan, len(self.target))
            self.fns_to_eval.append( params)
            cnt = len(self.cache)+len(self.fns_to_eval)
            if len(self.trace) != cnt:
                # got some difference. find Diff between current and prev trace.
                # work out diff to prev_trace which requires making a hashable rep of array.
                missing = set([a.tobytes() for a in self.prev_trace])-set([a.tobytes() for a in self.trace])
                missing = [np.frombuffer(d) for d in missing] # convert back to list of arrays
                # work out closest match and delta
                trace = np.array(self.trace)
                if missing:
                    logging.warning("Some prev_trace elements not in trace")
                for m in missing:
                    delta = np.abs(m-trace).sum(axis=1)
                    indx = delta.argmin() # find closest match
                    logging.warning(f"Closest match to prev_trace {m} is at {indx} = {self.trace[indx]} sum abs(delta)={delta[indx]}")

                message_str = f"Generating new eval after {len(self.trace)} < {cnt} "
                #raise ValueError(message_str)
                logging.warning(message_str)
                breakpoint()
        return result-self.target

    def sim_model(self, params):
        """
        Simulate model
        :param params: Parameters being evaluated
        :return: Simulated obs,
        """
        scale_params = self.param_range[0, :] - self.param_range[1, :]
        pscale = (params - self.param_range[1, :]) / scale_params
        pscale -= 0.5  # tgt is at params = 0.5
        result = []
        while len(result) < self.target.size:
            r = 100 * (pscale + pscale ** 2)
            result += r.tolist()
        result = np.array(result[0:len(self.target)])+self.target
        return result

    def eval_models(self):
        """
        Evaluate all the models that needed evaluation!
        Evaluates all in  self.fns_to_eval and adds the result to the self.cache,
        :return: Nada
        """
        print(f"Have {len(self.fns_to_eval)} models to evaluate.")
        for params in self.fns_to_eval:
            key = self.key(params)
            self.cache[key] = self.sim_model(params)
        print(f"Evaluated models")


prange = np.array([[1e-4, 1e-2, 1e-1, 1, 10, 100], [0, 0, 0, 0, 0, 0]])
opt_obj = optObj(target=np.array([10., 4, 12, 34, 120, 300, 500, 1e-3]),
                 param_range=prange)
userParams = {'logging.save_diagnostic_info': True,
              'logging.save_xk': True,
              "noise.additive_noise_level": 21.0,
              'general.check_objfun_for_overflow': False,
              'init.run_in_parallel': True,
              "init.random_initial_directions": True,
              'interpolation.throw_error_on_nans': True,  # make an error happen!
              }
while True: # go until loop done
    # Every time through the loop dfols.solve is run. If a linear algebra error, then any cases that
    # need evaluation are evaluated and the whole algorithm starts again.
    start_params = prange[0, :][:] # initial params are max values
    opt_obj.reset() # reset for fresh run.
    random.seed(123456)  # make sure rng as used by DFOLS takes same values every time it is run.
    try:
        solution = dfols.solve(opt_obj.opt_fn, start_params,
                   objfun_has_noise=True,
                   bounds=prange, scaling_within_bounds=True
                   , maxfun=100, rhobeg=1e-1, rhoend=1e-3
                   , print_progress=False, user_params=userParams)
        # got here so we have sucesfully run dfols
        if solution.flag not in (solution.EXIT_SUCCESS, solution.EXIT_MAXFUN_WARNING):
            print("dfols failed with flag %i error : %s" % (solution.flag, solution.msg))
            raise Exception("Problem with dfols")
        print("All done")
        break # exit the loop

    except np.linalg.linalg.LinAlgError: # catch linear algebra errors.
        # do the model evaluations
        opt_obj.eval_models()

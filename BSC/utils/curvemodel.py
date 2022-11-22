import numpy as np
import emcee
import inspect
import traceback
import lmfit
import logging
from scipy.stats import norm, kde
from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls

from .curvefunctions import all_models

def recency_weights(num):
    if num == 1:
        return np.ones(1)
    else:
        recency_weights = [10**(1./num)] * num
        recency_weights = recency_weights**(np.arange(0, num))
        return recency_weights

def masked_mean_x_greater_than(posterior_distribution, y):
    """
        P(E[f(x)] > E[y] | Data)
    """
    predictions = np.ma.masked_invalid(posterior_distribution)
    return np.sum(predictions > y) / float(np.sum(~predictions.mask))


class CurveModel(object):

    def __init__(self,
                 function,
                 function_der=None,
                 min_vals={},
                 max_vals={},
                 default_vals={}):
        """
            function: the function to be fit
            function_der: derivative of that function
        """
        self.function = function
        if function_der != None:
            raise NotImplementedError("function derivate is not implemented yet...sorry!")
        self.function_der = function_der
        assert isinstance(min_vals, dict)
        self.min_vals = min_vals.copy()
        assert isinstance(max_vals, dict)
        self.max_vals = max_vals.copy()
        function_args = inspect.getargspec(function).args
        assert "x" in function_args, "The function needs 'x' as a parameter."
        for default_param_name in default_vals.keys():
            if default_param_name == "sigma":
                continue
            msg = "function %s doesn't take default param %s" % (function.__name__, default_param_name)
            assert default_param_name in function_args, msg
        self.function_params = [param for param in function_args if param != 'x']
        #set default values:
        self.default_vals = default_vals.copy()
        for param_name in self.function_params:
            if param_name not in default_vals:
                print("setting function parameter %s to default of 1.0 for function %s" % (param_name,
                                                                                           self.function.__name__))
                self.default_vals[param_name] = 1.0
        self.all_param_names = [param for param in self.function_params]
        self.all_param_names.append("sigma")
        self.name = self.function.__name__
        self.ndim = len(self.all_param_names)
        
        #uniform noise prior over interval:
        if "sigma" not in self.min_vals:
            self.min_vals["sigma"] = 0.
        if "sigma" not in self.max_vals:
            self.max_vals["sigma"] = 1.0
        if "sigma" not in self.default_vals:
            self.default_vals["sigma"] = 0.05
    
    def default_function_param_array(self):
        return np.asarray([self.default_vals[param_name] for param_name in self.function_params])

    def are_params_in_bounds(self, theta):
        """
            Are the parameters in their respective bounds?
        """
        in_bounds = True
        
        for param_name, param_value in zip(self.all_param_names, theta):
            if param_name in self.min_vals:
                if param_value < self.min_vals[param_name]:
                    in_bounds = False
            if param_name in self.max_vals:
                if param_value > self.max_vals[param_name]:
                    in_bounds = False
        return in_bounds

    def split_theta(self, theta):
        """Split theta into the function parameters (dict) and sigma. """
        params = {}
        sigma = None
        for param_name, param_value in zip(self.all_param_names, theta):
            if param_name in self.function_params:
                params[param_name] = param_value
            elif param_name == "sigma":
                sigma = param_value
        return params, sigma

    def split_theta_to_array(self, theta):
        """Split theta into the function parameters (array) and sigma. """
        params = theta[:-1]
        sigma = theta[-1]
        return params, sigma

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def predict_given_theta(self, x, theta):
        """
            Make predictions given a single theta
        """
        params, sigma = self.split_theta(theta)
        predictive_mu = self.function(x, **params)
        return predictive_mu, sigma

    def likelihood(self, x, y):
        """
            for each y_i in y:
                p(y_i|x, model)
        """
        params, sigma = self.split_theta(self.ml_params)
        return norm.pdf(y-self.function(x, **params), loc=0, scale=sigma)


class MLCurveModel(CurveModel):
    """
        ML fit of a curve.
    """

    def __init__(self, recency_weighting=True,  **kwargs):
        super(MLCurveModel, self).__init__(**kwargs)

        #Maximum Likelihood values of the parameters
        self.ml_params = None
        self.recency_weighting = recency_weighting

    def fit(self, x, y, weights=None, start_from_default=True):
        """
            weights: None or weight for each sample.
        """
        return self.fit_ml(x, y, weights, start_from_default)

    def predict(self, x):
        #assert len(x.shape) == 1
        params, sigma = self.split_theta_to_array(self.ml_params)
        return self.function(x, *params)
        #return np.asarray([self.function(x_pred, **params) for x_pred in x])

    def fit_ml(self, x, y, weights, start_from_default):
        """
            non-linear least-squares fit of the data.

            First tries Levenberg-Marquardt and falls back
            to BFGS in case that fails.

            Start from default values or from previous ml_params?
        """
        successful = self.fit_leastsq(x, y, weights, start_from_default)
        if not successful:
            successful = self.fit_bfgs(x, y, weights, start_from_default)
            if not successful:
                return False
        return successful

    def ml_sigma(self, x, y, popt, weights):
        """
            Given the ML parameters (popt) get the ML estimate of sigma.
        """
        if weights is None:
            if self.recency_weighting:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=recency_weights(len(y)))
                sigma = np.sqrt(variance)
            else:
                sigma = (y-self.function(x, *popt)).std()
        else:
            if self.recency_weighting:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=recency_weights(len(y)) * weights)
                sigma = np.sqrt(variance)
            else:
                variance = np.average((y-self.function(x, *popt))**2,
                    weights=weights)
                sigma = np.sqrt(variance)
        return sigma

    def fit_leastsq(self, x, y, weights, start_from_default):    
        try:
            if weights is None:
                if self.recency_weighting:
                    residuals = lambda p: np.sqrt(recency_weights(len(y))) * (self.function(x, *p) - y)
                else:
                    residuals = lambda p: self.function(x, *p) - y
            else:
                #the return value of this function will be squared, hence
                #we need to take the sqrt of the weights here
                if self.recency_weighting:
                    residuals = lambda p: np.sqrt(recency_weights(len(y))*weights) * (self.function(x, *p) - y)
                else:
                    residuals = lambda p: np.sqrt(weights) * (self.function(x, *p) - y)

            
            if start_from_default:
                initial_params = self.default_function_param_array()
            else:
                initial_params, _ = self.split_theta_to_array(self.ml_params)

            popt, cov_popt, info, msg, status = leastsq(residuals,
                    x0=initial_params,
                    full_output=True, maxfev=5000)
                #Dfun=,
                #col_deriv=True)
            
            if np.any(np.isnan(info["fjac"])):
                return False

            leastsq_success_statuses = [1,2,3,4]
            if status in leastsq_success_statuses:
                if any(np.isnan(popt)):
                    return False
                #within bounds?
                if not self.are_params_in_bounds(popt):
                    return False

                sigma = self.ml_sigma(x, y, popt, weights)
                self.ml_params = np.append(popt, [sigma])

                # logging.info("leastsq successful for model %s" % self.function.__name__)
                return True
            else:
                # logging.warn("leastsq NOT successful for model %s, msg: %s" % (self.function.__name__, msg))
                # logging.warn("best parameters found: " + str(popt))
                return False
        except Exception as e:
            print(e)
            tb = traceback.format_exc()
            print(tb)
            return False

    def fit_bfgs(self, x, y, weights, start_from_default):
        try:
            def objective(params):
                if weights is None:
                    if self.recency_weighting:
                        return np.sum(recency_weights(len(y))*(self.function(x, *params) - y)**2)
                    else:
                        return np.sum((self.function(x, *params) - y)**2)
                else:
                    if self.recency_weighting:
                        return np.sum(weights * recency_weights(len(y)) * (self.function(x, *params) - y)**2)
                    else:
                        return np.sum(weights * (self.function(x, *params) - y)**2)
            bounds = []
            for param_name in self.function_params:
                if param_name in self.min_vals and param_name in self.max_vals:
                    bounds.append((self.min_vals[param_name], self.max_vals[param_name]))
                elif param_name in self.min_vals:
                    bounds.append((self.min_vals[param_name], None))
                elif param_name in self.max_vals:
                    bounds.append((None, self.max_vals[param_name]))
                else:
                    bounds.append((None, None))

            if start_from_default:
                initial_params = self.default_function_param_array()
            else:
                initial_params, _ = self.split_theta_to_array(self.ml_params)

            popt, fval, info= fmin_l_bfgs_b(objective,
                                            x0=initial_params,
                                            bounds=bounds,
                                            approx_grad=True)
            if info["warnflag"] != 0:
                logging.warn("BFGS not converged! (warnflag %d) for model %s" % (info["warnflag"], self.name))
                logging.warn(info)
                return False

            if popt is None:
                return False
            if any(np.isnan(popt)):
                logging.info("bfgs NOT successful for model %s, parameter NaN" % self.name)
                return False
            sigma = self.ml_sigma(x, y, popt, weights)
            self.ml_params = np.append(popt, [sigma])
            # logging.info("bfgs successful for model %s" % self.name)
            return True
        except:
            return False

    def aic(self, x, y):
        """
            Akaike information criterion
            http://en.wikipedia.org/wiki/Akaike_information_criterion
        """
        params, sigma = self.split_theta_to_array(self.ml_params)
        y_model = self.function(x, *params)
        log_likelihood = norm.logpdf(y-y_model, loc=0, scale=sigma).sum()
        return 2 * len(self.function_params) - 2 * log_likelihood


class MCMCCurveModel(CurveModel):
    """
        MLE curve fitting + MCMC sampling with uniform priors for parameter uncertainty.

        Model: y ~ f(x) + eps with eps ~ N(0, sigma^2)
    """
    def __init__(self,
                 function,
                 function_der=None,
                 min_vals={},
                 max_vals={},
                 default_vals={},
                 burn_in=300,
                 nwalkers=100,
                 nsamples=800,
                 nthreads=1,
                 recency_weighting=False):
        """
            function: the function to be fit
            function_der: derivative of that function
        """
        super(MCMCCurveModel, self).__init__(
            function=function,
            function_der=function_der,
            min_vals=min_vals,
            max_vals=max_vals,
            default_vals=default_vals)
        self.ml_curve_model = MLCurveModel(
            function=function,
            function_der=function_der,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            default_vals=self.default_vals,
            recency_weighting=recency_weighting)

        #TODO: have two burn-ins, one for when the ML fitting is successful and one for when not!
        self.burn_in = burn_in
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.nthreads = nthreads
        self.recency_weighting = recency_weighting
    
    def fit(self, x, y):
        try:
            if self.ml_curve_model.fit(x, y):
                logging.info("ML fit: " + str(self.ml_curve_model.ml_params))
                self.fit_mcmc(x, y)
                return True
            else:
                return False
        except Exception as e:
            print(e)
            tb = traceback.format_exc()
            print(tb)
            return False

    #priors
    def ln_prior(self, theta):
        """
            log-prior is (up to a constant)
        """
        if self.are_params_in_bounds(theta):
            return 0.0
        else:
            return -np.inf
    
    #likelihood
    def ln_likelihood(self, theta, x, y):
        """
           y = y_true + y_noise
            with y_noise ~ N(0, sigma^2)
        """
        params, sigma = self.split_theta(theta)
        y_model = self.function(x, **params)
        if self.recency_weighting:
            weight = recency_weights(len(y))
            ln_likelihood = (weight*norm.logpdf(y-y_model, loc=0, scale=sigma)).sum()
        else:
            ln_likelihood = norm.logpdf(y-y_model, loc=0, scale=sigma).sum()
        if np.isnan(ln_likelihood):
            return -np.inf
        else:
            return ln_likelihood
        
    def ln_prob(self, theta, x, y):
        """
            posterior probability
        """
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, x, y)
    
    def fit_mcmc(self, x, y):
        #initialize in an area around the starting position
        #pos = [start + 1e-4*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        assert self.ml_curve_model.ml_params is not None
        pos = [self.ml_curve_model.ml_params + 1e-6*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        if self.nthreads <= 1:
            sampler = emcee.EnsembleSampler(self.nwalkers,
                self.ndim,
                self.ln_prob,
                args=(x, y))
        else:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                model_ln_prob,
                args=(self, x, y),
                threads=self.nthreads)
        sampler.run_mcmc(pos, self.nsamples)
        self.mcmc_chain = sampler.chain
        
    def get_burned_in_samples(self):
        samples = self.mcmc_chain[:, self.burn_in:, :].reshape((-1, self.ndim))
        return samples

    def predictive_distribution(self, x, thin=1):
        assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        predictions = []
        for theta in samples[::thin]:
            params, sigma = self.split_theta(theta)
            predictions.append(self.function(x, **params))
        return np.asarray(predictions)

    def predictive_ln_prob_distribution(self, x, y, thin=1):
        """
            posterior log p(y|x,D) for each sample
        """
        #assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        ln_probs = []
        for theta in samples[::thin]:
            ln_prob = self.ln_likelihood(theta, x, y)
            ln_probs.append(ln_prob)
        return np.asarray(ln_probs)

    def posterior_ln_prob(self, x, y, thin=10):
        """
            posterior log p(y|x,D)

            1/S sum p(y|D,theta_s)
            equivalent to:
            logsumexp(log p(y|D,theta_s)) - log(S)
        """
        assert not np.isscalar(x)
        assert not np.isscalar(y)
        x = np.asarray(x)
        y = np.asarray(y)
        ln_probs = self.predictive_ln_prob_distribution(x, y)
        #print ln_probs
        #print np.max(ln_probs)
        #print np.min(ln_probs)
        #print np.mean(ln_probs)
        #print "logsumexp(ln_probs)", logsumexp(ln_probs)
        #print "np.log(len(ln_probs)) ", np.log(len(ln_probs))
        #print logsumexp(ln_probs) - np.log(len(ln_probs))
        return logsumexp(ln_probs) - np.log(len(ln_probs))

    def predict(self, x):
        """
            E[f(x)]
        """
        predictions = self.predictive_distribution(x)
        return np.ma.masked_invalid(predictions).mean()
    
    def predictive_density(self, x_pos, x_density):
        density = kde.gaussian_kde(self.predictive_distribution(x_pos))
        return density(x_density)

    def prob_x_greater_than(self, x, y, theta):
        """
            P(f(x) > y | Data, theta)
        """
        params, sigma = self.split_theta(theta)
        mu = self.function(x, **params)
        cdf = norm.cdf(y, loc=mu, scale=sigma)
        return 1. - cdf

    def posterior_mean_prob_x_greater_than(self, x, y, thin=1):
        """
            P(E[f(x)] > E[y] | Data)

            thin: only use every thin'th sample
        
            Posterior probability that the expected valuef(x) is greater than 
            the expected value of y.
        """
        posterior_distribution = self.predictive_distribution(x, thin)
        return masked_mean_x_greater_than(posterior_distribution, y)


    def posterior_prob_x_greater_than(self, x, y, thin=1):
        """
            P(f(x) > y | Data)
        
            Posterior probability that f(x) is greater than y.
        """
        assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(y, float) or isinstance(y, int)
        probs = []
        samples = self.get_burned_in_samples()
        for theta in samples[::thin]:
            probs.append(self.prob_x_greater_than(x, y, theta))

        return np.ma.masked_invalid(probs).mean()

    def posterior_log_likelihoods(self, x, y):
        #DEPRECATED!
        samples = self.get_burned_in_samples()
        log_likelihoods = []
        for theta in samples:
            params, sigma = self.split_theta(theta)
            log_likelihood = self.ln_likelihood(theta, x, y)
            #TODO: rather add a -np.inf?
            if not np.isnan(log_likelihood) and np.isfinite(log_likelihood):
                log_likelihoods.append(log_likelihood)
        return log_likelihoods

    def mean_posterior_log_likelihood(self, x, y):
        #DEPRECATED!
        return np.ma.masked_invalid(self.posterior_log_likelihoods(x, y)).mean()

    def median_posterior_log_likelihood(self, x, y):
        #DEPRECATED!
        masked_x = np.ma.masked_invalid(self.posterior_log_likelihoods(x, y))
        return np.ma.extras.median(masked_x)

    def max_posterior_log_likelihood(self, x, y):
        #DEPRECATED!
        return np.ma.masked_invalid(self.posterior_log_likelihoods(x, y)).max()

    def posterior_log_likelihood(self, x, y):
        #DEPRECATED!
        return self.median_posterior_log_likelihood(x, y)

    def predictive_std(self, x, thin=1):
        """
           sqrt(Var[f(x)])
        """
        predictions = self.predictive_distribution(x, thin)
        return np.ma.masked_invalid(predictions).std()

    def dic(self, x, y):
        """ Deviance Information Criterion. """
        samples = self.get_burned_in_samples()
        deviances = []
        for theta in samples:
            params, sigma = self.split_theta(theta)
            deviance = -2 * self.ln_likelihood(theta, x, y)
            deviances.append(deviance)
        mean_theta = samples.mean(axis=0)
        theta_mean_deviance = -2 * self.ln_likelihood(mean_theta, x, y)
        DIC = 2 * np.mean(deviances) - theta_mean_deviance
        return DIC
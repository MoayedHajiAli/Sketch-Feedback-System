from __future__ import division, print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd import grad

import scipy.optimize


# The code in this example implements a method for finding a stationary point of
# the negative binomial likelihood via Newton's method, described here:
# https://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation


def newton(f, x0):
    # wrap scipy.optimize.newton with our automatic derivatives
    return scipy.optimize.newton(f, x0, fprime=grad(f), fprime2=grad(grad(f)))


def negbin_loglike(r, p, x):
    # the negative binomial log likelihood we want to maximize
    return gammaln(r+x) - gammaln(r) - gammaln(x+1) + x*np.log(p) + r*np.log(1-p)

class tst:
    def ts(x):
        return x * 2.0

def negbin_sample(r, p, size):
    # a negative binomial is a gamma-compound-Poisson
    return npr.poisson(npr.gamma(r, p/(1-p), size=size))


def fit_maxlike(x, r_guess):
    # follows Wikipedia's section on negative binomial max likelihood
    assert np.var(x) > np.mean(x), "Likelihood-maximizing parameters don't exist!"
    loglike = lambda r, p: np.sum(negbin_loglike(r, p, x))
    p = lambda r: np.sum(x) / np.sum(r+x)
    rprime = lambda r: grad(loglike)(r, p(r))
    r = newton(rprime, r_guess)
    return r, p(r)


if __name__ == "__main__":
    # generate data
    npr.seed(0)
    data = negbin_sample(r=5, p=0.5, size=1000)

    # fit likelihood-extremizing parameters
    r, p = fit_maxlike(data, r_guess=1)

    # report fit
    print('Fit parameters:')
    print('r={r}, p={p}'.format(r=r, p=p))

    print('Check that we are at a local stationary point:')
    loglike = lambda r, p: np.sum(negbin_loglike(r, p, data))
    grad_both = grad(loglike, argnum=(0, 1))
    print("grad", grad_both(r, p))
    q = tst()
    print("grad", grad(tst.ts)(0.0))

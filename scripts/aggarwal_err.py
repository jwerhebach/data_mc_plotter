#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import scipy.stats.distributions as sc_dist


def __reduce__(lower, upper, mu, alpha):
    cdf_lower, cdf_upper = sc_dist.poisson.cdf([lower, upper], mu)
    while True:
        pmf_lower, pmf_upper = sc_dist.poisson.pmf([lower, upper], mu)
        if pmf_lower > pmf_upper:
            cdf_upper -= pmf_upper
            temp_lower = lower
            temp_upper = upper - 1
        else:
            cdf_lower += cdf_lower
            temp_lower = lower + 1
            temp_upper = upper
        if (cdf_upper - cdf_lower + pmf_lower) > alpha:
            lower = temp_lower
            upper = temp_upper
        else:
            break
    return lower, upper


def calc_limits(mu, alpha=0.68268949, interval_type='central'):
    mu = np.asarray(mu)
    alpha = np.asarray(alpha)
    lower, upper = sc_dist.poisson.interval(alpha, mu)
    if interval_type == 'shortest':
        for i, mu_i in enumerate(mu):
            lower[i], upper[i] = __reduce__(lower[i], upper[i], mu_i, alpha)
    return lower, upper

if __name__ == '__main__':
    mu = [3. + 1./3., 7, 8]
    alpha = 0.999
    a = calc_limits(mu, alpha)
    print('central')
    print(a)
    print(upper)
    lower, upper = calc_limits(mu, alpha, interval_type='shortest')
    print('shortest')
    print(lower)
    print(upper)

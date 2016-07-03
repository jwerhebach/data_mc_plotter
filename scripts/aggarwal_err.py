#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import scipy.stats.distributions as sc_dist


def __shorten_single_limits_single_mu__(lower, upper, mu, alpha):
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


def calc_limits(mu, alphas=0.68268949):
    if isinstance(alphas, float):
        lim_shape = list(mu.shape) + [2]
        lim = np.zeros(lim_shape)
        lower = [slice(None) for _ in range(len(mu.shape))]
        lower.append(0)
        upper = [slice(None) for _ in range(len(mu.shape))]
        upper.append(1)
        lim[lower], lim[upper] = sc_dist.poisson.interval(alphas, mu)
        return lim
    else:
        lim_shape = list(mu.shape) + [len(alphas), 2]
        lim = np.zeros(lim_shape)
        for i, a in enumerate(alphas):
            lower = [slice(None) for _ in range(len(mu.shape))]
            lower.extend([i, 0])
            upper = [slice(None) for _ in range(len(mu.shape))]
            upper.extend([i, 1])
            lim[lower], lim[upper] = sc_dist.poisson.interval(a, mu)
        return lim


def calc_limits_different_mode(mu, alpha=0.68268949, interval_type='central'):
    mu = np.asarray(mu)
    alpha = np.asarray(alpha)
    lower, upper = sc_dist.poisson.interval(alpha, mu)
    if interval_type == 'shortest':
        for i, mu_i in enumerate(mu):
            lower[i], upper[i] = __shorten_single_limits_single_mu__(
                lower[i], upper[i], mu_i, alpha)
    return lower, upper


if __name__ == '__main__':
    mu = np.arange(27).reshape((3, 3, 3))
    alpha = 0.999
    a = calc_limits(mu, alpha)
    print(a.shape)
    alpha = []
    a = calc_limits(mu, alpha)
    print(a.shape)

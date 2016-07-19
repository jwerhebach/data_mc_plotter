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


def calc_limits(mu, alphas=0.68268949, rel=True):
    if isinstance(alphas, float):
        lim_shape = list(mu.shape) + [2]
        lim = np.zeros(lim_shape)
        lower = [slice(None) for _ in range(len(mu.shape))]
        lower.append(0)
        upper = [slice(None) for _ in range(len(mu.shape))]
        upper.append(1)
        lim[lower], lim[upper] = sc_dist.poisson.interval(alphas, mu)
        zero_mask = mu > 0.
        return lim, lim[zero_mask] / mu[zero_mask]

    else:
        lim_shape = list(mu.shape) + [len(alphas), 2]
        mu_shape = mu.shape
        lim_abs = np.zeros(lim_shape)
        lim_rel = np.zeros(lim_shape)
        for i, a in enumerate(alphas):
            lower = [slice(None) for _ in range(len(mu.shape))]
            lower.extend([i, 0])
            upper = [slice(None) for _ in range(len(mu.shape))]
            upper.extend([i, 1])
            zero_mask = mu > 0.
            flat_mu = mu.reshape(np.prod(mu_shape))
            flat_mask = zero_mask.reshape(np.prod(mu_shape))
            lim_lower, lim_upper = sc_dist.poisson.interval(a,
                                                            flat_mu[flat_mask])
            lim_lower_t_rel = np.zeros_like(flat_mask, dtype=float)
            lim_upper_t_rel = np.zeros_like(flat_mask, dtype=float)
            lim_lower_t_abs = np.zeros_like(flat_mask, dtype=float)
            lim_upper_t_abs = np.zeros_like(flat_mask, dtype=float)
            lim_lower_t_rel[flat_mask] = lim_lower / flat_mu[flat_mask]
            lim_upper_t_rel[flat_mask] = lim_upper / flat_mu[flat_mask]
            lim_lower_t_abs[flat_mask] = lim_lower
            lim_upper_t_abs[flat_mask] = lim_upper
            lim_abs[lower] = lim_lower_t_abs.reshape(mu_shape)
            lim_abs[upper] = lim_upper_t_abs.reshape(mu_shape)
            lim_rel[lower] = lim_lower_t_rel.reshape(mu_shape)
            lim_rel[upper] = lim_upper_t_rel.reshape(mu_shape)
        return lim_abs, lim_rel


def calc_p_alpha_bands(ref_hist, hist):
    a_ref = sc_dist.poisson.cdf(ref_hist, ref_hist)
    hist_lower = hist[:, :, 0]
    hist_upper = hist[:, :, 1]
    uncert = np.ones_like(hist)
    uncert_lower = uncert[:, :, 0]
    uncert_upper = uncert[:, :, 1]
    for i, [mu, a_mu] in enumerate(zip(ref_hist, a_ref)):
        if mu > 0:
            a_shape = uncert_lower[i].shape
            x_lower = hist_lower[i].reshape(np.prod(a_shape))
            x_upper = hist_upper[i].reshape(np.prod(a_shape))
            a_lower = sc_dist.poisson.cdf(x_lower, mu)
            a_upper = sc_dist.poisson.cdf(x_upper, mu)
            a_lower -= sc_dist.poisson.pmf(x_lower, mu)
            a_upper = (1-a_upper)
            a_lower = (a_lower)
            a_upper /= (1-a_mu)
            a_lower /= a_mu
            uncert_lower[i] = a_lower.reshape(a_shape)
            uncert_upper[i] = a_upper.reshape(a_shape)
    uncert[:, :, 0] = uncert_lower
    uncert[:, :, 1] =uncert_upper
    return uncert


def calc_p_alpha_single(ref_hist, hist):
    # Input expected to be [N_BINS]
    a_ref = sc_dist.poisson.cdf(ref_hist, ref_hist)
    uncert = np.empty_like(hist)
    for i, [mu, a_mu, x] in enumerate(zip(ref_hist, a_ref, hist)):
        a = sc_dist.poisson.cdf(x, mu)
        if x == 0 and mu == 0:
            uncert[i] = np.NaN
        elif mu == 0:
            uncert[i] = np.inf
        elif x == 0:
            uncert[i] = -np.inf
        elif x > mu:
            uncert[i] = (1-a)/(1-a_mu)
        else:
            a_0 = sc_dist.poisson.pmf(x, mu)
            uncert[i] = (a-a_0)/(-1*a_mu)

    return uncert


def calc_p_alpha_bands_nobs(ref_hist, hist):
    # Input expected to be [N_BINS, N_ALPHAS, 2]
    a = np.empty_like(hist)
    for i in range(hist.shape[0]):
        a[i] = calc_p_alpha_bands(ref_hist[i], hist[i])
    return a

def calc_p_alphas_nobs(ref_hist, hist):
    # Input expected to be [N_BINS]
    a = np.empty_like(hist)
    for i in range(hist.shape[0]):
        a[i] = calc_p_alpha_single(ref_hist[i], hist[i])
    return a


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

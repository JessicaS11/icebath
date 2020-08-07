import numpy as np
import scipy.stats as stats

def H_fr_frbd(freeboard, rho_sw, rho_i):
    """
    Compute iceberg total height based on freeboard and seawater and ice density
    """
    berg_H = freeboard * (rho_sw/(rho_sw-rho_i))
    print(rho_sw)
    print(rho_i)
    return berg_H

def draft_fr_H(height, freeboard):
    """
    Compute iceberg draft based on iceberg total height and freeboard
    """
    draft = height - freeboard
    return draft

def apply_decrease_offset(draft, offset):
    """
    Apply a bias offset that will decrease the draft
    """
    corr_draft = draft - offset
    return corr_draft

def filter_vals(input, num_mad=3):
    """
    Filter the input values by the num_mad deviations
    """
    med = np.nanmedian(input)
    mad = stats.median_absolute_deviation(input, axis=None, nan_policy='omit')

    nan = np.isnan(input)
    input[nan] = -9999
    input[input>(med+num_mad*mad)] = -9999
    input[input<(med-num_mad*mad)] = -9999
    test_equiv_bool = np.isclose(input, -9999)
    input[test_equiv_bool] = np.nan

    return input



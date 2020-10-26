import datetime as dt
import holoviews as hv
import numpy as np
import scipy.stats as stats

from icebath.core import fjord_props
from pyTMD.compute_tide_corrections import compute_tide_corrections

def H_fr_frbd(freeboard, rho_sw, rho_i):
    """
    Compute iceberg total height based on freeboard and seawater and ice density
    """
    berg_H = freeboard * (rho_sw/(rho_sw-rho_i))
    return berg_H

def draft_fr_H(height, freeboard):
    """
    Compute iceberg draft based on iceberg total height and freeboard
    """
    draft = height - freeboard
    return draft

def draft_uncert_dens(rho_sw, rho_sw_err, rho_i, rho_i_err):
    """
    Uncertainty on iceberg draft derived from height as a result of ice and seawater
    density uncertainty
    (random error)
    """
    rho_i_partial = rho_sw/((rho_sw-rho_i)**2)
    rho_sw_partial = -rho_i/((rho_sw-rho_i)**2)
    rho_conversion = rho_sw/(rho_sw-rho_i)
    rho_conversion_err = ((rho_i_partial*rho_i_err)**2+(rho_sw_partial*rho_sw_err)**2)**(0.5)

    return rho_conversion, rho_conversion_err

def apply_decrease_offset(draft, offset):
    """
    Apply a bias offset that will decrease the draft
    """
    corr_draft = draft - offset
    return corr_draft

def filter_vals(input, num_mad=3):
    """
    Filter the input values by the num_mad deviations from the median
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

def predict_tides(loc=None, img_time=None, model_path=None,
                    model=None, epsg=None, plot=False):
    
        assert loc!=None, "You must enter a location!"
        
        loc = fjord_props.get_mouth_coords(loc)
        # st_time = img_time - dt.timedelta(hours=-12)
        st_time = img_time - np.timedelta64(-12,'h')
        
        # time series for +/- 12 hours of image time, or over 24 hours, every 5 minutes
        ts = np.arange(0, 24*60*60, 5*60)
        xs = np.ones(len(ts))*loc[0]
        ys = np.ones(len(ts))*loc[1]

        # convert from numpy datetime64 to dt.datetime for pyTMD
        unix_epoch = np.datetime64(0, 's')
        one_second = np.timedelta64(1, 's')
        seconds_since_epoch = (st_time - unix_epoch) / one_second
        epoch = dt.datetime.utcfromtimestamp(seconds_since_epoch[0]).timetuple()[0:6]

        tide_pred = compute_tide_corrections(xs,ys,ts,
            DIRECTORY=model_path, MODEL=model,
            EPOCH=epoch, TYPE='drift', TIME='utc', EPSG=epsg)

        # could add a test here that tide_pred.mask is all false to make sure didn't get any land pixels

        tidal_ht=tide_pred.data
        if plot==True:
            tidx = list(ts).index(dt.timedelta(hours=12).seconds)
            plot_tides = np.array([[ts[i], tidal_ht[i]] for i in range(0,len(ts))])
            tides = hv.Scatter(plot_tides, kdims='time',vdims='tidal height')
            imgtime = hv.Scatter((ts[tidx], tidal_ht[tidx]))
            return ts, tidal_ht, tides*imgtime
        
        else:
            return ts, tidal_ht, None



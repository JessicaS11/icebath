import datetime as dt
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
import warnings

# from icebath.core import fjord_props
from icebath.core import fl_ice_calcs as icalcs


def xarray_to_gdf(xr):
    """
    Takes an xarray DataSet and generates a geodataframe of icebergs from the DEMs
    """
    # print('marker1')
    berg_gdf = gpd.GeoDataFrame(data=None)
    # print(xr['dtime'])
    for num in range(0, len(xr['dtime'])):
        # print('marker2')
        temp_berg_df = gdf_of_bergs(xr.isel({'dtime':num}))
        # print(xr['dtime'].isel({'dtime':num}))
        berg_gdf = berg_gdf.append(temp_berg_df, ignore_index=True)

    berg_gdf.crs = xr.attrs['crs']

    return berg_gdf


def gdf_of_bergs(onedem):
    """
    Takes an xarray dataarray for one time period and returns the needed geodataframe of icebergs
    """
    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    
    bergs = onedem['berg_outlines'].item()
    values=np.empty_like(bergs, dtype=object)
    sl_adj=np.zeros(len(bergs))
    berg_poly=np.empty_like(bergs, dtype=object)
    # print(onedem['dtime'])

    i=0
    for i in range(0, len(bergs)):
        #store as a polygon to turn into a geopandas geometry and extract only exterior coordinates (i.e. no holes)
        berg_poly[i] = Polygon(bergs[i])#exterior.coords
        # print(berg_poly[i])
        # print(type(berg_poly[i]))
        # print(i)
        #get the elevation values for the pixels within the iceberg
        # bounds: (minx, miny, maxx, maxy)
        bound_box = berg_poly[i].bounds
        # print(bound_box)
        
        vals = onedem['elevation'].sel(x=slice(bound_box[0], bound_box[2]),
                                        y=slice(bound_box[1], bound_box[3]))#.values.flatten()
        # print(vals)
        vals=vals.values.flatten()

        if np.any(vals > 500):
            print('iceberg too tall. Removing...')
            continue
        # remove nans because causing all kinds of issues down the processing pipeline (returning nan as a result and converting entire array to nan)
        vals = vals[~np.isnan(vals)]
        # print(vals)
        # print(onedem.attrs['berg_threshold'])
        values[i] = vals[vals>=onedem.attrs['berg_threshold']]
        # print(values[i])
        # print('check1')
        #get the regional elevation values and determine the sea level adjustment
        #make this border/boundary larger than one pixel (and do it by number of pixels!)
        bvals = onedem['elevation'].sel(x=slice(bound_box[0]-100, bound_box[2]+100),
                                        y=slice(bound_box[1]-100, bound_box[3]+100)).values.flatten()
        # print(bvals)
        bvals=bvals[~np.isnan(bvals)]
        sea = bvals[bvals<onedem.attrs['berg_threshold']]
        # print(bvals)
        # print(bvals[0])
        # print(type(bvals[0]))
        # print(np.isnan(bvals[0]))
        # print(sea)
        sl_adj[i] = np.nanmedian(sea)
        # print(sl_adj[i])
        #add a check here to make sure the sea level adjustment is reasonable
        # print('check for a reasonable sea level adjustment')
        
        #apply the sea level adjustment to the elevation values
        # if np.isnan(sl_adj[i]):
        #     sladj=0
        # else:
        #     sladj=sl_adj[i]
        # values[i] = values[i] - sladj

        values[i] = icalcs.apply_decrease_offset(values[i], sl_adj[i])

    temp_berg_df = gpd.GeoDataFrame({"DEMarray":values, 'sl_adjust':sl_adj, 'berg_poly':berg_poly}, geometry='berg_poly')

    fjord='JI'
    # add values that are same for all icebergs in DEM
    for name,value in (zip(['fjord', 'date', 'tidal_ht_offset', 'tidal_ht_min', 'tidal_ht_max'],
                        [fjord, onedem['dtime'].values, onedem['tidal_corr'].item(), onedem['min_tidal_ht'].item(), onedem['max_tidal_ht'].item()])):
        temp_berg_df[name] = value

    return temp_berg_df
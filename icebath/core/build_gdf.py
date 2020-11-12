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
    berg_gdf = gpd.GeoDataFrame(data=None)

    for num in range(0, len(xr['dtime'])):
        temp_berg_df = gdf_of_bergs(xr.isel({'dtime':num}))
        berg_gdf = berg_gdf.append(temp_berg_df, ignore_index=True)

    berg_gdf.crs = xr.attrs['crs']

    return berg_gdf


def gdf_of_bergs(onedem):
    """
    Takes an xarray dataarray for one time period and returns the needed geodataframe of icebergs
    """
    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    
    bergs = onedem['berg_outlines'].item()
    elevs=np.empty_like(bergs, dtype=object)
    sl_adj=np.zeros(len(bergs))
    berg_poly=np.empty_like(bergs, dtype=object)


    i=0
    rem_idx=[]
    for i in range(0, len(bergs)):
        #store as a polygon to turn into a geopandas geometry and extract only exterior coordinates (i.e. no holes)
        berg_poly[i] = Polygon(bergs[i])#exterior.coords
        #get the elevation values for the pixels within the iceberg
        # bounds: (minx, miny, maxx, maxy)
        bound_box = berg_poly[i].bounds
        # print(bound_box)
        
        vals = onedem['elevation'].sel(x=slice(bound_box[0], bound_box[2]),
                                        y=slice(bound_box[1], bound_box[3]))
        vals=vals.values.flatten()

        if np.any(vals > 500):
            print('"iceberg" too tall. Removing...')
            rem_idx.append(i)
            continue
        # remove nans because causing all kinds of issues down the processing pipeline (returning nan as a result and converting entire array to nan)
        vals = vals[~np.isnan(vals)]
        
        elevs[i] = vals[vals>=onedem.attrs['berg_threshold']]
        # get the regional elevation values and determine the sea level adjustment
        #TODO: make this border/boundary larger than one pixel (and do it by number of pixels!)
        bvals = onedem['elevation'].sel(x=slice(bound_box[0]-100, bound_box[2]+100),
                                        y=slice(bound_box[1]-100, bound_box[3]+100)).values.flatten()
        # print(bvals)
        bvals=bvals[~np.isnan(bvals)]
        sea = bvals[bvals<onedem.attrs['berg_threshold']]
        # print(sea)
        sl_adj[i] = np.nanmedian(sea)
        # print(sl_adj[i])
        #add a check here to make sure the sea level adjustment is reasonable
        # print('check for a reasonable sea level adjustment')
        
        #apply the sea level adjustment to the elevation values
        elevs[i] = icalcs.apply_decrease_offset(elevs[i], sl_adj[i])


    # Note: this step is critical for removing the partial/empty rows. Otherwise the adding of tidal_ht_min
    # (this function) or append (function calling this one, gdf_of_bergs) segmentation faults
    elevs = np.delete(elevs, rem_idx)
    sl_adj = np.delete(sl_adj, rem_idx)
    berg_poly = np.delete(berg_poly, rem_idx)
    
    temp_berg_df = gpd.GeoDataFrame({"DEMarray":elevs, 'sl_adjust':sl_adj, 'berg_poly':berg_poly}, geometry='berg_poly')

    fjord='JI'
    # add values that are same for all icebergs in DEM
    names = ['fjord', 'date', 'tidal_ht_offset', 'tidal_ht_min', 'tidal_ht_max']
    col_val = [fjord, onedem['dtime'].values, onedem['tidal_corr'].item(), onedem['min_tidal_ht'].item(), onedem['max_tidal_ht'].item()]
    
    for name,val in (zip(names,col_val)):
        temp_berg_df[name] = val

    return temp_berg_df
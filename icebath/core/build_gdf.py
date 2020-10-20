import datetime as dt
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
import warnings

# from icebath.core import fjord_props

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
    bergs = onedem['berg_outlines'].item()
    values=np.empty_like(bergs)
    sl_adj=np.zeros(len(bergs))
    berg_poly=np.empty_like(bergs)

    i=0
    for i in range(0,len(bergs)):
        #store as a polygon to turn into a geopandas geometry
        berg_poly[i] = Polygon(bergs[i])
        # print(berg_poly[i])
        
        #get the elevation values for the pixels within the iceberg
        # bounds: (minx, miny, maxx, maxy)
        bound_box = berg_poly[i].bounds
        print(bound_box)
        
        vals = onedem['elevation'].sel(x=slice(bound_box[0], bound_box[2]),
                                        y=slice(bound_box[1], bound_box[3])).values.flatten()
        # print(vals)
        values[i] = vals[vals>=onedem.attrs['berg_threshold']]
        # print(values[i])
        #get the regional elevation values and determine the sea level adjustment
        #make this border/boundary larger than one pixel (and do it by number of pixels!)
        bvals = onedem['elevation'].sel(x=slice(bound_box[0]-100, bound_box[2]+100),
                                        y=slice(bound_box[1]-100, bound_box[3]+100)).values.flatten()
        sea = bvals[bvals<onedem.attrs['berg_threshold']]
        # print(bvals)
        sl_adj[i] = np.median(sea)
        print(sl_adj[i])
        #add a check here to make sure the sea level adjustment is reasonable
        print('check for a reasonable sea level adjustment')
        
        #apply the sea level adjustment to the elevation values
        values[i] = values[i] - sl_adj[i]

    temp_berg_df = gpd.GeoDataFrame({"DEMarray":values, 'sl_adjust':sl_adj, 'berg_poly':berg_poly}, geometry='berg_poly')

    fjord='JI'
    # add values that are same for all icebergs in DEM
    for name,value in (zip(['fjord', 'date', 'tidal_ht_offset', 'tidal_ht_min', 'tidal_ht_max'],
                        [fjord, onedem['dtime'].values, onedem['tidal_corr'].item(), onedem['min_tidal_ht'].item(), onedem['max_tidal_ht'].item()])):
        temp_berg_df[name] = value

    return temp_berg_df
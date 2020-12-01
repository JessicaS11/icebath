import datetime as dt
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
import warnings

# from icebath.core import fjord_props
from icebath.core import fl_ice_calcs as icalcs
from icebath.utils import raster_ops


def xarray_to_gdf(xr):
    """
    Takes an xarray DataSet and generates a geodataframe of icebergs from the DEMs
    """
    berg_gdf = gpd.GeoDataFrame(data=None)

    for num in range(0, len(xr['dtime'])):
        temp_berg_df = gdf_of_bergs(xr.isel({'dtime':num}))
        berg_gdf = berg_gdf.append(temp_berg_df, ignore_index=True)

    berg_gdf.crs = xr.attrs['crs']
    berg_gdf.sl_adjust.attrs['note'] = "sea level adjustment is relative to tidal height, not 0msl"
    berg_gdf.sl_adjust.attrs['units'] = "meters"

    return berg_gdf


def gdf_of_bergs(onedem):
    """
    Takes an xarray dataarray for one time period and returns the needed geodataframe of icebergs
    """
    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    
    # get potential icebergs from edges in raster image (or some other method - do it for the thresholding first since have a function for it)
    threshold=60
    bergs = raster_ops.poly_from_thresh(onedem.x, onedem.y, onedem.elevation, threshold)

    print("Got potential icebergs for an image")

    # remove icebergs that don't meet the requirements
    filt_bergs = []
    elevs = []
    sl_adjs = []

    for berg in bergs:
        # make a valid shapely Polygon of the berg vertices
        berg = Polygon(berg)
        if berg.is_valid == False:
            continue
        # remove holes
        if berg.interiors:
            berg = Polygon(list(berg.exterior.coords))
        
        # skip too-small bergs
        minarea = 100
        if berg.area < minarea:
            continue

        # get the raster pixel values for the iceberg
        # bounds: (minx, miny, maxx, maxy)
        bound_box = berg.bounds
        # print(bound_box)
        
        vals = onedem['elevation'].sel(x=slice(bound_box[0], bound_box[2]),
                                        y=slice(bound_box[1], bound_box[3]))
        vals=vals.values.flatten()

        # skip bergs that likely contain a lot of cloud (or otherwise unrealistic elevation) pixels
        if np.any(vals > 500):
            print('"iceberg" too tall. Removing...')
            continue
        # remove nans because causing all kinds of issues down the processing pipeline (returning nan as a result and converting entire array to nan)
        vals = vals[~np.isnan(vals)]
        
        # get the regional elevation values and determine the sea level adjustment
        #TODO: make this border/boundary larger than one pixel (and do it by number of pixels!)
        bvals = onedem['elevation'].sel(x=slice(bound_box[0]-100, bound_box[2]+100),
                                        y=slice(bound_box[1]-100, bound_box[3]+100)).values.flatten()
        bvals=bvals[~np.isnan(bvals)]
        sea = [val for val in bvals if val not in vals]
        # print(sea)
        # NOTE: sea level adjustment (m) is relative to tidal height at the time of image acquisition, not 0 msl
        #add a check here to make sure the sea level adjustment is reasonable
        sl_adj = np.nanmedian(sea)
        
        #apply the sea level adjustment to the elevation values
        vals = icalcs.apply_decrease_offset(vals, sl_adj)

        filt_bergs.append(berg)
        elevs.append(vals)
        sl_adjs.append(sl_adj)


    temp_berg_df = gpd.GeoDataFrame({"DEMarray":elevs, 'sl_adjust':sl_adjs, 'berg_poly':filt_bergs}, geometry='berg_poly')

    # TODO: generalize the fjord input
    fjord='JI'
    # add values that are same for all icebergs in DEM
    names = ['fjord', 'date', 'tidal_ht_offset', 'tidal_ht_min', 'tidal_ht_max']
    col_val = [fjord, onedem['dtime'].values, onedem['tidal_corr'].item(), onedem['min_tidal_ht'].item(), onedem['max_tidal_ht'].item()]
    
    for name,val in (zip(names,col_val)):
        temp_berg_df[name] = val

    print("Generated geodataframe of icebergs for this image")
    return temp_berg_df
   

    '''
    sea level adjustment uncertainty from matlab (likely need to implement here)
    sl_dz1 = icefree_area1(~isnan(icefree_area1)); sl_dz1(sl_dz1>nanmedian(sl_dz1)+2*2.9 | sl_dz1<nanmedian(sl_dz1)-2*2.9) = NaN; %I think this is removing 2 sigma outliers, but I'm not sure what the significance of the 2.9 is (sigma calculated when/how?)
    '''

    '''

    lines 756-763
    %calculate some basic info about the iceberg polygon/DEM data coverage
    %%non_nans = ~isnan(IB.zo.values.map);
    %%nans = isnan(IB.zo.values.map);
    IB.zo.cov.numpix = sum(sum(IB.berg_mask));
    IB.zo.cov.numpix_nan = sum(sum(IB.berg_mask.*isnan(IB.zo.values.map)));
    IB.zo.cov.area = IB.zo.cov.numpix * DEM1_pixel_area; %=base_area, below
    IB.zo.cov.area_nan = IB.zo.cov.numpix_nan * DEM1_pixel_area;
    IB.zo.cov.percent = IB.zo.cov.area_nan/IB.zo.cov.area*100;

    '''


    

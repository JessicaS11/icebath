import datetime as dt
import numpy as np
import os
import geopandas as gpd
import rasterio
import rioxarray
import scipy.stats as stats
from shapely.geometry import Polygon
import xarray as xr
import warnings

from icebath.core import fjord_props
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
    # threshold=60
    # bergs = raster_ops.poly_from_thresh(onedem.x, onedem.y, onedem.elevation, threshold)
    
    try:
        onedem.elevation.attrs['crs'] = onedem.attrs['crs']
    except KeyError:
        try:
            onedem.elevation.attrs['proj4'] = onedem.attrs['proj4']
        except KeyError:
            print("Your input DEM does not have a CRS attribute")

    trans=onedem.attrs['transform']
    flipax=[]
    if trans[0] < 0:
        flipax.append(1)
    if trans[4] < 0:
        flipax.append(0)

    # TODO: generalize the fjord input
    fjord='JI'
    max_freebd = fjord_props.get_ice_thickness(fjord)/10.0
    min_area = fjord_props.get_min_berg_area(fjord)



    res = onedem.attrs['res'][0] #Note: the pixel area will be inaccurate if the resolution is not the same in x and y
    # labeled_arr = raster_ops.labeled_from_edges(onedem.elevation.values, sigma=6, resolution=res, min_area=4000, flipax=flipax)
    labeled_arr = raster_ops.labeled_from_segmentation(onedem.elevation.values, [3,10], resolution=res, min_area=min_area, flipax=flipax)
    print("Got labeled raster of potential icebergs for an image")

    print(np.shape(labeled_arr))

    # create iceberg polygons, excluding icebergs that don't meet the requirements
    # Note: features.shapes returns a generator. However, if we try to iterate through it with a for loop, the StopIteration exception
    # is not passed up into the for loop and execution hangs when it hits the end of the for loop withouth completing the function
    poss_bergs = list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(labeled_arr, transform=trans))[:-1]
    
    print(len(poss_bergs))

    bergs = []
    elevs = []
    sl_adjs = []


    for berg in poss_bergs: # note: features.shapes returns a generator
        # make a valid shapely Polygon of the berg vertices
        berg = Polygon(berg)
        if berg.is_valid == False:
            continue
        # remove holes
        if berg.interiors:
            berg = Polygon(list(berg.exterior.coords))

        # too-small polygons are eliminated in the edge map labeling step (raster_ops.labeled_from_edges)
        # # skip too-small bergs
        # minarea = 100
        # if berg.area < minarea:
        #     continue

        # skip bergs that are too large to realistically be just one berg
        # these should now be removed during the iceberg list generation from segmentation step
        # if berg.area > 1000000:
        #     print('"iceberg" too large. Removing...')
        #     continue

        # get the subset (based on a buffered bounding box) of the DEM that contains the iceberg
        # bounds: (minx, miny, maxx, maxy)
        bound_box = berg.bounds
        # print(bound_box)
    
        # 10 pixel buffer
        buffer = 10 * res
        berg_dem = onedem['elevation'].sel(x=slice(bound_box[0]-buffer, bound_box[2]+buffer),
                                        y=slice(bound_box[1]-buffer, bound_box[3]+buffer))
                                        
        # extract the iceberg elevation values
        vals = berg_dem.rio.clip([berg], crs=onedem.attrs['crs']).values.flatten()

        # skip bergs that likely contain a lot of cloud (or otherwise unrealistic elevation) pixels
        if np.nanmedian(vals) > max_freebd:  # units in meters, matching those of the DEM elevation
            print('"iceberg" too tall. Removing...')
            continue

        # remove nans because causing all kinds of issues down the processing pipeline (returning nan as a result and converting entire array to nan)
        vals = vals[~np.isnan(vals)]

        # get the regional elevation values and use to determine the sea level adjustment
        bvals = berg_dem.values.flatten()
        bvals=bvals[~np.isnan(bvals)]

        sea = [val for val in bvals if val not in vals]
        # print(sea)
        # NOTE: sea level adjustment (m) is relative to tidal height at the time of image acquisition, not 0 msl
        # add a check here to make sure the sea level adjustment is reasonable
        sl_adj = np.nanmedian(sea)
        # print(sl_adj)
        
        # check that the median freeboard elevation (pre-filtering) is at least 5 m above sea level
        if abs(np.nanmedian(vals)-sl_adj) < 5:
            print('median iceberg freeboard less than 5 m')
            continue

        # print(np.nanmedian(vals)-sl_adj)
        # apply the sea level adjustment to the elevation values
        vals = icalcs.apply_decrease_offset(vals, sl_adj)

        bergs.append(berg)
        elevs.append(vals)
        sl_adjs.append(sl_adj)

    # delete generator object so no issues between DEMs
    try:
        del poss_bergs
    except NameError:
        pass
    
    print(len(bergs))

    temp_berg_df = gpd.GeoDataFrame({"DEMarray":elevs, 'sl_adjust':sl_adjs, 'berg_poly':bergs}, geometry='berg_poly')
    
    # filter out "icebergs" that have sea level adjustments outside the median +/- two times the median absolute deviation
    sl_adj_med = np.nanmedian(temp_berg_df.sl_adjust)
    sl_adj_mad = stats.median_abs_deviation(temp_berg_df.sl_adjust, nan_policy='omit')

    temp_berg_df = temp_berg_df[(temp_berg_df['sl_adjust'] > sl_adj_med - 2*sl_adj_mad) & 
                                (temp_berg_df['sl_adjust'] < sl_adj_med + 2*sl_adj_mad)]

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


    

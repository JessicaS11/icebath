import datetime as dt
import dask
import dask.array as da
import numpy as np
import os
import geopandas as gpd
import pandas as pd
import rasterio
import rioxarray
import scipy.stats as stats
from shapely.geometry import Polygon
import xarray as xr
import warnings

from icebath.core import fjord_props
from icebath.core import fl_ice_calcs as icalcs
from icebath.utils import raster_ops
from icebath.utils import vector_ops


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


# Code Improvement: take the actual steps of computations out of this function and just combine the steps here
# (e.g. functionize the array manipulations, calling either a dask or numpy version)
def gdf_of_bergs(onedem, usedask=True):
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
    
    fjord = onedem.attrs['fjord']
    min_area = fjord_props.get_min_berg_area(fjord)

    res = onedem.attrs['res'][0] #Note: the pixel area will be inaccurate if the resolution is not the same in x and y

    if usedask==True:
        # Daskify the iceberg segmentation process. Note that dask-image has some functionality to operate
        # directly on dask arrays (e.g. dask_image.ndfilters.sobel), which would need to be put into utils.raster.py
        # https://dask-image.readthedocs.io/en/latest/dask_image.ndfilters.html
        # However, as of yet there doesn't appear to be a way to easily implement the watershed segmentation, other than in chunks
        
        # see else statement with non-dask version for descriptions of what each step is doing
        def seg_wrapper(tiles):
            return raster_ops.labeled_from_segmentation(tiles, [3,10], resolution=res, min_area=min_area, flipax=[])
        def filter_wrapper(tiles, elevs):
            return raster_ops.border_filtering(tiles, elevs, flipax=[])

        elev_copy = onedem.elevation.data # should return a dask array
        for ax in flipax:
            elev_copy = da.flip(elev_copy, axis=ax)
        print(type(elev_copy))
        elev_overlap = da.overlap.overlap(elev_copy, depth=10, boundary='nearest')
        seglabeled_overlap = da.map_overlap(seg_wrapper, elev_overlap, trim=False) # including depth=10 here will ADD another overlap
        print("Got labeled raster of potential icebergs for an image")
        labeled_overlap = da.map_overlap(filter_wrapper, seglabeled_overlap, elev_overlap, trim=False, dtype='int32')
        labeled_arr = da.overlap.trim_overlap(labeled_overlap, depth=10)
        
        print(da.min(labeled_arr).compute())
        print(da.max(labeled_arr).compute())
        
        print("about to get the list of possible bergs")
        
        # I think that by using concatenate=True, it might not actually be using multiple dask works for the computation
        # however, the below dask approach bring in geospatial issues
        poss_bergs_list = []
        def get_bergs(labeled_blocks):
            block_bergs = list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(
                                labeled_blocks.astype('int32'), transform=trans))[:-1]
            poss_bergs_list.append(block_bergs)
        
        da.blockwise(get_bergs, '', labeled_arr, 'ij', 
                        meta=pd.DataFrame({'c':[]}), concatenate=True).compute()
        # print(poss_bergs_list[0])
        # print(type(poss_bergs_list))
        poss_bergs_gdf = gpd.GeoDataFrame({'geometry':[Polygon(poly) for poly in poss_bergs_list[0]]})
        
        '''
        This approach, as recommended in response to my SO post, worked great, excepting the geospatial part
        Specifically, because the transform is for the entire xarray extent, when applied to each chunk by dask
        it results in incorrect polygon coordinates, so the results have the right shapes but in the wrong places
        (and wrong relative locations/orientations).
        URL: https://stackoverflow.com/questions/66232232/produce-vector-output-from-a-dask-array/66245347?noredirect=1#comment117119583_66245347

        @dask.delayed
        def get_bergs(labeled_blocks):
            return list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(
                                labeled_blocks.astype('int32'), transform=trans))[:-1]

        poss_bergs_list = dask.compute([get_bergs(bl) for bl in labeled_arr.to_delayed().ravel()])[0]
        print(poss_bergs_list)

        # unnest the list of polygons returned by using dask to polygonize
        concat_list = [item for sublist in poss_bergs_list for item in sublist if len(item)!=0]
        print(concat_list)
        
        poss_bergs_gdf = gpd.GeoDataFrame({'geometry':[Polygon(poly) for poly in concat_list]})
        
        '''
        # convert to a geodataframe, combine geometries (in case any bergs were on chunk borders), and generate new polygon list
        # print(poss_bergs_gdf)
        poss_berg_combined = gpd.overlay(poss_bergs_gdf, poss_bergs_gdf, how='union')
        # print(poss_berg_combined)
        print(poss_berg_combined.geometry.plot())
        poss_bergs = [berg for berg in poss_berg_combined.geometry]
        # print(poss_bergs)
        print(len(poss_bergs))

        try:
            del elev_copy
            del elev_overlap
            del seglabeled_overlap
            del labeled_overlap
            del labeled_arr
            del poss_bergs_list
            # del concat_list
            del poss_berg_combined
        except NameError:
            pass
        
    else:
        # create copy of elevation values so original dataset values are not impacted by image manipulations
        # and positive/negative coordinate systems can be ignored (note flipax=[] below)
        # something wonky is happening and when I ran this code on Pangeo I needed to NOT flip the elevation values here and then switch the bounding box y value order
        # Not entirely sure what's going on, but need to be aware of this!!
        print("NOTE: check for proper orientation of results depending on compute environment. Pangeo results were upside down.")
        # elev_copy = np.copy(onedem.elevation.values)
        elev_copy = np.copy(np.flip(onedem.elevation.values, axis=flipax))
        # flipax=[]
        
        # generate a labeled array of potential iceberg features, excluding those that are too large or small
        seglabeled_arr = raster_ops.labeled_from_segmentation(elev_copy, [3,10], resolution=res, min_area=min_area, flipax=[])
        print("Got labeled raster of potential icebergs for an image")
        # remove features whose borders are >50% no data values (i.e. the "iceberg" edge is really a DEM edge)
        labeled_arr = raster_ops.border_filtering(seglabeled_arr, elev_copy, flipax=[]).astype(seglabeled_arr.dtype)
        # apparently rasterio can't handle int64 inputs, which is what border_filtering returns   

        # create iceberg polygons 
        poss_bergs = list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(labeled_arr, transform=trans))[:-1]
  
        try:
            del elev_copy
            del seglabeled_arr
            del labeled_arr
        
        except NameError:
            pass
   
    # print(type(poss_bergs))
    # print(len(poss_bergs))

    # Note: features.shapes returns a generator. However, if we try to iterate through it with a for loop, the StopIteration exception
    # is not passed up into the for loop and execution hangs when it hits the end of the for loop without completing the function
    # Exclude icebergs that don't meet the requirments and put the rest into a generator (below)
    bergs, elevs, sl_adjs = get_good_bergs(poss_bergs, onedem)

    # delete generator object so no issues between DEMs
    # delete arrays in memory (should be done automatically by garbage collector)
    try:
        del poss_bergs        
    except NameError:
        pass

    temp_berg_df = gpd.GeoDataFrame({"DEMarray":elevs, 'sl_adjust':sl_adjs, 'berg_poly':bergs}, geometry='berg_poly')
    
    # look at DEM-wide sea level adjustments
    # filter out "icebergs" that have sea level adjustments outside the median +/- two times the median absolute deviation
    # sl_adj_med = np.nanmedian(temp_berg_df.sl_adjust)
    # sl_adj_mad = stats.median_abs_deviation(temp_berg_df.sl_adjust, nan_policy='omit')

    # print(sl_adj_med)
    # print(sl_adj_mad)
    # temp_berg_df = temp_berg_df[(temp_berg_df['sl_adjust'] > sl_adj_med - 2*sl_adj_mad) & 
    #                             (temp_berg_df['sl_adjust'] < sl_adj_med + 2*sl_adj_mad)]
    # print(len(temp_berg_df))
    # Description of above filter: For a given DEM, an individual iceberg's sea level adjustment 
    # (***more on that later***) must fall within the median +/- 2 mean absolute deviations of 
    # the sea level adjustments across that entire DEM. Otherwise the candidate iceberg is excluded 
    # because it is likely subject to DEM generation errors or not correctly adjusted due to a 
    # lack of nearby open water pixels in the DEM.
    
    # add values that are same for all icebergs in DEM
    names = ['fjord', 'date', 'tidal_ht_offset', 'tidal_ht_min', 'tidal_ht_max']
    col_val = [fjord, onedem['dtime'].values, onedem['tidal_corr'].item(), onedem['min_tidal_ht'].item(), onedem['max_tidal_ht'].item()]
    
    for name,val in (zip(names,col_val)):
        temp_berg_df[name] = val

    print("Generated geodataframe of icebergs for this image")

    return temp_berg_df


def get_good_bergs(poss_bergs, onedem):
    """
    Test each potential iceberg for validity, and if valid compute the sea level adjustment and
    get elevation pixel values for putting into the geodataframe.

    Parameter
    ---------
    poss_bergs : iterator of potential iceberg geometries
    """

    bergs = []
    elevs = []
    sl_adjs = []

    fjord = onedem.attrs['fjord']
    max_freebd = fjord_props.get_ice_thickness(fjord)/10.0
    res = onedem.attrs['res'][0] #Note: the pixel area will be inaccurate if the resolution is not the same in x and y

    # 10 pixel buffer
    buffer = 10 * res

    for berg in poss_bergs:
        # make a valid shapely Polygon of the berg vertices
        # print(berg)
        origberg = Polygon(berg)

        if origberg.is_valid == False or origberg.is_empty == True:
            print("invalid or empty berg geometry")
            continue

        # create a negatively buffered berg outline to exclude border/water pixels
        berg = origberg.buffer(-buffer)
        if berg.is_valid == False or berg.is_empty == True:
            print("invalid buffered inner-berg geometry")
            continue

        # get the largest polygon from a multipolygon (if one was created during buffering)
        if berg.geom_type == 'MultiPolygon':
            subbergs = list(berg)
            area = []
            for sb in subbergs:
                sb = Polygon(sb)
                area.append(sb.area)
            idx = np.where(area == np.nanmax(area))[0]
            berg = Polygon(subbergs[idx[0]])
            print('tried to trim down a multipolygon')
        
        if berg.is_valid == False:
            print("invalid buffered multipology extraction")
            continue

        # remove holes
        if berg.interiors:
            berg = Polygon(list(berg.exterior.coords))
            print('removed some holes')
        
        if berg.is_valid == False:
            print("invalid buffered interiors geometry")
            continue

        # get the polygon complexity and skip if it's above the threshold
        complexity = vector_ops.poly_complexity(berg)
        if complexity >= 0.07:
            print('border too complex. Removing...')
            continue

        # get the subset (based on a buffered bounding box) of the DEM that contains the iceberg
        # bounds: (minx, miny, maxx, maxy)
        bound_box = origberg.bounds
        berg_dem = onedem['elevation'].sel(x=slice(bound_box[0]-buffer, bound_box[2]+buffer),
                                        # y=slice(bound_box[3]+buffer, bound_box[1]-buffer)) # pangeo? May have been because of issues with applying transform to right-side-up image above?
                                        y=slice(bound_box[1]-buffer, bound_box[3]+buffer)) # my comp
        
        
        # print(bound_box)
        # print(berg_dem)
        # print(np.shape(berg_dem))
        # extract the iceberg elevation values
        # Note: rioxarray does not carry crs info from the dataset to individual variables
        # print(berg)
        # print(len(bergs))
        vals = berg_dem.rio.clip([berg], crs=onedem.attrs['crs']).values.flatten()
        # print(vals)

        # remove nans because causing all kinds of issues down the processing pipeline (returning nan as a result and converting entire array to nan)
        vals = vals[~np.isnan(vals)]
        # print(vals)

        # skip bergs that likely contain a lot of cloud (or otherwise unrealistic elevation) pixels
        if np.nanmedian(vals) > max_freebd:  # units in meters, matching those of the DEM elevation
            print('"iceberg" too tall. Removing...')
            continue

        # get the pixel values for the original berg extent and a buffered version for the sea level adjustment
        orig_vals = berg_dem.rio.clip([origberg], crs=onedem.attrs['crs']).values.flatten()       
        orig_vals = orig_vals[~np.isnan(orig_vals)]

        slberg = origberg.buffer(buffer) # get geometry bordering iceberg for sea level adjustment
        # get the regional elevation values and use to determine the sea level adjustment
        slvals = berg_dem.rio.clip([slberg], crs=onedem.attrs['crs']).values.flatten()
        slvals=slvals[~np.isnan(slvals)]

        sea = [val for val in slvals if val not in orig_vals]
        # NOTE: sea level adjustment (m) is relative to tidal height at the time of image acquisition, not 0 msl
        sl_adj = np.nanmedian(sea)
        # print(sl_adj)
        
        # check that the median freeboard elevation (pre-filtering) is at least 15 m above sea level
        if abs(np.nanmedian(vals)-sl_adj) < 15:
            # print(np.nanmedian(vals))
            # print(sl_adj)
            # print('median iceberg freeboard less than 15 m')
            continue

        # apply the sea level adjustment to the elevation values
        vals = icalcs.apply_decrease_offset(vals, sl_adj)        
        
        bergs.append(berg)
        elevs.append(vals)
        sl_adjs.append(sl_adj)
    
    print(len(bergs))

    return bergs, elevs, sl_adjs

    

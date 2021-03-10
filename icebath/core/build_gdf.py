import datetime as dt
import dask
import dask.array as da
import numpy as np
import os
import geopandas as gpd
import pandas as pd
import rasterio
import rioxarray
from rioxarray.exceptions import NoDataInBounds
import scipy.stats as stats
from shapely.geometry import Polygon
import xarray as xr
import warnings
import itertools as it

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


# DevGoal: This function could have improvements to its parallelization (especially in the later steps)
# and could certainly be refactored into a larger number of simpler functions
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

    # print("going to enter the rasterize function")
    poss_bergs = get_poss_bergs_fr_raster(onedem, usedask)
    print("done rasterizing and getting possible icebergs")
    print(len(poss_bergs))

    # Exclude icebergs that don't meet the requirements
    bergs, elevs, sl_adjs = filter_pot_bergs(poss_bergs, onedem)

    # delete generator object so no issues between DEMs
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
    col_val = [onedem.attrs['fjord'], onedem['dtime'].values, onedem['tidal_corr'].item(), onedem['min_tidal_ht'].item(), onedem['max_tidal_ht'].item()]
    
    for name,val in (zip(names,col_val)):
        temp_berg_df[name] = val

    print("Generated geodataframe of icebergs for this image")

    return temp_berg_df


def get_poss_bergs_fr_raster(onedem, usedask):
        
    # trans=onedem.attrs['transform']
    flipax=[]
    # if trans[0] < 0:
    #     flipax.append(1)
    # if trans[4] < 0:
    #     flipax.append(0)
    if pd.Series(onedem.x).is_monotonic_decreasing:
        flipax.append(1)
    if pd.Series(onedem.y).is_monotonic_increasing:
        flipax.append(0)
    
    fjord = onedem.attrs['fjord']
    min_area = fjord_props.get_min_berg_area(fjord)
    res = onedem.attrs['res'][0] #Note: the pixel area will be inaccurate if the resolution is not the same in x and y  

    if usedask==True:
        # Daskify the iceberg segmentation process. Note that dask-image has some functionality to operate
        # directly on dask arrays (e.g. dask_image.ndfilters.sobel), which would need to be put into utils.raster.py
        # https://dask-image.readthedocs.io/en/latest/dask_image.ndfilters.html
        # However, as of yet there doesn't appear to be a way to easily implement the watershed segmentation, other than in chunks
        
        # print(onedem)
        # see else statement with non-dask version for descriptions of what each step is doing
        def seg_wrapper(tiles):
            return raster_ops.labeled_from_segmentation(tiles, [3,10], resolution=res, min_area=min_area, flipax=[])
        def filter_wrapper(tiles, elevs):
            return raster_ops.border_filtering(tiles, elevs, flipax=[])

        elev_copy = onedem.elevation.data # should return a dask array
        for ax in flipax:
            elev_copy = da.flip(elev_copy, axis=ax)
        # print(type(elev_copy))

        elev_overlap = da.overlap.overlap(elev_copy, depth=10, boundary='nearest')
        seglabeled_overlap = da.map_overlap(seg_wrapper, elev_overlap, trim=False) # including depth=10 here will ADD another overlap
        print("Got labeled raster of potential icebergs for an image")
        labeled_overlap = da.map_overlap(filter_wrapper, seglabeled_overlap, elev_overlap, trim=False, dtype='int32')
        labeled_arr = da.overlap.trim_overlap(labeled_overlap, depth=10)
        
        # re-flip the labeled_arr so it matches the orientation of the original elev data that's within the xarray
        for ax in flipax:
            labeled_arr = da.flip(labeled_arr, axis=ax)
        # import matplotlib.pyplot as plt
        # print(plt.imshow(labeled_arr))

        try:
            del elev_copy
            del elev_overlap
            del seglabeled_overlap
            del labeled_overlap
            # print("deleted the intermediate steps")
        except NameError:
            pass

        # print(da.min(labeled_arr).compute())
        # print(da.max(labeled_arr).compute())
        
        print("about to get the list of possible bergs")
        print('Please note the transform computation is very application specific (negative y coordinates) and may need generalizing')
        print("this transform computation is particularly sensitive to axis order (y,x) because it is accessed by index number")

        poss_bergs_list = []
        '''
        # I think that by using concatenate=True, it might not actually be using dask for the computation
        
        def get_bergs(labeled_blocks):
            # Note: features.shapes returns a generator. However, if we try to iterate through it with a for loop, the StopIteration exception
            # is not passed up into the for loop and execution hangs when it hits the end of the for loop without completing the function
            block_bergs = list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(
                                labeled_blocks.astype('int32'), transform=onedem.attrs['transform']))[:-1]
            poss_bergs_list.append(block_bergs)
        
        da.blockwise(get_bergs, '', labeled_arr, 'ij', 
                        meta=pd.DataFrame({'c':[]}), concatenate=True).compute()
        # print(poss_bergs_list[0])
        # print(type(poss_bergs_list))
        
        poss_bergs_gdf = gpd.GeoDataFrame({'geometry':[Polygon(poly) for poly in poss_bergs_list[0]]})
        
        # another approach could be to try and coerce the output from map_blocks into an array, but I suspect you'd still have the geospatial issue
        # https://github.com/dask/dask/issues/3590#issuecomment-464609620

        '''

        # URL: https://stackoverflow.com/questions/66232232/produce-vector-output-from-a-dask-array/66245347?noredirect=1#comment117119583_66245347
        def getpx(chunkid, chunksz):
            amin = chunkid[0] * chunksz[0][0]
            amax = amin + chunksz[0][0]
            bmin = chunkid[1] * chunksz[1][0]
            bmax = bmin + chunksz[1][0]
            return (amin, amax, bmin, bmax)

        def get_transform(onedem, chunk0, chunk1):
            # order of all inputs (and outputs) should be y, x when axis order is used
            chunksz = (onedem.chunks['y'], onedem.chunks['x'])
            # rasterio_trans = rasterio.transform.guard_transform(onedem.attrs["transform"])
            # print(rasterio_trans)
            ymini, ymaxi, xmini, xmaxi = getpx((chunk0, chunk1), chunksz) 

            # print(chunk0, chunk1)
            # print(xmini)
            # print(xmaxi)
            # print(ymini)
            # print(ymaxi)

            # use rasterio Windows and rioxarray to construct transform
            # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html#window-transforms
            chwindow = rasterio.windows.Window(xmini, ymini, xmaxi-xmini, ymaxi-ymini)
            return onedem.rio.isel_window(chwindow).rio.transform(recalc=True)
        
        @dask.delayed
        def get_bergs(labeled_blocks, trans):
            
            print("running the dask delayed function")
            # NOTE: From Don: Originally, onedem was called within the delayed function... 
            # I have a feeling this might have caused onedem to be copied in memory a whole bunch of time
            # Among the x number of workers ...
            # I have pulled out the figuring out transform to the outside as a non-delayed function.
            # I found that this transform calculation was very quick, so it should be okay being non-parallel.
            # For future reference, I suggest scattering the data if you want to be able to access it within the workers
            # https://distributed.dask.org/en/latest/locality.html

            return list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(
                                labeled_blocks.astype('int32'), transform=trans))[:-1]

        
        # NOTE: Itertools would flatten the dask delayeds so you don't have a for loop
        # this would make the complexity O(n) rather than O(n^2)
        grid_delayeds = [d for d in it.chain.from_iterable(labeled_arr.to_delayed())]
        for dd in grid_delayeds:
            _, chunk0, chunk1 = dd.key
            trans = get_transform(onedem, chunk0, chunk1)
            piece = get_bergs(dd, trans) # If a function already have delayed decorator, don't need it anymore
            poss_bergs_list.append(piece)
        
        poss_bergs_list = dask.compute(*poss_bergs_list)
        # tried working with this instead of the for loops above
        # poss_bergs_list = dask.compute([get_bergs(bl, *bl.key) for bl in obj for __, obj in enumerate(labeled_arr.to_delayed())])[0]
        # print(poss_bergs_list)

        # unnest the list of polygons returned by using dask to polygonize
        concat_list = [item for sublist in poss_bergs_list for item in sublist if len(item)!=0]
        # print(concat_list)
        
        poss_bergs_gdf = gpd.GeoDataFrame({'geometry':[Polygon(poly) for poly in concat_list]})

        # convert to a geodataframe, combine geometries (in case any bergs were on chunk borders), and generate new polygon list
        print(poss_bergs_gdf)
        # print(poss_bergs_gdf.geometry.plot())
        poss_berg_combined = gpd.overlay(poss_bergs_gdf, poss_bergs_gdf, how='union')
        # print(poss_berg_combined)
        # print(poss_berg_combined.geometry.plot())
        poss_bergs = [berg for berg in poss_berg_combined.geometry]
        # print(poss_bergs)
        print(len(poss_bergs))

        try:
            del labeled_arr
            del poss_bergs_list
            del concat_list
            del poss_berg_combined
        except NameError:
            pass
        
    else:
        print("NOT USING DASK")
        # create copy of elevation values so original dataset values are not impacted by image manipulations
        # and positive/negative coordinate systems can be ignored (note flipax=[] below)
        # something wonky is happening and when I ran this code on Pangeo I needed to NOT flip the elevation values here and then switch the bounding box y value order
        # Not entirely sure what's going on, but need to be aware of this!!
        # print("Note: check for proper orientation of results depending on compute environment. Pangeo results were upside down.")
        elev_copy = np.copy(np.flip(onedem.elevation.values, axis=flipax))
        # flipax=[]
        
        # generate a labeled array of potential iceberg features, excluding those that are too large or small
        seglabeled_arr = raster_ops.labeled_from_segmentation(elev_copy, [3,10], resolution=res, min_area=min_area, flipax=[])
        print("Got labeled raster of potential icebergs for an image")
        # remove features whose borders are >50% no data values (i.e. the "iceberg" edge is really a DEM edge)
        labeled_arr = raster_ops.border_filtering(seglabeled_arr, elev_copy, flipax=[]).astype(seglabeled_arr.dtype)
        # apparently rasterio can't handle int64 inputs, which is what border_filtering returns   

        # import matplotlib.pyplot as plt
        # print(plt.imshow(labeled_arr))
        # create iceberg polygons
        # somehow a < 1 pixel berg made it into this list... I'm doing a secondary filtering by area in the iceberg filter step for now
        poss_bergs = list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(labeled_arr, transform=onedem.attrs['transform']))[:-1]
  
        try:
            del elev_copy
            del seglabeled_arr
            del labeled_arr
        
        except NameError:
            pass

    return poss_bergs


def getexval(potvals, coord, val):
    idx = (np.abs(potvals - val)).argmin()
    nearval = potvals.isel({coord: idx}).item()
    return nearval

def filter_pot_bergs(poss_bergs, onedem):
    """
    Test each potential iceberg for validity, and if valid compute the sea level adjustment and
    get elevation pixel values for putting into the geodataframe.

    Parameter
    ---------
    poss_bergs : list of potential iceberg geometries
    """

    bergs = []
    elevs = []
    sl_adjs = []

    fjord = onedem.attrs['fjord']
    max_freebd = fjord_props.get_ice_thickness(fjord)/10.0
    minfree = fjord_props.get_min_freeboard(fjord)
    res = onedem.attrs['res'][0] #Note: the pixel area will be inaccurate if the resolution is not the same in x and y

    # 10 pixel buffer
    buffer = 10 * res

    for berg in poss_bergs:
        # make a valid shapely Polygon of the berg vertices
        # print(berg)
        origberg = Polygon(berg)
        # print('got a new iceberg')

        if origberg.is_valid == False or origberg.is_empty == True:
            # print("invalid or empty berg geometry")
            continue

        # create a negatively buffered berg outline to exclude border/water pixels
        berg = origberg.buffer(-buffer)
        if berg.is_valid == False or berg.is_empty == True:
            # print("invalid buffered inner-berg geometry")
            continue

        # get the largest polygon from a multipolygon (if one was created during buffering)
        if berg.geom_type == 'MultiPolygon':
            subbergs = list(berg)
            area = []
            for sb in subbergs:
                sb = Polygon(sb)
                area.append(sb.area)
            # print(area)
            idx = np.where(area == np.nanmax(area))[0]
            berg = Polygon(subbergs[idx[0]])
            # print('tried to trim down a multipolygon')
        
        if berg.is_valid == False:
            # print("invalid buffered multipology extraction")
            continue

        # remove holes
        if berg.interiors:
            berg = Polygon(list(berg.exterior.coords))
            # print('removed some holes')
        
        if berg.is_valid == False:
            # print("invalid buffered interiors geometry")
            continue

        # get the polygon complexity and skip if it's above the threshold
        complexity = vector_ops.poly_complexity(berg)
        if complexity >= 0.07:
            # print('border too complex. Removing...')
            continue

        # get the subset (based on a buffered bounding box) of the DEM that contains the iceberg
        # bounds: (minx, miny, maxx, maxy)
        print(onedem.rio._internal_bounds())
        bound_box = origberg.bounds
        try: berg_dem = onedem['elevation'].rio.slice_xy(*bound_box)
        except NoDataInBounds:
            coords = ('x','y','x','y')
            exbound_box = []
            for a, b in zip(bound_box, coords):
                exbound_box.append(getexval(onedem[b], b, a))
            berg_dem = onedem['elevation'].rio.slice_xy(*exbound_box)
            if np.all(np.isnan(berg_dem.values)):
                print("all nan area - no actual berg")
                continue

        # berg_dem = onedem['elevation'].sel(x=slice(bound_box[0]-buffer, bound_box[2]+buffer),
        #                                 # y=slice(bound_box[3]+buffer, bound_box[1]-buffer)) # pangeo? May have been because of issues with applying transform to right-side-up image above?
                                        # y=slice(bound_box[1]-buffer, bound_box[3]+buffer)) # my comp
        
        # print(bound_box)
        # print(np.shape(berg_dem))
        # print(berg_dem)
        # print(berg_dem.elevation.values)

        # extract the iceberg elevation values
        # Note: rioxarray does not carry crs info from the dataset to individual variables
        # print(berg)
        # print(len(bergs))
        # print(berg.area)
        try:
            vals = berg_dem.rio.clip([berg], crs=onedem.attrs['crs']).values.flatten()
        except NoDataInBounds:
            if berg.area < (res**2.0) * 10.0:
                continue
            # vals = berg_dem.rio.clip([berg], crs=onedem.attrs['crs'], all_touched=True).values.flatten()

        # remove nans because causing all kinds of issues down the processing pipeline (returning nan as a result and converting entire array to nan)
        vals = vals[~np.isnan(vals)]
        # print(vals)

        # skip bergs that likely contain a lot of cloud (or otherwise unrealistic elevation) pixels
        if np.nanmedian(vals) > max_freebd:  # units in meters, matching those of the DEM elevation
            # print('"iceberg" too tall. Removing...')
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
        
        # check that the median freeboard elevation (pre-filtering) is at least x m above sea level
        if abs(np.nanmedian(vals)-sl_adj) < minfree:
            # print(np.nanmedian(vals))
            # print(sl_adj)
            print('median iceberg freeboard less than ' +  str(minfree) +' m')
            continue

        # apply the sea level adjustment to the elevation values
        vals = icalcs.apply_decrease_offset(vals, sl_adj)        
        
        bergs.append(berg)
        elevs.append(vals)
        sl_adjs.append(sl_adj)
    
    print(len(bergs))

    return bergs, elevs, sl_adjs

    

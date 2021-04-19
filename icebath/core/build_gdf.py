import datetime as dt
import dask
import dask.array as da
import numpy as np
from geocube.api.core import make_geocube
import geopandas as gpd
import os
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


def xarray_to_gdf(ds):
    """
    Takes an xarray DataSet and generates a geodataframe of icebergs from the DEMs
    """
    berg_gdf = gpd.GeoDataFrame(data=None)

    for num in range(0, len(ds['dtime'])):
        temp_berg_df = gdf_of_bergs(ds.isel({'dtime':num}))
        berg_gdf = berg_gdf.append(temp_berg_df, ignore_index=True) 
    
    # print(berg_gdf)
    # print(len(berg_gdf))
    try:
        berg_gdf.crs = ds.attrs['crs']
        berg_gdf.sl_adjust.attrs['note'] = "sea level adjustment is relative to tidal height, not 0msl"
        berg_gdf.sl_adjust.attrs['units'] = "meters"
    except AttributeError:
        pass

    return berg_gdf


def gdf_of_bergs(onedem, usedask=True):
    """
    Takes an xarray dataarray for one time period and returns the needed geodataframe of icebergs
    """
   
    try:
        onedem.elevation.attrs['crs'] = onedem.attrs['crs']
    except KeyError:
        try:
            onedem.elevation.attrs['proj4'] = onedem.attrs['proj4']
        except KeyError:
            print("Your input DEM does not have a CRS attribute")

    # process the raster and polygonize the potential icebergs
    poss_bergs = get_poss_bergs_fr_raster(onedem, usedask)
    print(len(poss_bergs))

    if len(poss_bergs) == 0:
        return gpd.GeoDataFrame()

    # Exclude icebergs that don't meet the requirements
    bergs, elevs, sl_adjs = filter_pot_bergs(poss_bergs, onedem, usedask)

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
    col_val = [onedem.attrs['fjord'], onedem['dtime'].values, onedem['tidal_corr'].values, onedem['min_tidal_ht'].values, onedem['max_tidal_ht'].values]
    
    for name,val in (zip(names,col_val)):
        temp_berg_df[name] = val

    print("Generated geodataframe of icebergs for this image")

    return temp_berg_df


def get_poss_bergs_fr_raster(onedem, usedask):
        
    flipax=[]
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
            return raster_ops.labeled_from_segmentation(tiles, markers=[3,10], resolution=res, min_area=min_area, flipax=[])
        def filter_wrapper(tiles, elevs):
            return raster_ops.border_filtering(tiles, elevs, flipax=[])

        elev_copy = onedem.elevation.data # should return a dask array
        for ax in flipax:
            elev_copy = da.flip(elev_copy, axis=ax)
        # import matplotlib.pyplot as plt
        # print(plt.imshow(elev_copy))

        try:
            elev_overlap = da.overlap.overlap(elev_copy, depth=10, boundary='nearest')
        except ValueError:
            elev_copy = elev_copy.rechunk(onedem.chunks['x'][0]+1024)
            elev_overlap = da.overlap.overlap(elev_copy, depth=10, boundary='nearest')
        seglabeled_overlap = da.map_overlap(seg_wrapper, elev_overlap, trim=False) # including depth=10 here will ADD another overlap
        labeled_overlap = da.map_overlap(filter_wrapper, seglabeled_overlap, elev_overlap, trim=False, dtype='int32')
        print("Got labeled raster of potential icebergs for an image")
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
        except NameError:
            pass
        
        print("about to get the list of possible bergs")
        print('Please note the transform computation is very application specific (negative y coordinates) and may need generalizing')
        print("this transform computation is particularly sensitive to axis order (y,x) because it is accessed by index number")

        
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

        def get_bl_transform(onedem, chunk0, chunk1):
            # order of all inputs (and outputs) should be y, x when axis order is used
            chunksz = (onedem.chunks['y'], onedem.chunks['x'])
            # rasterio_trans = rasterio.transform.guard_transform(onedem.attrs["transform"])
            # print(rasterio_trans)
            ymini, ymaxi, xmini, xmaxi = getpx((chunk0, chunk1), chunksz) 


            # use rasterio Windows and rioxarray to construct transform
            # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html#window-transforms
            chwindow = rasterio.windows.Window(xmini, ymini, xmaxi-xmini, ymaxi-ymini)
            return onedem.rio.isel_window(chwindow).rio.transform(recalc=True)
        
        @dask.delayed
        def polyganize_bergs(labeled_blocks, trans):
            
            # print("running the dask delayed function")
            # NOTE: From Don: Originally, onedem was called within the delayed function... 
            # I have a feeling this might have caused onedem to be copied in memory a whole bunch of time
            # Among the x number of workers ...
            # I have pulled out the figuring out transform to the outside as a non-delayed function.
            # I found that this transform calculation was very quick, so it should be okay being non-parallel.
            # For future reference, I suggest scattering the data if you want to be able to access it within the workers
            # https://distributed.dask.org/en/latest/locality.html

            return list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(
                                labeled_blocks.astype('int32'), transform=trans))[:-1]

        
        poss_bergs_list = []
        # NOTE: Itertools would flatten the dask delayeds so you don't have a for loop
        # this would make the complexity O(n) rather than O(n^2)
        grid_delayeds = [d for d in it.chain.from_iterable(labeled_arr.to_delayed())]
        for labeled_blocks in grid_delayeds:
            _, chunk0, chunk1 = labeled_blocks.key
            trans = get_bl_transform(onedem, chunk0, chunk1)
            piece = polyganize_bergs(labeled_blocks, trans) # If a function already have delayed decorator, don't need it anymore
            poss_bergs_list.append(piece)

        # for __, obj in enumerate(labeled_arr.to_delayed()):
        #     for bl in obj:
        #         bl_trans = get_bl_trans(onedem, *bl.key)
        #         piece = polyganize_bergs(bl, bl_trans)
        #         # piece = dask.delayed(polyganize_bergs)(bl, *bl.key, chunksz)
        #         poss_bergs_list.append(piece)
        #         del piece
        
        poss_bergs_list = dask.compute(*poss_bergs_list)
        # tried working with this instead of the for loops above
        # poss_bergs_list = dask.compute([get_bergs(bl, *bl.key) for bl in obj for __, obj in enumerate(labeled_arr.to_delayed())])[0]
        # print(poss_bergs_list)

        # unnest the list of polygons returned by using dask to polygonize
        concat_list = [item for sublist in poss_bergs_list for item in sublist if len(item)!=0]
        # print(concat_list)
        
        poss_bergs_gdf = gpd.GeoDataFrame({'geometry':[Polygon(poly) for poly in concat_list]})

        # convert to a geodataframe, combine geometries (in case any bergs were on chunk borders), and generate new polygon list
        # print(poss_bergs_gdf)
        # print(poss_bergs_gdf.geometry.plot())
        if len(poss_bergs_gdf) == 0:
            return poss_bergs_gdf
        else:
            poss_berg_combined = gpd.overlay(poss_bergs_gdf, poss_bergs_gdf, how='union')
            # print(poss_berg_combined)
            # print(poss_berg_combined.geometry.plot())
            poss_bergs = [berg for berg in poss_berg_combined.geometry]
        # print(poss_bergs)
        # print(len(poss_bergs))

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
        # elev_copy = np.copy(onedem.elevation.values)
        # import matplotlib.pyplot as plt
        # print(plt.imshow(elev_copy))
        
        # generate a labeled array of potential iceberg features, excluding those that are too large or small
        seglabeled_arr = raster_ops.labeled_from_segmentation(elev_copy, [3,10], resolution=res, min_area=min_area, flipax=[])
        print("Got labeled raster of potential icebergs for an image")
        # remove features whose borders are >50% no data values (i.e. the "iceberg" edge is really a DEM edge)
        labeled_arr = raster_ops.border_filtering(seglabeled_arr, elev_copy, flipax=[]).astype(seglabeled_arr.dtype)
        # apparently rasterio can't handle int64 inputs, which is what border_filtering returns   

        import matplotlib.pyplot as plt
        print(plt.imshow(labeled_arr))
        # create iceberg polygons
        # somehow a < 1 pixel berg made it into this list... I'm doing a secondary filtering by area in the iceberg filter step for now
        poss_bergs_list = list(poly[0]['coordinates'][0] for poly in rasterio.features.shapes(labeled_arr, transform=onedem.attrs['transform']))[:-1]
  
        poss_bergs = [Polygon(poly) for poly in poss_bergs_list]
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

def filter_pot_bergs(poss_bergs, onedem, usedask):
    """
    Test each potential iceberg for validity, and if valid compute the sea level adjustment and
    get elevation pixel values for putting into the geodataframe.

    Parameter
    ---------
    poss_bergs : list of potential iceberg geometries
    """

    fjord = onedem.attrs['fjord']
    max_freebd = fjord_props.get_ice_thickness(fjord)/10.0
    minfree = fjord_props.get_min_freeboard(fjord)
    res = onedem.attrs['res'][0] #Note: the pixel area will be inaccurate if the resolution is not the same in x and y

    try:
        crs = onedem.attrs['crs']
    except KeyError:
        try:
            crs = onedem.attrs['proj4']
        except KeyError:
            print("Your input DEM does not have a CRS attribute")


    # for berg in poss_bergs:
    #     try: hold = Polygon(berg)
    #     except NotImplementedError:
    #         print(berg)
        
    # Note: list of poss_bergs must be a list of shapely geometry types
    # the previous version, which used Polygon(berg) for berg in poss_bergs in the next line,
    # was a problem when a multipolygon got created after combining results from dask chunks
    poss_gdf = gpd.GeoDataFrame({'origberg': poss_bergs}, geometry='origberg')
    poss_gdf = poss_gdf.set_crs(crs)
    print("Potential icebergs found: " + str(len(poss_gdf)))
    if len(poss_gdf) == 0:
        return [], [], []

    # remove empty or invalid geometries
    poss_gdf = poss_gdf[~poss_gdf.origberg.is_empty]
    poss_gdf = poss_gdf[poss_gdf.origberg.is_valid]
    # print(len(poss_gdf))

    # 10 pixel buffer
    buffer = 10 * res

    # create a negatively buffered berg outline to exclude border/water pixels
    poss_gdf['berg'] = poss_gdf.origberg.buffer(-buffer)

    # get the largest polygon from a multipolygon (if one was created during buffering)
    def get_largest_from_multi(multipolygons):
        bergs = []
        for multipolygon in multipolygons:
            subbergs = list(multipolygon)
            area = []
            for sb in subbergs:
                sb = Polygon(sb)
                area.append(sb.area)
            # print(area)
            idx = np.where(area == np.nanmax(area))[0]
            berg = Polygon(subbergs[idx[0]])
            bergs.append(berg)
        return bergs

    poss_multis = (poss_gdf.berg.geom_type == "MultiPolygon")
    poss_gdf.loc[poss_multis, 'berg'] = get_largest_from_multi(poss_gdf[poss_multis].berg)
    del poss_multis
    # print(len(poss_gdf))

    # remove holes, where present in buffered polygons
    poss_ints = ([len(interior) > 0 for interior in poss_gdf.berg.interiors])
    poss_gdf.loc[poss_ints, 'berg'] = [Polygon(list(getcoords.exterior.coords)) for getcoords in poss_gdf[poss_ints].berg]
    del poss_ints

    poss_gdf = poss_gdf[~poss_gdf.berg.is_empty]
    poss_gdf = poss_gdf[poss_gdf.berg.is_valid]
    # print("Potential icebergs after buffering and invalid, multi, and interior polygons removed: " + str(len(poss_gdf)))

    # get the polygon complexity and remove if it's above the threshold
    poss_gdf['complexity'] = [vector_ops.poly_complexity(oneberg) for oneberg in poss_gdf.berg]
    if res == 2.0:
        complexthresh = 0.07
    elif res ==4.0:
        complexthresh = 0.10
    else:
        complexthresh = 0.08
        print("using a default complexity threshold value - add one for your resolution")
    poss_gdf = poss_gdf[poss_gdf.complexity < complexthresh]
    print("Potential icebergs after complex ones removed: " + str(len(poss_gdf)))
    if len(poss_gdf)  == 0:
        return [], [], []
    poss_gdf = poss_gdf.reset_index().drop(columns=["index", "complexity"])

    total_bounds = poss_gdf.total_bounds
    try: onedem = onedem.rio.slice_xy(*total_bounds)
    except NoDataInBounds:
        coords = ('x','y','x','y')
        exbound_box = []
        for a, b in zip(total_bounds, coords):
            exbound_box.append(getexval(onedem[b], b, a))
        onedem = onedem['elevation'].rio.slice_xy(*exbound_box)
    # onedem = onedem.chunk({'x': 1024, 'y':1024})
    # onedem = onedem.rio.clip_box(*total_bounds).chunk({'x': 1024, 'y':1024})

    # rasterize the icebergs; get the buffered iceberg elevation values for computing draft
    poss_gdf['bergkey'] = poss_gdf.index.astype(int)
    poss_gdf["geometry"] = poss_gdf.berg
    gdf_grid = make_geocube(vector_data=poss_gdf,
                        measurements=["bergkey"],
                        like=onedem,
                        fill=np.nan
                        )

    # gdf_grid = gdf_grid.chunk({'x': 1024, 'y':1024}) #DevGoal: make this a variable
    poss_gdf["freeboardmed"] = [0.0] * len(poss_gdf.index)
    poss_gdf["elevs"] = '' # so that it's object type, not int, for a variable length array
    
    for bkey in poss_gdf["bergkey"]:
        bergdem = onedem.where(gdf_grid["bergkey"] == bkey, drop=True)
        pxvals = bergdem["elevation"].values
        pxvals = pxvals[~np.isnan(pxvals)]
        poss_gdf.at[poss_gdf[poss_gdf["bergkey"]==bkey].index[0], "elevs"] = pxvals
        poss_gdf.at[poss_gdf[poss_gdf["bergkey"]==bkey].index[0], "freeboardmed"] = np.nanmedian(pxvals)
        del bergdem
    del gdf_grid

    # skip bergs that returned all nan elevation values (and thus a nan median value)
    poss_gdf = poss_gdf[poss_gdf["freeboardmed"] != np.nan]
    # print(len(poss_gdf))

    # skip bergs that likely contain a lot of cloud (or otherwise unrealistic elevation) pixels
    poss_gdf = poss_gdf[poss_gdf['freeboardmed'] < max_freebd] # units in meters, matching those of the DEM elevation
    print("Potential icebergs after too-tall ones removed: " + str(len(poss_gdf)))
    if len(poss_gdf) == 0:
        return [], [], []
    # print(poss_gdf)

    # get the regional elevation values and use to determine the sea level adjustment
    def get_sl_poly(origberg):
        """
        Create a polygon (with a hole) for getting pixels to use for the sea level adjustment
        """
        # outer extent of ocean pixels used
        outer = list(origberg.buffer(2*buffer).exterior.coords)
        # inner extent of ocean pixels used
        inner = list(origberg.buffer(buffer).exterior.coords)
        return Polygon(outer, holes=[inner])

    poss_gdf['sl_aroundberg'] = poss_gdf.origberg.apply(get_sl_poly)
    
    def get_sl_adj(sl_aroundberg):
        """
        Clip the polygon from the elevation DEM and get the pixel values.
        Compute the sea level offset
        """
        try:
            slvals = onedem.elevation.rio.clip([sl_aroundberg], crs=onedem.attrs['crs']).values.flatten() #from_disk=True
        except NoDataInBounds:
            if sl_aroundberg.area < (res**2.0) * 10.0:
                slvals = []
            else:
                try:
                    slvals = onedem.elevation.rio.clip([sl_aroundberg], crs=onedem.attrs['crs'], all_touched=True).values.flatten()
                except NoDataInBounds:
                    print("Manually check this DEM for usability")
                    print(sl_aroundberg.area)
                    print((res**2.0) * 10.0)
                    print(onedem.elevation.rio.bounds(recalc=True))
                    print(sl_aroundberg.bounds)

        sl_adj = np.nanmedian(slvals)
        return sl_adj
    # print(onedem)
    onedem['elevation'] = onedem.elevation.rio.write_crs(onedem.attrs['crs'], inplace=True)
    
    # NOTE: sea level adjustment (m) is relative to tidal height at the time of image acquisition, not 0 msl
    poss_gdf["sl_adj"] = poss_gdf.sl_aroundberg.apply(get_sl_adj)

    # check that the median freeboard elevation (pre-filtering) is at least x m above sea level
    poss_gdf = poss_gdf[abs(poss_gdf.freeboardmed - poss_gdf.sl_adj) > minfree]
    print("Potential icebergs after too small ones removed: " + str(len(poss_gdf)))
    if len(poss_gdf) == 0:
        return [], [], []
    
    # apply the sea level adjustment to the elevation values
    def decrease_offset_wrapper(gpdrow):
        corrpxvals = icalcs.apply_decrease_offset(gpdrow["elevs"], gpdrow["sl_adj"])
        # gpdrow["elevs"] = corrpxvals
        return corrpxvals

    poss_gdf["elevs"] = poss_gdf.apply(decrease_offset_wrapper, axis=1)

    print("Final icebergs for estimating water depths: " + str(len(poss_gdf)))

    return poss_gdf.berg, poss_gdf.elevs, poss_gdf.sl_adj


    # Attempts to use dask to eliminate memory crashing issues; some had minor errors, but overall it
    # was coarsening the data that proved most effective. This is also leftover from moving away from groupby

    # if usedask == True:
        
    #     @dask.delayed
    #     def get_berg_px_vals(bkey, onedem, gdf_grid):
    #         pxvals = onedem.where(gdf_grid["bergkey"] == bkey, drop=True)["elevation"].values
    #         pxvals = pxvals[~np.isnan(pxvals)]
    #         return {key, pxvals}

    #     pxdict = {}
    #     print("using dask to iterate through the berg keys")
    #     bkey_delayeds = [d for d in it.chain.from_iterable(poss_gdf["bergkey"])]
    #     for bkey in bkey_delayeds:
    #         keypx_dict = get_berg_px_vals(bkey, onedem, gdf_grid)
    #         pxdict.update(keypx_dict)
    #     pxdict = dask.compute(*pxdict)
    #     print(pxdict)
    #     print(type(pxdict))

    #     for key, val in pxdict.items():
    #             poss_gdf.at[poss_gdf[poss_gdf["bergkey"]==key].index[0], "elevs"] = val
    #             poss_gdf.at[poss_gdf[poss_gdf["bergkey"]==key].index[0], "freeboardmed"] = np.nanmedian(val)
            
    #     del pxdict

    '''
    gdf_grid['elev'] = onedem.reset_coords(drop=True)["elevation"]
    gdf_grid = gdf_grid.chunk({'x': 1024, 'y':1024}) #DevGoal: make this a variable
    grouped = gdf_grid.drop("spatial_ref").groupby(gdf_grid.bergkey)
    
    @dask.delayed
    def get_berg_px_vals(key, vals):
        pxvals = vals.elev.values
        pxvals = pxvals[~np.isnan(pxvals)]
        return {key: pxvals}

    if usedask == True:
        pxdict = {}
        print("using dask to iterate through the groups")
        group_delayeds = [d for d in it.chain.from_iterable(grouped.to_delayed())]
        for key, vals in group_delayeds:
            keypx_dict = get_berg_px_vals(key, vals)
            pxdict.update(keypx_dict)
        pxdict = dask.compute(*pxdict)
        print(pxdict)
        print(type(pxdict))

        for key, val in pxdict.items():
            poss_gdf.at[poss_gdf[poss_gdf["bergkey"]==key].index[0], "elevs"] = val
            poss_gdf.at[poss_gdf[poss_gdf["bergkey"]==key].index[0], "freeboardmed"] = np.nanmedian(val)
        
        del pxdict

    else:
        for key, vals in grouped:
            pxvals = vals.elev.values
            pxvals = pxvals[~np.isnan(pxvals)]
            poss_gdf.at[poss_gdf[poss_gdf["bergkey"]==key].index[0], "elevs"] = pxvals
            poss_gdf.at[poss_gdf[poss_gdf["bergkey"]==key].index[0], "freeboardmed"] = np.nanmedian(pxvals)
    del grouped
    '''

    # NOTE: sea level adjustment (m) is relative to tidal height at the time of image acquisition, not 0 msl
    # if usedask == True:
    #     print("using dask geopandas to iterate through the bergs")
    #     import dask_geopandas as dgpd
    #     dask_poss_gdf = dgpd.from_geopandas(poss_gdf, npartitions=2)
    #     sl_adjs = dask_poss_gdf.apply(get_sl_adj).compute()
    #     poss_gdf["sl_adj"] = sl_adjs
    #     del dask_poss_gdf
    #     del sl_adjs
    # else:
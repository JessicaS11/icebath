import datetime as dt
import numpy as np
import os
import pandas as pd
import rasterio.transform
from rasterio.errors import RasterioIOError
import rioxarray
import xarray as xr
import warnings

from icebath.core import fjord_props

import faulthandler
faulthandler.enable()

def xrds_from_dir(path=None, fjord=None, metastr='_mdf'):
    """
    Builds an XArray dataset of DEMs for finding icebergs when passed a path to a directory"
    """
    warnings.warn("This function currently assumes a constant grid and EPSG for all input files")
    assert fjord != None, "You must specify the fjord code for these DEMs"

    files = [f for f in os.listdir(path) if f.endswith('dem.tif')]

    # for DEMs nested in directories
    if len(files) == 0:
        try: 
            os.remove(path+'.DS_Store')
        except FileNotFoundError:
            pass
        dirs = [dir for dir in os.listdir(path)]
        nestfiles=[]
        for dir in dirs:
            nestfiles.append([dir+'/'+f for f in os.listdir(path+dir) if f.endswith('dem.tif')])
        files = [items for nest in nestfiles for items in nest]

    i=0
    darrays = list(np.zeros(len(files)))
    dtimes = list(np.zeros(len(files)))
    for f in files:
        print(f)

        metaf = f.rpartition("_dem.tif")[0] +  metastr + ".txt"
        try:
            meta = read_meta(path+metaf)
            # print(meta)
            dtimes[i] = get_DEM_img_times(meta)
        except FileNotFoundError:
            print("You must manually enter dtimes for these files within the code")
            # dtimes[0] = dt.datetime(2012, 6, 29, hour=15, minute=26, second=30)
            # dtimes[1] = dt.datetime(2010, 8, 14, hour=15, minute=34)
        # except KeyError:
        #     raise
        except AssertionError:
            print("These stereo image times are >30 min apart... skipped")
            continue

        try:
            darrays[i] = read_DEM(path+f, fjord)
            # darrays[i] = read_DEM(path+f.rpartition("_dem.tif")[0] + "_dem_geoidcomp.tif")
        except RasterioIOError:
            print("RasterioIOError on your input file")
            break        

        i = i + 1

    if len(darrays)==1:
        assert np.all(darrays[0]) != 0, "Your DEM will not be put into XArray"
    else:
        assert np.all(darrays[darrays!=0]) != 0, "None of your DEMs will be put into XArray"
    
    # darr = xr.combine_nested(darrays, concat_dim=['dtime'])
    darr = xr.concat(darrays, 
                    dim=pd.Index(dtimes, name='dtime'), 
                    # coords=['x','y'], 
                    join='outer').chunk({'dtime': 1, 'x':2000, 'y':2000}) # figure out a better value for chunking this (it slows the JI one with 3 dems way down)
                    # combine_attrs='no_conflicts' # only in newest version of xarray

    try:
        for arr in darrays:
            arr.close()
            print('closed the arrays')
    except:
        pass
    del darrays
    
    # convert to dataset with elevation as a variable and add attributes
    attr = darr.attrs
    ds = darr.to_dataset()
    ds.attrs = attr
    ds.attrs['fjord'] = fjord
    attr=None
    # newest version of xarray (0.16) has promote_attrs=True kwarg. Earlier versions don't...
    # ds = ds.to_dataset(name='elevation', promote_attrs=True).squeeze().drop('band')
    
    # using rioxarray means the transform is read in/created as part of the geospatial info, so it's unnecessary to manually create a transform
    # create affine transform for concatted dataset
    print('Please note the transform is computed assuming a coordinate reference system\
 where x(min) is west and y(min) is south')
    # inputs: west, south, east, north, width, height
    transform = rasterio.transform.from_bounds(ds.x.min().item()-0.5*ds.attrs['res'][0], ds.y.min().item()-0.5*ds.attrs['res'][1], 
                                             ds.x.max().item()+0.5*ds.attrs['res'][0], ds.y.max().item()+0.5*ds.attrs['res'][1], 
                                             len(ds.x), len(ds.y))
    ds.attrs['transform'] = transform
    # set the transform and crs as attributes since that's how they're accessed later in the pipeline
    # ds.attrs['transform'] = (ds.spatial_ref.GeoTransform)
    # ds.attrs['crs'] = ds.spatial_ref.crs_wkt
    
    return ds


def read_DEM(fn=None, fjord=None):
    """
    Reads in the DEM (only accepts GeoTiffs right now) into an XArray Dataarray with the desired format.
    """
    # intake.open_rasterio accepts a list of input files and may effectively do what this function does!
    # try using cropped versions of the input files. Doesn't seem to make a difference r.e. crashing
    '''
    cropped_fn = fn.rpartition(".tif")[0] + "_cropped.tif"
    print(cropped_fn)
    if os._exists(cropped_fn):
        fn = cropped_fn
    elif fjord != None:
        bbox = fjord_props.get_fjord_bounds(fjord)
        ds = rioxarray.open_rasterio(fn)
        trimmed_ds = ds.rio.slice_xy(*bbox)
        trimmed_ds.rio.to_raster(fn.rpartition(".tif")[0] + "_cropped.tif")
        del ds
        del trimmed_ds
        fn = cropped_fn   
    '''

    # Rasterio automatically checks that the file exists
    # ultimately switch to using rioxarray, but it causes issues down the pipeline so it will need to be debugged through
    # with rioxarray.open_rasterio(fn) as src:
    with xr.open_rasterio(fn) as darr:
        # darr = src

        # open_rasterio automatically brings the geotiff in as a DataArray with 'band' as a dimensional coordinate
        # we rename it and remove the band as a coordinate, since our DEM only has one dimension
        # squeeze removes dimensions of length 0 or 1, in this case our 'band'
        # Then, drop('band') actually removes the 'band' dimension from the Dataset
        darr = darr.rename('elevation').squeeze().drop('band')
        # darr = darr.rename({'band':'dtime'})
    
        # if we wanted to instead convert it to a dataset
        # attr = darr.attrs
        # darr = darr.to_dataset(name='elevation').squeeze().drop('band')
        # darr.attrs = attr
        # attr=None
        # newest version of xarray (0.16) has promote_attrs=True kwarg. Earlier versions don't...
        # darr = darr.to_dataset(name='elevation', promote_attrs=True).squeeze().drop('band')

        # mask out the nodata values, since the nodatavals attribute is wrong
        darr = darr.where(darr != -9999.)

        # the gdalwarp geoid files have this extra attribute in the geoTiff, which when brought in
        # ultimately causes a "__module__" related error when trying to plot with hvplot
        try:
            del darr.attrs["units"] 
        except KeyError:
            pass

        if fjord != None:
            # USE RIOXARRAY - specifically, slicexy() which can be fed the bounding box
            # darr = darr.rio.slice_xy(fjord_props.get_fjord_bounds(fjord))
            bbox = fjord_props.get_fjord_bounds(fjord)
            if pd.Series(darr.y).is_monotonic_increasing:
                darr = darr.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))
            else:
                darr = darr.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1]))
        
        return darr


def get_dtime(string, startidx, endidx, fmtstr):    
    # print(string)
    # print(string[startidx:endidx])
    if '/' in string:
        string = string.split("/")[-1]
        # print(string)
    dtime = dt.datetime.strptime(string[startidx:endidx], fmtstr)
    return dtime

def get_DEM_img_times(meta):
    
    posskeys = {"sourceImage1":"sourceImage2",
                "Image_1_Acquisition_time":"Image_2_Acquisition_time",
                "Image 1":"Image 2"}
    # posskeys = {"fakekey": 12}

    dtstrings = {"sourceImage1":(6,20, '%Y%m%d%H%M%S'),
                "Image_1_Acquisition_time":(0, -1,  '%Y-%m-%dT%H:%M:%S.%f'),
                "Image 1":(5,19, '%Y%m%d%H%M%S')}  # -70,-56

    if np.any([thekey in list(meta.keys()) for thekey in list(posskeys.keys())]):
        pass
    else:
        raise KeyError("Appropriate metadata keys are not available or need to be added to the list")

    for key in list(posskeys.keys()):
        if key in list(meta.keys()):
            d1strs = meta[key]
            d2strs = meta[posskeys[key]]
            break
        else:
            continue

    dtime1list = []
    dtime2list = []
    for (d1str, d2str) in zip(d1strs, d2strs):
        dtime1list.append(get_dtime(d1str, *dtstrings[key]))
        dtime2list.append(get_dtime(d2str, *dtstrings[key]))
    
    dtimelist = []
    for i in range(0, len(dtime1list)):
        assert (abs(dtime1list[i]-dtime2list[i]) < dt.timedelta(minutes=30)), "These stereo image times are >30 min apart"
        dtimelist.append(dt.datetime.fromtimestamp((dtime1list[i].timestamp() + dtime2list[i].timestamp())//2))
        
        if i > 0:
            assert (abs(dtimelist[i]-dtimelist[i-1]) < dt.timedelta(minutes=30)), "These DEM times are >30 min apart"
        
    if len(dtimelist) == 1:
        dtime = dtimelist[0]
    else:
        sumdtime = dt.datetime.fromtimestamp(np.sum(dtime.timestamp() for dtime in dtimelist))
        dtime = dt.datetime.fromtimestamp(sumdtime.timestamp()//len(dtimelist))

    return dtime


def add_to_xrds(xrds, add_xr, **kwargs):
    """
    adds the XArray dataarray or dataset to the existing dataset
    """
    # create the dataset if it doesn't exist
    if xrds==None:
        xrds = xr.Dataset(coords={'dtime':[], 'x':('dtime',[]), 'y':('dtime',[])}, 
                        data_vars={'elevation':('dtime', [])})

    print("not to be implemented at this time - see notes in module code")

"""
Notes on working with XArray
- the issue of reprojecting and regridding raster data is an ongoing one in xarray.
- Right now, there's no clear path for combining the dataarrays into a dataset (even with rioxarray)
- My initial approach was to open each geotiff as a dataarray, manually restructure the dataarray,
  then add the dataarray to a main dataset. This still does not solve the grid issues (can there be
  separate x,y coordinates for each time, if x and y depend on time? When/where/how does regridding
  and resampling of the data happen?)
"""


def read_meta(metafn=None):
    """
    Read a [metadata] text file into a dictionary.
    Function modified from https://github.com/JessicaS11/python_geotif_fns/blob/master/Landsat_TOARefl.py
    """

    metadata = {}

    # potential future improvement: strip quotation marks from strings, where applicable. Will then need to adjust
    # the indices used to get the dates and times in the functions above 
    # (get_DEM_img_times: dtstrings = {"sourceImage1":(5,19, '%Y%m%d%H%M%S')})

    #each key is equated with '='. This loop strips and seperates then fills the dictonary.
    with open(metafn) as f:    
        for line in f:
            if not line.strip(';') == "END":
                val = line.strip().split('=')
                if len(val) == 1:
                    continue
                else:
                    metadata.setdefault(val[0].strip(), []).append(val[1].strip().strip(';'))     
            else:
                break
	
    return metadata
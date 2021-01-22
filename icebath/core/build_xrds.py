import datetime as dt
import numpy as np
import os
import pandas as pd
import rasterio.transform
from rasterio.errors import RasterioIOError
import xarray as xr
import warnings

from icebath.core import fjord_props

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
        # ToDo: Update this to carry through the correct assertion error? Or return more explicit errors?
        # except AssertionError:
        #     print("These stereo image times are >30 min apart... skipped")
        #     continue

        try:
            darrays[i] = read_DEM(path+f)
            # darrays[i] = read_DEM(path+f.rpartition("_dem.tif")[0] + "_dem_geoidcomp.tif")
        except RasterioIOError:
            print("RasterioIOError on your input file")
            break        

        i = i + 1

    # assert np.all(darrays) != 0, "None of your DEMs will be put into Xarray"
    assert np.all(darrays[darrays!=0]) != 0, "None of your DEMs will be put into XArray"
    # darr = xr.combine_nested(darrays, concat_dim=['dtime'])
    darr = xr.concat(darrays, dim=pd.Index(dtimes, name='dtime'))#, 
                    # coords=['x','y'], join='outer')
                    # combine_attrs='no_conflicts' # only in newest version of xarray

    # convert to dataset with elevation as a variable and add attributes
    attr = darr.attrs
    ds = darr.to_dataset()
    ds.attrs = attr
    ds.attrs['fjord'] = fjord
    attr=None
    # newest version of xarray (0.16) has promote_attrs=True kwarg. Earlier versions don't...
    # ds = ds.to_dataset(name='elevation', promote_attrs=True).squeeze().drop('band')
    
    # create affine transform for concatted dataset
    print('Please note the transform is computed assuming a coordinate reference system\
 where x(min) is west and y(min) is south')
    # inputs: west, south, east, north, width, height
    transform = rasterio.transform.from_bounds(ds.x.min().item()-0.5*ds.attrs['res'][0], ds.y.min().item()-0.5*ds.attrs['res'][1], 
                                             ds.x.max().item()+0.5*ds.attrs['res'][0], ds.y.max().item()+0.5*ds.attrs['res'][1], 
                                             len(ds.x), len(ds.y))
    ds.attrs['transform'] = transform
    
    return ds


def read_DEM(fn=None):
    """
    Reads in the DEM (only accepts GeoTiffs right now) into an XArray Dataarray with the desired format
    """
    
    # Rasterio automatically checks that the file exists
    with xr.open_rasterio(fn) as src:
        darr = src

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

    return darr


def get_dtime(string, startidx, endidx, fmtstr):
    dtime = dt.datetime.strptime(string[startidx:endidx], fmtstr)
    return dtime

def get_DEM_img_times(meta):
    
    posskeys = {"sourceImage1":"sourceImage2",
                "Image_1_Acquisition_time":"Image_2_Acquisition_time",
                "Image 1":"Image 2"}

    dtstrings = {"sourceImage1":(6,20, '%Y%m%d%H%M%S'),
                "Image_1_Acquisition_time":(0, -1,  '%Y-%m-%dT%H:%M:%S.%fZ'),
                "Image 1":(6,20, '%Y%m%d%H%M%S')}  # still need to update this...

    # print(posskeys.keys())
    # print(meta.keys())
    assert np.any(list(posskeys.keys()) in thekeys for thekeys in list(meta.keys())), "Appropriate metadata keys are not available or need to be added to the list"

    print("checkpoint 1")
    for key in list(posskeys.keys()):
        if key in list(meta.keys()):
            d1strs = meta[key]
            d2strs = meta[posskeys[key]]
            print(d1strs)
            print("pre-break")
            break
        else:
            continue
    
    print(d1strs)
    if len(d1strs) == 1:
        dtime1 = get_dtime(d1strs, *dtstrings[key])
    else:
        dtime1list = []
        for d1str in d1strs:
            dtime1list.append(get_dtime(d1str, *dtstrings[key]))
            dtime1 = np.mean(dtime1list)

    if len(d2strs) == 1:
        dtime2 = get_dtime(d2strs, *dtstrings[key])
    else:
        dtime2list = []
        for d2str in d2strs:
            dtime2list.append(get_dtime(d2str, *dtstrings[key]))
            dtime2 = np.mean(dtime2list)

    
    
    # try:
    #     img1 = meta["sourceImage1"]
    #     img2 = meta["sourceImage2"]
    #     dtime1 = dt.datetime.strptime(img1[6:20], '%Y%m%d%H%M%S')
    #     dtime2 = dt.datetime.strptime(img2[6:20], '%Y%m%d%H%M%S')
    # except KeyError:
    #     img1 = meta['Image_1_Acquisition_time']
    #     img2 = meta['Image_2_Acquisition_time']
    #     dtime1 = dt.datetime.strptime(img1, '%Y-%m-%dT%H:%M:%S.%fZ')
    #     dtime2 = dt.datetime.strptime(img2, '%Y-%m-%dT%H:%M:%S.%fZ') 

    # # START HERE with trying to get it to read yet another format of metadata  

    assert (abs(dtime1-dtime2) < dt.timedelta(minutes=30)), "These stereo image times are >30 min apart"
    dtime = dt.datetime.fromtimestamp((dtime1.timestamp() + dtime2.timestamp())//2)

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
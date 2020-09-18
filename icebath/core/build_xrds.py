# import rasterio
import datetime as dt
import numpy as np
import os
import pandas as pd
import xarray as xr
import warnings

from icebath.core import fjord_props

def xrds_from_dir(path=None):
    """
    Builds an XArray dataset of DEMs for finding icebergs when passed a path to a directory"
    """
    warnings.warn("This function currently assumes a constant grid and EPSG for all input files")

    files = [f for f in os.listdir(path) if f.endswith('.tif')]

    i=0
    darrays = list(np.zeros(len(files)))
    dtimes = list(np.zeros(len(files)))
    for f in files:
        darrays[i], dtimes[i] = read_DEM(path+f)
        dtimes[i] = dtimes[i] + (i * dt.timedelta(hours=12))
        # If need to set time on dataarray directly, must use numpy timedelta instead
        # ds['dtime'] = [ds.dtime.values[0], ds.dtime.values[1]+np.timedelta64(12,'h')]
        i = i + 1

    # darr = xr.combine_nested(darrays, concat_dim=['dtime'])
    darr = xr.concat(darrays, dim=pd.Index(dtimes, name='dtime'))#, 
                    # coords=['x','y'], join='outer')

    # convert to dataset with elevation as a variable
    attr = darr.attrs
    ds = darr.to_dataset()
    ds.attrs = attr
    attr=None
    # newest version of xarray (0.16) has promote_attrs=True kwarg. Earlier versions don't...
    # ds = ds.to_dataset(name='elevation', promote_attrs=True).squeeze().drop('band')
    return ds


def read_DEM(fn=None):
    """
    Reads in the DEM (only accepts GeoTiffs right now) into an XArray Dataarray with the desired format
    """
    
    # Rasterio automatically checks that the file exists
    darr = xr.open_rasterio(fn)

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

    # add time as a dimension
    # for now fill in an arbitrary, artificial value
    dtime = dt.datetime(2012, 8, 16, hour=10, minute=33)
    # darr.attrs['date-time'] = dtime
    # darr = darr.assign_coords(coords={'date-time': dtime})
    # darr['dtime'] = [dtime]

    return darr, dtime


def get_tidal_pred(loc=None, img_time=None, model_path='/home/jovyan/pyTMD/models',
                    model='AOTIM-5-2018', epsg=3413):
 
    assert loc!=None, "You must enter a location!"

    loc = fjord_props.get_mouth_coords(loc)

    st_time = img_time - dt.timedelta(hours=-12)
    
    # time series for +/- 12 hours of image time, or over 24 hours, every 5 minutes
    ts = np.arange(0, 24*60*60, 5*60)
    xs = np.ones(len(ts))*loc[0]
    ys = np.ones(len(ts))*loc[1]

    tide_pred = compute_tide_corrections(xs,ys,ts,
        DIRECTORY='/home/jovyan/pyTMD/models', MODEL=model,
        EPOCH=st_time.timetuple()[0:6], TYPE='drift', TIME='utc', EPSG=epsg)

    # could add a test here that tide_pred.mask is all false to make sure didn't get any land pixels

    return ts, tide_pred.data
    



def add_to_xrds(xrds, add_xr, **kwargs):
    """
    adds the XArray dataarray or dataset to the existing dataset
    """
    # create the dataset if it doesn't exist
    if xrds==None:
        xrds = xr.Dataset(coords={'date-time':[], 'x':('date-time',[]), 'y':('date-time',[])}, 
                        data_vars={'elevation':('date-time', [])})

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
import datetime as dt
import numpy as np
import os
import pandas as pd
import rasterio.transform
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
        print(f)
        darrays[i], dtimes[i] = read_DEM(path+f)
        i = i + 1

    print('NOTE: currently dates and times are hard-coded in. Need to automate this in read_DEM still')
    # If need to set time on dataarray directly, must use numpy timedelta instead
    # ds['dtime'] = [ds.dtime.values[0], ds.dtime.values[1]+np.timedelta64(12,'h')]
    dtimes[0] = dt.datetime(2012, 6, 29, hour=15, minute=26, second=30)
    dtimes[1] = dt.datetime(2010, 8, 14, hour=15, minute=34)

    # darr = xr.combine_nested(darrays, concat_dim=['dtime'])
    darr = xr.concat(darrays, dim=pd.Index(dtimes, name='dtime'))#, 
                    # coords=['x','y'], join='outer')
                    # combine_attrs='no_conflicts' # must only be in newest version of xarray

    # convert to dataset with elevation as a variable
    attr = darr.attrs
    ds = darr.to_dataset()
    ds.attrs = attr
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
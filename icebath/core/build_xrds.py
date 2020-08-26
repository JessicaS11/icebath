# import rasterio
import datetime as dt
from pyTMD import predict_tidal_ts
import xarray

def from_dir(path=None):
    """
    Builds an XArray dataset of DEMs for finding icebergs when passed a path to a directory"
    """
    print("a work in progress")


def read_DEM(fn=None):
    """
    Reads in the DEM (only accepts GeoTiffs right now) into an XArray Dataset
    """
    
    #Check that file exists...

    data = xr.open_rasterio(fn)

    # convert the dataarray to a dataset and transfer attributes
    # open_rasterio automatically brings the geotiff in as a DataArray, with 'band' as a dimensional coordinate.
    # Thus, we convert it to a dataset so that the band, which for a DEM/single band geotiff, only has one dimension
    # squeeze removes dimensions of length 0 or 1, in this case our 'band', so that the data variable is no longer using it as a dimension
    # Then, drop('band') actually removes the 'band' dimension from the Dataset
    attr = data.attrs
    data = data.to_dataset(name='elevation').squeeze().drop('band')
    data.attrs = attr
    attr=None

    # newest version of xarray (0.16) has promote_attrs=True kwarg. Earlier versions don't...
    # data = data.to_dataset(name='elevation', promote_attrs=True).squeeze().drop('band')
    
    # mask out the nodata values
    data = data.where(data['elevation'] != -9999.)

    # add time coordinate dimension
    # dtime = #get this from the filename or a lookup table...
    # for now fill in an arbitrary, artificial value
    dtime = dt.datetime(2012, 8, 16, hour=10, minute=33)
    data = data.assign_coords(coords={'date-time': dtime})


def get_tidal_pred(loc=None, img_time=None):
    st_time = img_time - dt.timedelta(hours=-12)
    deltat = dt.timedelta(days=1)

    # st_time_days = st_time-dt.datetime(1992, 1, 1, hour=0, minute=0, second=0)

    # ht = predict_tidal_ts()


def add_to_xrds(xrds, add_xr, **kwargs):
    """
    adds the XArray dataarray or dataset to the existing dataset
    """

    print("not yet implemented")

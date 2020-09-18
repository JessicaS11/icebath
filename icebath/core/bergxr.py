import pandas as pd
import numpy as np
# import scipy.io as spio
# import scipy.stats as stats
# import ogr
# import os
# import fnmatch
import xarray as xr

from icebath.utils import raster_ops
from icebath.core import fl_ice_calcs as icalcs
from icebath.core import fjord_props


@xr.register_dataset_accessor("bergxr")
class BergXR:
    """
    An extension for an XArray dataset that will calculate tidal offsets and delineate
    icebergs in DEMs brought in from a GeoTiff.
    """

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        xrds,
    ):

        self._xrds = xrds
        self._validate(self, req_dim = ['x','y','dtime'], req_vars = {'elevation':['x','y','dtime']})
        
       


    # ----------------------------------------------------------------------
    # Properties


    # ----------------------------------------------------------------------
    # Methods

    @staticmethod
    def _validate(self, req_dim=None, req_vars=None):
        '''
        Make sure the xarray dataset (or dataarray) has the correct coordinates and variables
        '''

        # if type(xrds) == dataarray
        if req_dim is not None:
            if all([dim not in list(self._xrds.dims) for dim in req_dim]):
                raise AttributeError("Required dimensions are missing")
        if req_vars is not None:
            if all([var not in self._xrds.variables for var in req_vars.keys()]):
                raise AttributeError("Required variables are missing")
    
        #if type(xrds) == dataset
        # for a dataset rather than a datarray
        # if all([dim not in list(xrds.dims.keys()) for dim in req_dim]):
        #     raise AttributeError("Required dimensions are missing")
        # if all ([var not in list(xrds.keys()) for var in req_vars.keys()]):
        #     raise AttributeError("Required variables are missing")
        # for key in req_vars.keys():
        #     if all([dim not in list(xrds[key].dims) for dim in req_vars[key]]):
        #         raise AttributeError("Variables do not have all required coordinates")

    
  
    # def calc_medmaxmad(self, column=''):
    #     """
    #     Compute median, maximum, and median absolute devation from an array of values
    #     specified by the string of the input column name and add columns to hold the results.
    #     Input values might be from a filtered raster of iceberg pixel drafts or a series of measurements.
        
    #     Parameters
    #     ---------
    #     column: str, default ''
    #         Column name on which to compute median, maximum, and median absolute deviation
    #     """

        
    #     req_cols = [column] # e.g. 'draft' for iceberg water depths, 'depth' for measured depths
    #     self._validate(self._gdf, req_cols)


    def get_tidal_pred(self, req_dim=['dtime'], req_vars=None,
                        loc=None, **kwargs):
    
        assert loc!=None, "You must enter a location!"
        self._validate(self, req_dim, req_vars)
        
        # kwargs: model_path='/home/jovyan/pyTMD/models',
        #                 **model='AOTIM-5-2018', **epsg=3413


        loc = fjord_props.get_mouth_coords(loc)

        i=0
        tide_arr = np.zeros(len(self._xrds.dtime.values))
        for t in xrds.dtime.values:
            time, tidal_ht = icalcs.predict_tides(loc='JI',img_time=t, 
                                                    model_path='/home/jovyan/pyTMD/models',
                                                    model='AOTIM-5-2018', epsg=3413, plot=False)
            
            tidx = list(t).index(dt.timedelta(hours=12).seconds)
            vals = [tidal_ht[tidx], np.min(tidal_ht), np.max(tidal_ht)]
            tide_arr[i] = vals
            i = i+1
        
        self._xrds = self._xrds.assign(tides=('dtime',tide_arr)) # this will throw an error because tide_arr elements each have three elements
        self._xrds=self._xrds.tides.attrs({'value_key':'tidal_ht_corr(m), tide_min(m),tide_max(m)'}) 

        # gb= self._xrds.groupby('dtime')
        # .apply(icalcs.predict_tides, plot=False)


    def get_icebergs(self, req_dim=['dtime','x','y'], 
                    req_vars={'elevation':['x','y','dtime']}, threshold=None):
        '''
        Get iceberg polygons for each DEM in the dataset
        '''

        self._validate(self, req_dim, req_vars)
        
        if threshold:
            self._xrds.attrs['berg_threshold'] = threshold
        else:
            print("Remember to set a desired threshold!")

        self._xrds=self._xrds.groupby('dtime', squeeze=False).apply(self._poly_from_thresh_wrapper)
        # print(self._xrds)

        return self._xrds

    def _poly_from_thresh_wrapper(self,gb):
        """
        XArray wrapper for the raster_ops.poly_from_thresh function to be able to use it with
        `.groupby().apply()`
        
        Parameters
        ----------
        gb : groupby object
            Must contain the fields x, y, elev...
        """

        x = gb.x
        y = gb.y
        elev = gb.elevation.isel(dtime=0)
        
        polys = raster_ops.poly_from_thresh(x,y,elev,gb.attrs['berg_threshold'])
        bergs=pd.Series({'bergs':[polys]}, dtype='object')

        return gb.assign(berg_outlines=('dtime',bergs))



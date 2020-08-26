# import pandas as pd
import numpy as np
# import scipy.io as spio
# import scipy.stats as stats
# import ogr
# import os
# import fnmatch
import xarray as xr




@xr.register_dataset_accessor("berg")
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

        self._validate(xrds, req_dim = ['x','y','date-time'], req_vars = {'elevation':['x','y','date-time']})
        self._xrds = xrds
       


    # ----------------------------------------------------------------------
    # Properties


    # ----------------------------------------------------------------------
    # Methods

    @staticmethod
    def _validate(xrds, req_dim=None, req_vars=None):
        '''
        Make sure the xarray dataset has the correct coordinates and variables
        '''

        if all([dim not in list(xrds.dims.keys()) for dim in req_dim]):
            raise AttributeError("Required dimensions are missing")
        if all ([var not in list(xrds.keys()) for var in req_vars.keys()]):
            raise AttributeError("Required variables are missing")
        for key in req_vars.keys():
            if all([dim not in list(xrds[key].dims) for dim in req_vars[key]):
                raise AttributeError("Variables do not have all required coordinates")

    
  
    def calc_medmaxmad(self, column=''):
        """
        Compute median, maximum, and median absolute devation from an array of values
        specified by the string of the input column name and add columns to hold the results.
        Input values might be from a filtered raster of iceberg pixel drafts or a series of measurements.
        
        Parameters
        ---------
        column: str, default ''
            Column name on which to compute median, maximum, and median absolute deviation
        """

        
        req_cols = [column] # e.g. 'draft' for iceberg water depths, 'depth' for measured depths
        self._validate(self._gdf, req_cols)


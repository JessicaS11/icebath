import pandas as pd
import numpy as np
import scipy.io as spio
import ogr
import os
import fnmatch
import geopandas as gpd

"""
Notes:
currently waiting to find out (via StackExchange) if I can create a pandas extension on a geodataframe...
otherwise I'll have to come up with plan B

https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-register-accessors
https://stackoverflow.com/questions/63160557/creating-a-pandas-extension-on-a-geopandas-dataframe
"""
'''
@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

'''



@pd.api.extensions.register_dataframe_accessor("berg")
class BergGDF:
    """

    """

    DEM_ARR = 'DEMarray'
    SLADJ = 'sea_level_adj'
    DATE = 'date'
    TIDE_HT = 'modeled_tide_ht'
    
    # # ----------------------------------------------------------------------
    # # Constructors

    def __init__(
        self,
        geopandas_obj    
    ):

    # self._validate(geopandas_obj)
        self._obj = geopandas_obj

    def offcenter(self):
        return self._obj.geometry.centroid #+ [15, 20] 

    # @staticmethod
    # def _validate(obj):
    #     # verify there is a column latitude and a column longitude
    #     if 'latitude' not in obj.columns or 'longitude' not in obj.columns:
    #         raise AttributeError("Must have 'latitude' and 'longitude'.")


    # # ----------------------------------------------------------------------
    # # Properties

    # @property
    # def dataset(self):
    #     """
    #     Return the short name dataset ID string associated with the query object.

    #     Examples
    #     --------
    #     >>> reg_a = icepyx.query.Query('ATL06',[-55, 68, -48, 71],['2019-02-20','2019-02-28'])
    #     >>> reg_a.dataset
    #     'ATL06'
    #     """
    #     return self._dset    


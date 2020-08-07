import pandas as pd
import numpy as np
import scipy.io as spio
import scipy.stats as stats
import ogr
import os
import fnmatch
import geopandas as gpd
from icebath.core import fjord_props as fjord
from icebath.core import fl_ice_calcs as icalcs


"""
Dev notes:

Columns ultimately used/needed:  
        cols = ['fjord',
        'DEMarray',
        'sl_adjust',
        'date',
        'modeled_tide_ht',
        "filtered_draft",
        "depth_median",
        "depth_max",
        "depth_mad",
        ]
"""


@pd.api.extensions.register_dataframe_accessor("berg")
class BergGDF:
    """
    An extension for a pandas dataframe that brings in an iceberg per row from the data source and 
    does the needed calculations to infer water depth from icebergs
    """

    #density of iceberg ice, kg/m3
    rho_i = 900
    rho_i_err = 20
    
    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        geodf,
        #NOTE: as of pandas 1.1.0, there is now a beta attribute attrs, which is a dictionary of attributes!!!   
    ):

        self._validate(geodf, req_cols = ['date'])
        self._gdf = geodf
       


    # ----------------------------------------------------------------------
    # Properties


    # ----------------------------------------------------------------------
    # Methods

    @staticmethod
    def _validate(gdf, req_cols=None):
        '''
        Make sure the dataframe has the correct columns for the calculations
        '''

        if all([col not in gdf.columns.tolist() for col in req_cols]):
            raise AttributeError("Required columns are missing")


    def add_berg(self):
        """
        Add icebergs to the dataframe, where each row is one iceberg
        """

        print('not yet implemented')
    
    
    # Create a generalized version of itertuples that will do the row-wise calculations (if possible)
    # Also, add the possiblity herein (or in each function) to check for existing values and update (with boolean trigger)
    # def row_calc(self, calc, update=False):
    #     """
    #     Iterate over all rows of the dataframe and complete the input calculation
    #     (does not check for valid input columns)

    #     Parameters
    #     ----------
    #     update : boolean, default False
    #         update all values or only fill in empty ones
    #     """

    #     for datarow in self.itertuples(index=True, name='Pandas'):



    def calc_filt_draft(self):
        """
        Calculate an array of iceberg draft values from input freeboards.
        Applies a sea level adjustment and a 3-median_absolute_deviation filter to values.
        Values are stored as an array, but do not retain their spatial (raster) form because Pandas cells cannot hold dim>1 arrays.
        """

        req_cols = ['fjord', 'DEMarray', 'sl_adjust']
        self._validate(self._gdf, req_cols)
        
        try: self._gdf['filtered_draft']
        except KeyError:
            self._gdf['filtered_draft'] = np.ndarray

        for datarow in self._gdf.itertuples(index=True, name='Pandas'):
            rho_sw, rho_sw_err = fjord.get_sw_dens(datarow.fjord)
            H = icalcs.H_fr_frbd(datarow.DEMarray, rho_sw, self.rho_i)
            dft = icalcs.draft_fr_H(H, datarow.DEMarray)
            dft = icalcs.apply_decrease_offset(dft, datarow.sl_adjust)
            self._gdf.at[datarow.Index,'filtered_draft'] = icalcs.filter_vals(dft, num_mad=3)

  
    def est_wat_depths(self):
        """
        Estimate water depths based on filtered raster of iceberg pixel drafts
        """

        req_cols = ['filtered_draft']
        self._validate(self._gdf, req_cols)
        
        for key in ['depth_med','depth_max','depth_mad']:
            try: self._gdf[key]
            except KeyError:
                self._gdf[key] = float

        for datarow in self._gdf.itertuples(index=True, name='Pandas'):
            self._gdf.at[datarow.Index,'depth_med'] = np.nanmedian(datarow.filtered_draft)
            self._gdf.at[datarow.Index,'depth_max'] = np.nanmax(datarow.filtered_draft)
            self._gdf.at[datarow.Index,'depth_mad'] = stats.median_absolute_deviation(datarow.filtered_draft)

    def get_meas_wat_depth(self, source):
        """
        Get water depths where measurements are available
        """

        print('not yet implemented')


    


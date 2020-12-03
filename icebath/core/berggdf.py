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


@pd.api.extensions.register_dataframe_accessor("berggdf")
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
            # re-adjust to local 0msl reference frame
            dft = icalcs.apply_decrease_offset(dft, datarow.sl_adjust)
            self._gdf.at[datarow.Index,'filtered_draft'] = icalcs.filter_vals(dft, num_mad=3)

  
    def calc_rowwise_medmaxmad(self, column=''):
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

        for key in [column+'_med', column+'_max', column+'_mad']:
            try: self._gdf[key]
            except KeyError:
                self._gdf[key] = float

        for datarow in self._gdf.itertuples(index=True, name='Pandas'):
            indata = datarow[self._gdf.columns.get_loc(column)+1]  #needs the +1 because an index column is added
            self._gdf.at[datarow.Index, column+'_med'] = np.nanmedian(indata)
            self._gdf.at[datarow.Index,column+'_max'] = np.nanmax(indata)
            self._gdf.at[datarow.Index,column+'_mad'] = stats.median_absolute_deviation(indata, nan_policy='omit')


    def wat_depth_uncert(self, column=''):
        """
        Estimate the water depth uncertainty by propagating errors and uncertainties of density, etc.
        A more thorough discussion of errors related to these calculations is in Scheick et al 2019, Rem. Sens.
        """

        req_cols = [column, 'tidal_ht_min', 'tidal_ht_max']
        self._validate(self._gdf, req_cols)

        try: self._gdf[column+'_err']
        except KeyError:
            self._gdf[column+'_err'] = float

        for datarow in self._gdf.itertuples(index=True, name='Pandas'):
            rho_sw, rho_sw_err = fjord.get_sw_dens(datarow.fjord)
            rho_conversion, rho_conversion_err = icalcs.draft_uncert_dens(rho_sw, rho_sw_err, self.rho_i, self.rho_i_err)
            
            freeboard_err = max([abs(x) for x in [datarow.tidal_ht_min, datarow.tidal_ht_max]])

            med_val = datarow[self._gdf.columns.get_loc(column+'_med')+1]
            med_freebd = med_val/((rho_sw/(rho_sw-self.rho_i))-1)
            med_H =  icalcs.H_fr_frbd(med_freebd, rho_sw, self.rho_i)
            med_H_err = abs(med_H)*((freeboard_err/med_freebd)**2+(rho_conversion_err/rho_conversion)**2)**(0.5)
            err = ((med_H_err)**2+(freeboard_err)**2)**(0.5)
            self._gdf.at[datarow.Index, column+'_err'] = err
   


    def get_meas_wat_depth(self, source):
        """
        Get water depths where measurements are available
        """

        print('not yet implemented')


    


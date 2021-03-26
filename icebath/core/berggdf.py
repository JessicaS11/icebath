import pandas as pd
import numpy as np
import scipy.io as spio
import scipy.stats as stats
import ogr
import os
import fnmatch
from geocube.api.core import make_geocube
import geopandas as gpd
import rioxarray
from rioxarray.rioxarray import NoDataInBounds
import xarray as xr

from icebath.core import fjord_props as fjord
from icebath.core import fl_ice_calcs as icalcs
from icebath.core import bergxr
from icebath.core import build_xrds
from icebath.core import berggdf as bgdf


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

def get_needed_extent(src_fl, bounds):
    exbound = [0]*len(bounds)
    with rioxarray.open_rasterio(src_fl) as src:
        exbound[0] = int(getexval(src.x, 'x', bounds[0])) # later processing can't handle floats
        exbound[2] = int(getexval(src.x, 'x', bounds[2]))
        exbound[1] = int(getexval(src.y, 'y', bounds[1]))
        exbound[3] = int(getexval(src.y, 'y', bounds[3]))
    return tuple(exbound)

def getexval(potvals, coord, val):
    idx = (np.abs(potvals - val)).argmin()
    nearval = potvals.isel({coord: idx}).item()
    return nearval

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
            rho_sw, _ = fjord.get_sw_dens(datarow.fjord)
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
        
        # set type for column (since default is now object)
        for key in [column+'_med', column+'_max', column+'_mad']:
            self._gdf[str(key)] = self._gdf[str(key)].astype('float64')


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
        
        self._gdf[column+'_err'] = self._gdf[column+'_err'].astype(float)
   

    # ToDo: generalize this function to be for any input geometry and raster (with matching CRS)
    # May be able to remove this function...
    @staticmethod
    def get_px_vals(datarow, geom_name, raster, crs=None):
        '''
        Extract pixel values where the input geometry overlaps a raster. It speeds the process up by first
        subsetting the raster to the geometry bounds before extracting each overlapping pixel.
        Currently, it's assumed that the CRS of all inputs match.
        The input geometry is assumed to be a row of a GeoDataFrame, with geometry in column "geom_name"
        
        Parameters:
            geom_name : string
                        string name of the geometry column (cannot use built-in 'geometry') because `apply()`
                        turns it into a non-geo dataseries - currently noted as a bug with rioxarray (2021-01-11)
            raster : rioDataArray
        '''
        
        # Bug (rioxarray): rasterio.rio.set_crs and clip box work, but just using raster.rio.clip 
        # with a specified crs (and bounds as the polygon) gives a crs error. Huh?
        # Rioxarray improvement: in rio.clip_box, if there's not an exact match it returns a
        # 1D raster error rather than matching to the nearest coordinate value or indicating that's the issue
        # Code update/improvement: use GeoWombat, the xarray extension that has a function xr.gw.extract(geometry)
        # https://geowombat.readthedocs.io/en/latest/extraction.html#extracting-data-with-polygon-geometry
        
        raster.rio.set_crs(crs)
        try:
            # subset_raster = raster.rio.clip_box(*datarow[geom_name].bounds)
            # subset_raster = raster.rio.clip([datarow[geom_name].bounds], crs=crs)
            bounds = datarow[geom_name].bounds
            subset_raster = raster.rio.slice_xy(*bounds)
            # below line caused "no data" issues due to negative y coords
            # subset_raster = raster.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[1],bounds[3]))
            vals = subset_raster.rio.clip([datarow[geom_name]], crs=crs).values.flatten()
            # rioxarray introduces a fill value, regardless of the input nodata setup
            vals[vals==-9999] = np.nan
        except NoDataInBounds:
            print('no data')
            print(datarow[geom_name].bounds)
            vals = np.nan
        return vals

    def get_meas_wat_depth(self, src_fl, vardict={}, nanval=None):
        """
        Get water depths where measurements are available
        
        Parameters
        ----------
        dataset : XArray dataset
                dataset containing the spatial area of interest with x and y dimensions
        src_fl : source file, string
                The full path of the measurement data source file
        vardict : variable mapping, dictionary
                Key-value pairs mapping the source dataset keys to their new variable names in the dataset
        """

        # assert type(dataset)==xr.core.dataset.Dataset, "You must input an Xarray dataset from which to get measured values"
        assert vardict != {}, "You must specify your origin variables and their dataset names"

        if type(src_fl) != list:
            src_fl = [src_fl]
        
        exbounds = bgdf.get_needed_extent(src_fl[0], self._gdf.total_bounds)
        measds = build_xrds.read_netcdfs(src_fl, exbounds).chunk({'x':2048, 'y':2048})
        measds = measds.squeeze(drop=True)
        
        self._gdf['bergkey'] = self._gdf.index.astype(int)
        try:
            self._gdf["geometry"] = self._gdf.berg_poly
        except AttributeError:
            pass
        from functools import partial
        from geocube.rasterize import rasterize_image
        gdf_grid = make_geocube(vector_data=self._gdf,
                            measurements=["bergkey"],
                            like=measds,
                            fill=np.nan,
                            rasterize_function=partial(rasterize_image, filter_nan=True, all_touched=True)
                            )
        measds["bergkey"] = gdf_grid["bergkey"]
        del gdf_grid
       
        for bkey in self._gdf['bergkey']:
            bergds = measds.where(measds['bergkey']==bkey, drop=True)
            for key in vardict.keys():
                vals = bergds[key].values
                valmed = np.nanmedian(vals)
                # if key == "bed":
                #     print(vals)
                #     print(valmed)
                self._gdf.at[self._gdf[self._gdf["bergkey"]==bkey].index[0], vardict[key]] = valmed
            if bkey%10 == 0:
                print("On berg " + str(bkey))

        # return measds

        """
        Notes on computing a nanmedian with dask:
        using .median() in just about any form resulted in a not-implemented error
        (even when axis arguments were passed), despite the existance of median methods
        for both dask arrays and groupby objects.
        In my exploration, I ran the geocube example: https://corteva.github.io/geocube/stable/examples/grid_to_vector_map.html
        which promptly fails if you chunk the xarray dataset (i.e. use a dask array).
        Stacking dimensions, or manually specifying the stacked dimension created by the groupby,
        seemed to have no effect.
        Turns out there are a few related issues, too: https://nbviewer.jupyter.org/gist/rabernat/30e7b747f0e3583b5b776e4093266114
        Many of them are still open...
        """

        """
        Notes on other ways to get this info/iterate through icebergs
        Specifically: GROUPBY IS VERY SLOW AND DOESN'T STAY LAZY
        Some relevant issues/posts:
        https://github.com/pydata/xarray/issues/2852
        https://github.com/pydata/xarray/issues/659
        Though sometimes the issue may be lazy loading of netCDF files...
        http://xarray.pydata.org/en/stable/io.html#netcdf
        Noting this here as a future dev opportunity/space...
        """


        
        
        
        
        # # ToDo: add check to see if the layers are already there...
        # # Note: assumes compatible CRS systems
        # for key in vardict.keys():
        #     dataset.bergxr.get_new_var_from_file(req_dim=['x','y'], 
        #                                          newfile=src_fl, 
        #                                          variable=key, 
        #                                          varname=vardict[key])
        #     if nanval != None:
        #         dataset[vardict[key]] = dataset[vardict[key]].where(dataset[vardict[key]] != nanval)      
        
        # # Note: rioxarray does not carry crs info from the dataset to individual variables



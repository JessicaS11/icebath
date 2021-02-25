import geopandas as gpd
import pandas as pd
import numpy as np
# import scipy.io as spio
# import scipy.stats as stats
# import ogr
# import os
# import fnmatch
import pyproj
import rasterio.features
from rasterio.vrt import WarpedVRT
import rioxarray
from shapely.geometry import box
from shapely.geometry import Polygon as shpPolygon
from shapely.ops import unary_union
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


    def _calc_allpoints(self, function, req_dim=None, req_vars=None):
        """
        Helper function to do a pixel-wise calculation that requires using x and y dimension values
        as inputs. This version does the computation over all available timesteps as well.

        Point-based iteration based on example code by Ryan Abernathy from:
        https://gist.github.com/rabernat/bc4c6990eb20942246ce967e6c9c3dbe
        """

        # note: the below code will need to be generalized for using this function outside of to_geoid
        self._validate(self, req_dim, req_vars)

        def _time_wrapper(gb):
            gb = gb.groupby('dtime', squeeze=False).apply(function)
            return gb
        
        # stack x and y into a single dimension called allpoints
        stacked = self._xrds.stack(allpoints=['x','y'])
        # groupby time and apply the function over allpoints to calculate the trend at each point
        newelev = stacked.groupby('allpoints', squeeze=False).apply(_time_wrapper)
        # unstack back to x y coordinates
        self._xrds = newelev.unstack('allpoints')

        return self._xrds


    def get_mask(self, req_dim=['x','y'], req_vars=None, 
                    name=None,
                    shpfile='/home/jovyan/icebath/notebooks/supporting_docs/Land_region.shp'):
        """
        Get a shapefile of land (or area of interest) boundaries and add to the dataset
        as a mask layer that matches the extent and x/y coordinates.
        """
        self._validate(self, req_dim)

        #read in shapefile
        shpfl = gpd.read_file(shpfile)

        #confirm and correct projection if needed
        shpfl = shpfl.to_crs(self._xrds.attrs['crs'])

        # apply buffer since mask isn't exact
        shpfl['geometry'] = shpfl.buffer(5)

        mask = rasterio.features.geometry_mask(shpfl.geometry,
                                         out_shape = (len(self._xrds.y), len(self._xrds.x)),
                                         transform= self._xrds.attrs['transform'],
                                         invert=False)        

        # check for coordinate monotony. If true, then flip along the appropriate x/y coordinates before putting into xarray dataset
        flipax=[]
        if pd.Series(self._xrds.x).is_monotonic_decreasing:
            flipax.append(1)
        if pd.Series(self._xrds.y).is_monotonic_increasing:
            flipax.append(0)

        mask = xr.DataArray(np.flip(mask, axis=flipax), coords={'y':self._xrds.y, 'x':self._xrds.x}, dims=['y','x'])
        self._xrds.coords[name] = mask
        
        # clip original shapefile to XArray extent plus a half-pixel buffer
        # clipped_shpfl = gpd.clip(shpfl, box(self._xrds.x.min().item()-0.5*self._xrds.attrs['res'][0],
        #                                     self._xrds.y.min().item()-0.5*self._xrds.attrs['res'][1], 
        #                                     self._xrds.x.max().item()+0.5*self._xrds.attrs['res'][0], 
        #                                     self._xrds.y.max().item()+0.5*self._xrds.attrs['res'][1]))
        # self._xrds.attrs[name] = unary_union(clipped_shpfl.geometry) #[shpfl.geometry.exterior[row_id].coords for row_id in range(shpfl.shape[0])])



    def get_new_var_from_file(self, req_dim=['x','y'], newfile=None, variable=None, varname=None):
        """
        Get info from another dataset (NetCDF) and resample it and add it to the dataset.
        Note: this requires you have a local copy of the NetCDF you are using.
        Note: this also assumes reconciled crs between the existing and input variables.
        """

        self._validate(self, req_dim)

        print("Note that the new file is reprojected to have the same CRS as the dataset to which it is being added.\
        However, if the two CRSs are compatible, the spatial properties of the new file may be added to or overwrite the ones of the existing dataset")

        # add check for existing file?
        assert newfile != None, "You must provide an input file of the dataset to add."
        assert variable != None, "You must specify which variable you'd like to add"
        
        # read in a netcdf as a virtual raster; will need a different rasterio.open for other file types
        # read this in as a virtual raster
        with rasterio.open('netcdf:'+newfile+':'+variable) as src:
            # print('Source CRS:' +str(src.crs))
            with WarpedVRT(src,resampling=1,src_crs=src.crs,crs=self._xrds.attrs['crs']) as vrt:
                            # warp_mem_limit=12000,warp_extras={'NUM_THREADS':2}) as vrt:
                # print('Destination CRS:' +str(vrt.crs))
                newdset = rioxarray.open_rasterio(vrt).chunk({'x': 1000, 'y': 1000})
                # ds = rioxarray.open_rasterio(vrt).chunk({'x':1500,'y':1500,'band':1}).to_dataset(name='HLS_Red')


        # in an ideal world, we'd read this in chunked with dask. however, this means (in the case of Pangeo) that the file
        # needs to be in cloud storage, since the cluster can't access your home directory
        # https://pangeo.io/cloud.html#cloud-object-storage
        # with rioxarray.open_rasterio(newfile) as newdset: #, chunks={'x': 500, 'y': 500}) as newdset:
                try: newdset=newdset.squeeze().drop('band')
                except ValueError: pass
    #         newdset = xr.open_dataset(newfile, chunks={'x': 500, 'y': 500})
            # Improvement: actually check CRS matching
            # apply the existing chunking to the new dataset
                # newdset = newdset.rio.reproject(dst_crs=self._xrds.attrs['crs']).chunk({'x': 1000, 'y': 1000})
                # newvar = newdset[variable].interp(x=self._xrds['x'], y=self._xrds['y']).chunk({key:self._xrds.chunks[key] for key in req_dim})
                
                newvar = newdset.interp(x=self._xrds['x'], y=self._xrds['y']).chunk({key:self._xrds.chunks[key] for key in req_dim})

                self._xrds[varname] = newvar
                del newvar


        

    def to_geoid(self, req_dim=['dtime','x','y'], req_vars={'elevation':['x','y','dtime','geoid']},
                 source=None):
        """
        Get geoid layer from BedMachine (you must have the NetCDF stored locally; filename is hardcoded in)
        and apply to all elevation values.
        Adds 'geoid_offset' keyword to "offsets" attribute
        """

        try:
            values = (self._xrds.attrs['offset_names'])
            assert 'geoid_offset' not in values, "You've already applied the geoid offset!"
            values = list([values])+ ['geoid_offset']
        except KeyError:
            values = ['geoid_offset']

        self._validate(self, req_dim, req_vars)

        self.get_new_var_from_file(newfile=source,
                              variable='geoid', varname='geoid')
        

        self._xrds['elevation'] = self._xrds.elevation - self._xrds.geoid

        self._xrds.attrs['offset_names'] = values

        return self._xrds


    def to_geoid_pixel(self, req_dim=['dtime','x','y'], req_vars={'elevation':['x','y','dtime']}, geoid=None):
        """
        Change the elevation values to be relative to the geoid rather than the ellipsoid
        (as ArcticDEM data comes) by iterating over each pixel (over time).
        Gets a keyword added to the "offsets" attribute

        Note: CRS codes are hard-coded in for EPSG:3413 (NSIDC Polar Stereo) and EPSG:3855 (EGM08 Geoid)
        """

        try:
            values = (self._xrds.attrs['offset_names'])
            assert 'geoid_offset' not in values, "You've already applied the geoid offset!"
            values = list([values])+ ['geoid_offset']
        except KeyError:
            values = ['geoid_offset']

        self._validate(self, req_dim, req_vars)
            
        # self._xrds['elevation_orig'] = self._xrds['elevation']

        self._calc_allpoints(self._to_geoid_wrapper) #don't supply req_dim and req_vars since same as submitted to this fn

        self._xrds.attrs['crs'] = pyproj.Proj("EPSG:3413+3855")
        
        self._xrds.attrs['offset_names'] = values

        return self._xrds


    def _to_geoid_wrapper(self, gb):
        """
        XArray wrapper for the raster_ops.crs2crs function to be able to use it with
        `.groupby().apply()` to get geoid heights. It also checks that the x and y values
        are not affected by computing the geoid offset.
        
        Parameters
        ----------
        gb : groupby object
            Must contain the fields ...
        """  
        # print(gb)
        x=gb.allpoints.x.values
        y=gb.allpoints.y.values
        z=gb.elevation.values[0]
        nx, ny, nz = raster_ops.crs2crs(pyproj.Proj("EPSG:3413"), pyproj.Proj("EPSG:3413+3855"), x, y, z)
        
        assert np.isclose(x, nx)==True, "x values have changed a bunch"
        assert np.isclose(y, ny)==True, "y values have changed a bunch"
        gb = gb.assign(elevation=('dtime', nz))
        
        return gb


    def tidal_corr(self, req_dim=['dtime'], req_vars={'elevation':['x','y','dtime']},
                        loc=None, **kwargs):
        """
        Gets tidal predictions for the image date/time in the fjord of interest,
        then applies the tidal correction to the elevation field. The dataset
        gets a keyword added to the "offsets" attribute, and time dependent variables
        for the tidal offset, tidal max, and tidal min are added. If you want to model
        tides and see output plots, see fl_ice_calcs.predict_tides.
        """

        print("Note that tide model and epsg are hard coded in!")
        print("They can also be provided as keywords if the wrapper function is updated to handle them")

        try:
            values = (self._xrds.attrs['offset_names'])
            assert 'tidal_corr' not in values, "You've already applied a tidal correction!"
            values = list(values)+ ['tidal_corr']
        except KeyError:
            values = ['tidal_corr']
        
        self._validate(self, req_dim, req_vars)
        
        self._xrds = self._xrds.groupby('dtime', squeeze=False).apply(self._tidal_corr_wrapper, args=(loc), **kwargs)

        self._xrds.attrs['offset_names'] = values

        return self._xrds     


    def _tidal_corr_wrapper(self, gb, loc, **kwargs):
        """
        XArray wrapper for the fl_ice_calcs.predict_tides function to be able to use it with
        `.groupby().apply()` to get and apply tidal corrections
        
        Parameters
        ----------
        gb : groupby object
            Must contain the fields ...
        """  

        try: model_path = kwargs['model_path']
        except KeyError:
            raise AssertionError("You must specify a model path")
        

        time, tidal_ht, plots = icalcs.predict_tides(loc, 
                                                    img_time=gb.dtime.values, 
                                                    model_path=model_path, 
                                                    model='AOTIM-5-2018', 
                                                    epsg=3413)
        tidx = list(time).index(np.timedelta64(12, 'h').item().total_seconds())
        vals = [tidal_ht[tidx], np.min(tidal_ht), np.max(tidal_ht)]

        gb['elevation'] = gb.elevation + vals[0]
        gb = gb.assign(tidal_corr = ('dtime', [vals[0]]), 
                        min_tidal_ht = ('dtime', [vals[1]]), 
                        max_tidal_ht = ('dtime', [vals[2]]))

        return gb
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import hvplot.xarray
import holoviews as hv
hv.extension('bokeh','matplotlib')
from holoviews import dim, opts
import datetime as dt
import os
import panel as pn
pn.extension()
import pyproj

import icebath as icebath
from icebath.core import build_xrds
from icebath.utils import raster_ops as raster_ops
from icebath.utils import vector_ops as vector_ops
from icebath.core import fl_ice_calcs as icalcs
from icebath.core import build_gdf

import faulthandler

ds = build_xrds.xrds_from_dir('/Users/jessica/projects/bathymetry_from_bergs/DEMs/50m/')

ds.bergxr.get_mask(req_dim=['x','y'], req_vars=None, name='land_mask', shpfile='/Users/jessica/projects/bathymetry_from_bergs/geospatial_layers/testing/Land_region.shp')
# ds.land_mask.plot()
ds['elevation'] = ds['elevation'].where(ds.land_mask == True)

ds=ds.bergxr.tidal_corr(loc=["JI"], model_path='/Users/jessica/computing/tidal_model_files')

threshold=10
ds=ds.bergxr.get_icebergs(threshold=threshold) #650
ds

faulthandler.enable()

gdf = build_gdf.xarray_to_gdf(ds)
gdf.groupby('date').berg_poly.plot()
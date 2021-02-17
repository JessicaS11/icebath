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
import rioxarray

import icebath as icebath
from icebath.core import build_xrds
from icebath.utils import raster_ops as raster_ops
from icebath.utils import vector_ops as vector_ops
from icebath.core import fl_ice_calcs as icalcs
from icebath.core import build_gdf

import faulthandler
faulthandler.enable()


# indir = '/Users/jessica/projects/bathymetry_from_bergs/DEMs/50m/'
# fjord = 'JI'

indir = '/Users/jessica/projects/bathymetry_from_bergs/DEMs/Kane/'
fjord="KB"
metastr="_meta"

outdir = "/Users/jessica/projects/bathymetry_from_bergs/prelim_results/"
print("note: still need to give it an output package name that's unique and test the gpkg export")

def run_workflow(indir, outdir):
    ds = build_xrds.xrds_from_dir(indir)

    ds.bergxr.get_mask(req_dim=['x','y'], req_vars=None, name='land_mask', shpfile='/Users/jessica/projects/bathymetry_from_bergs/geospatial_layers/testing/Land_region.shp')
    ds['elevation'] = ds['elevation'].where(ds.land_mask == True)

    ds = ds.bergxr.to_geoid(source='/Users/jessica/mapping/datasets/160281892/BedMachineGreenland-2017-09-20_3413_'+ds.attrs['fjord']+'.nc')

    model_path='/Users/jessica/computing/tidal_model_files'
    ds=ds.bergxr.tidal_corr(loc=[ds.attrs["fjord"]], model_path=model_path)

    # faulthandler.enable()

    gdf = build_gdf.xarray_to_gdf(ds)
    gdf.groupby('date').berg_poly.plot()

    gdf.berggdf.calc_filt_draft()
    gdf.berggdf.calc_rowwise_medmaxmad('filtered_draft')
    gdf.berggdf.wat_depth_uncert('filtered_draft')

    measfile='/Users/jessica/mapping/datasets/160281892/BedMachineGreenland-2017-09-20_3413_'+ds.attrs['fjord']+'.nc'
    gdf.berggdf.get_meas_wat_depth(ds, measfile, 
                               vardict={"bed":"bmach_bed", "errbed":"bmach_errbed", "source":"bmach_source"},
                               nanval=-9999)

    measfile2a='/Users/jessica/mapping/datasets/IBCAO_v4_200m_ice_3413_'+ds.attrs['fjord']+'.nc'
    measfile2b='/Users/jessica/mapping/datasets/IBCAO_v4_200m_TID_3413.nc'
    gdf.berggdf.get_meas_wat_depth(ds, measfile2a, 
                               vardict={"z":"ibcao_bed"}) # no associated uncertainties
    gdf.berggdf.get_meas_wat_depth(ds, measfile2b, 
                               vardict={"z":"ibcao_source"})

    shpgdf = gdf.copy(deep=True)
    shpgdf.to_file(outdir+"icebergs.gpkg", driver="GPKG")
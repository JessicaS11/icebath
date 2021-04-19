from icebath.core import build_xrds
from icebath.core import build_gdf
import os

# import faulthandler
# faulthandler.enable()

def run_workflow(indir, fjord, outdir, outfn, metastr=None, bitmask=False):
    ds = build_xrds.xrds_from_dir(indir, fjord, metastr, bitmask)

    if ds is "nodems":
        # print("no dems to use")
        return
    
    else:
        ds.bergxr.get_mask(req_dim=['x','y'], req_vars=None, name='land_mask', shpfile='/Users/jessica/mapping/shpfiles/Greenland/Land_region/Land_region.shp')
        ds['elevation'] = ds['elevation'].where(ds.land_mask == True)

        ds = ds.bergxr.to_geoid(source='/Users/jessica/mapping/datasets/160281892/BedMachineGreenland-2017-09-20_3413_'+ds.attrs['fjord']+'.nc')

        model_path='/Users/jessica/computing/tidal_model_files'
        ds=ds.bergxr.tidal_corr(loc=[ds.attrs["fjord"]], model_path=model_path)

        # try removing some of xarray variables (even though they should only be lazily loaded?)
        # to free up memory for the next steps
        ds = ds.drop('geoid')

        # print("Going to start getting icebergs")
        gdf = build_gdf.xarray_to_gdf(ds)

        fjord = ds.attrs["fjord"]
        try:
            del ds
        except NameError:
            pass
        # print(gdf)

        if len(gdf) == 0:
            pass
        else:
        
            gdf.groupby('date').berg_poly.plot()
            gdf.berggdf.calc_filt_draft()
            gdf.berggdf.calc_rowwise_medmaxmad('filtered_draft')
            gdf.berggdf.wat_depth_uncert('filtered_draft')

            measfile='/Users/jessica/mapping/datasets/160281892/BedMachineGreenland-2017-09-20_3413_'+fjord+'.nc'
            measfile2a='/Users/jessica/mapping/datasets/IBCAO_v4_200m_ice_3413.nc'
            measfile2b='/Users/jessica/mapping/datasets/IBCAO_v4_200m_TID_3413.nc'
            
            gdf.berggdf.get_meas_wat_depth([measfile, measfile2a, measfile2b], 
                               vardict={"bed":"bmach_bed", "errbed":"bmach_errbed", "source":"bmach_source",
                                       "ibcao_bathy":"ibcao_bed", "z":"ibcao_source"},
                               nanval=-9999)
            
            shpgdf = gdf.copy(deep=True)
            del shpgdf['DEMarray']
            del shpgdf['filtered_draft']
            
            if os._exists(outdir+outfn):
                shpgdf.to_file(outdir+outfn, driver="GPKG", mode="a")
            else:
                shpgdf.to_file(outdir+outfn, driver="GPKG")
        
        try:
            del gdf
        except NameError:
            pass
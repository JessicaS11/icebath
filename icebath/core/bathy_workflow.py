from icebath.core import build_xrds
from icebath.core import build_gdf

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

        # print("Going to start getting icebergs")
        gdf = build_gdf.xarray_to_gdf(ds)
        gdf.groupby('date').berg_poly.plot()

        # print(gdf)

        if len(gdf) == 0:
            pass
        else:
        
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
            del shpgdf['DEMarray']
            del shpgdf['filtered_draft']
            shpgdf.to_file(outdir+outfn, driver="GPKG", mode="a")
        
        try:
            del ds
            del gdf
        except NameError:
            pass
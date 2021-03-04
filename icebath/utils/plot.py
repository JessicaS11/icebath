import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy
import scipy.optimize as opt
import scipy.stats as stats
# import statsmodels.stats.api as sms


def scat_lin_fit_sl1(indep, dep):
    """
    Fit a line to point data with a forced slope of one.

    Parameters
    ----------
    indep : array or list (?)
            independent (x-axis) data
    dep : array or list (?)
          dependent (y-axis) data corresponding to indep
    """

    #find intercept given a fixed slope of 1
    slope=1
    intercept = np.mean(dep-slope*indep)

    #extract info needed to plot fit line
    linx = np.linspace(min(indep)-200, max(indep)+400, 50)
    liny = [slope*xx + intercept for xx in linx]
        
    fit_y = [slope*xx + intercept for xx in indep]

    #code from stackoverflow for confidence intervals here!
    #I used this code: https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
    
    # I tried updating this from my old code to use stats.t.interval (or find another library to use), but:
    # 1. seaborn lets you plot it, but won't give you the underlying statistical info
    # 2. `stats.t.interval(0.95, len(fit_y)-1, loc=np.mean(fit_y), scale=stats.sem(fit_y))` returned single bounds,
    #  and as of 15 Jan 2021 there's no docs page for stats.t.interval so I wasn't entirely sure what it was doing anyway

    # Statistics
    n = dep.size                           # number of observations
    m = 2 #fit.size                                    # number of parameters
    dof = n - m                                    # degrees of freedom
    t = stats.t.ppf(0.95, dof)                  # used for CI and PI bands
    
    # Estimates of Error in Data/Model
    resid = dep - fit_y                           
    # chi2 = np.sum((resid/fit_y)**2)             # chi-squared; estimates error in data
    # chi2_red = chi2/(dof)                          # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2)/(dof))        # standard deviation of the error
    conf_int = t*s_err*np.sqrt(1/n + (linx-np.mean(indep))**2/np.sum((linx-np.mean(indep))**2))
    lowy = liny-conf_int
    upy = liny+conf_int

    #calculate RMSE
    rmse=np.sqrt(((fit_y-dep)**2).mean())
    
    #calculate mean absolute error
    # mae=(np.absolute(fit_y-dep)).mean()

    return linx, liny, lowy, upy, rmse, intercept


def makeplot(plotinfo, ax):
    """
    Given information about the plot, create it on the given axis.

    Parameters
    ----------
    plotinfo: dict
        Information needed to construct the plot, including the input data and plot formatting kwargs
    ax: matplotlib axes object
        Axis on which to put the plot
    """
    
    if plotinfo['plottype'] == 'scatter':
        ax.errorbar(plotinfo['x'], plotinfo['y'], **plotinfo['kwargs'])
        return ax
        
    elif plotinfo['plottype'] == 'line':
        linx, liny, lowy, upy, rmsefit, interceptfit = scat_lin_fit_sl1(plotinfo['x'], plotinfo['y'])            
            
        ax.plot(linx, liny, color=plotinfo['color'])
        ax.fill_between(linx, lowy, upy, color=plotinfo['color'], alpha=0.2)
        ax.text(0.55, 0.07-0.05*plotinfo['ct'],'$RMSE=%0.2f$, $int=%0.2f$'% (rmsefit, interceptfit),
                color=plotinfo['color'], transform=ax.transAxes)
        return ax


def meas_vs_infer_fig(berg_data, save=False):
    """
    Create a figure of measured versus inferred data. On the left is a comparison of inferred values to directly measured values.
    On the right is a plot of inferred values to gridded values that have not been directly observed.
    """    
  
    # direct-measured subdataset
    low_err_data = berg_data.where((berg_data.bmach_source >= 10) & (berg_data.ibcao_source <40)).dropna(axis='index')
    # indirect-measured subdataset
    high_err_data = berg_data.where((berg_data.bmach_source<10) | (berg_data.ibcao_source >=40)).dropna(axis='index')

    # for each item plotted, a dict is created with the input data, plot type, and plot format information.
    # These are ultimately passed into `makeplot`

    # Direct measured - scatter    
    bmach_med_inf = {'x': -low_err_data['bmach_bed'],
                     'y': low_err_data['filtered_draft_med'],
                     'plottype': 'scatter',
                     'kwargs': {'xerr': -low_err_data['bmach_errbed'],
                              'yerr': low_err_data['filtered_draft_err'],
                              'fmt': 's',
                              'color':(0,.5,0), #dark green,
                              'label': 'Median, BMach'}
                    }
    
    bmach_max_inf = {'x': -low_err_data['bmach_bed'],
                     'y': low_err_data['filtered_draft_max'],
                     'plottype': 'scatter',
                     'kwargs': {'xerr': -low_err_data['bmach_errbed'],
                              'yerr': low_err_data['filtered_draft_err'],
                              'fmt': 's',
                              'color':(0.2,0.9,0.2), #light green,
                              'label': 'Maximum, BMach'}
                    }

    ibcao_med_inf = {'x': -low_err_data['ibcao_bed'],
                     'y': low_err_data['filtered_draft_med'],
                     'plottype': 'scatter',
                     'kwargs': {'fmt': 'd',
                              'color':(0,.5,0), #dark green,
                              'label': 'Median, IBCAO'}
                    }
    
    ibcao_max_inf = {'x': -low_err_data['ibcao_bed'],
                     'y': low_err_data['filtered_draft_max'],
                     'plottype': 'scatter',
                     'kwargs': {'fmt': 'd',
                              'color':(0.2,0.9,0.2), #light green,
                              'label': 'Maximum, IBCAO'}
                    }

        
    # Direct measured - fit lines    
    med_inf_fit = {'x': np.array(-low_err_data['bmach_bed'], -low_err_data['ibcao_bed']),
                     'y': np.array(low_err_data['filtered_draft_med'], low_err_data['filtered_draft_med']),
                     'plottype': 'line',
                     'color':(0,.5,0), #dark green,
                     'ct': 0
                    }

    max_inf_fit = {'x': np.array(-low_err_data['bmach_bed'], -low_err_data['ibcao_bed']),
                     'y': np.array(low_err_data['filtered_draft_max'], low_err_data['filtered_draft_max']),
                     'plottype': 'line',
                     'color':(0.2,0.9,0.2), #light green,
                     'ct': 1
                    }

    # Indirect measured - scatter    
    bmach_indir_med_inf = {'x': -high_err_data['bmach_bed'],
                     'y': high_err_data['filtered_draft_med'],
                     'plottype': 'scatter',
                     'kwargs': {'xerr': -high_err_data['bmach_errbed'],
                              'yerr': high_err_data['filtered_draft_err'],
                              'fmt': 's',
                              'color':(0,.5,0), #dark green,
                              'label': 'Median, BMach'}
                    }
    
    bmach_indir_max_inf = {'x': -high_err_data['bmach_bed'],
                     'y': high_err_data['filtered_draft_max'],
                     'plottype': 'scatter',
                     'kwargs': {'xerr': -high_err_data['bmach_errbed'],
                              'yerr': high_err_data['filtered_draft_err'],
                              'fmt': 's',
                              'color':(0.2,0.9,0.2), #light green,
                              'label': 'Maximum, BMach'}
                    }

    ibcao_indir_med_inf = {'x': -high_err_data['ibcao_bed'],
                     'y': high_err_data['filtered_draft_med'],
                     'plottype': 'scatter',
                     'kwargs': {'fmt': 'd',
                              'color':(0,.5,0), #dark green,
                              'label': 'Median, IBCAO'}
                    }
    
    ibcao_indir_max_inf = {'x': -high_err_data['ibcao_bed'],
                     'y': high_err_data['filtered_draft_max'],
                     'plottype': 'scatter',
                     'kwargs': {'fmt': 'd',
                              'color':(0.2,0.9,0.2), #light green,
                              'label': 'Maximum, IBCAO'}
                    }

        
    # Indirect measured - fit lines    
    med_indir_fit = {'x': np.array(-high_err_data['bmach_bed'], -high_err_data['ibcao_bed']),
                     'y': np.array(high_err_data['filtered_draft_med'], high_err_data['filtered_draft_med']),
                     'plottype': 'line',
                     'color':(0,.5,0), #dark green,
                     'ct': 0
                    }

    max_indir_fit = {'x': np.array(-high_err_data['bmach_bed'], -high_err_data['ibcao_bed']),
                     'y': np.array(high_err_data['filtered_draft_max'], high_err_data['filtered_draft_max']),
                     'plottype': 'line',
                     'color':(0.2,0.9,0.2), #light green,
                     'ct': 1
                    }
    
    
    # a list of individual plots (in the dicts above) that go into each subplot (column) of the figure
    if len(low_err_data)==0 and len(high_err_data)==0:
        print("no valid data in your geodataframe")
        exit()
    elif len(low_err_data)==0:
        cols = {'Indirect Measurements':[bmach_indir_med_inf, bmach_indir_max_inf, ibcao_indir_med_inf, ibcao_indir_max_inf, med_indir_fit, max_indir_fit]
           }
    elif len(high_err_data)==0: 
        cols = {'Direct Measurements': [bmach_med_inf, bmach_max_inf, ibcao_med_inf, ibcao_max_inf, med_inf_fit, max_inf_fit],
            }
    else:  
        cols = {'Direct Measurements': [bmach_med_inf, bmach_max_inf, ibcao_med_inf, ibcao_max_inf, med_inf_fit, max_inf_fit],
                'Indirect Measurements':[bmach_indir_med_inf, bmach_indir_max_inf, ibcao_indir_med_inf, ibcao_indir_max_inf, med_indir_fit, max_indir_fit]
            }

    plot_title = 'Comparison of Gridded Products and Iceberg Freeboard-inferred Bathymetry Values'

    sz=6.5 #4.5
    nrows=1
    ncols=len(cols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*sz,nrows*sz), squeeze=False) #fig size is width, ht in inches; orig 10,5

    for i in range(0,nrows):
        
        for j, plotcol in zip(range(0, ncols), cols.keys()):
            # plot the data
            for indplot in cols[plotcol]:
                axes[i,j] = makeplot(indplot, axes[i,j])
        
            axes[i,j].set_title(plotcol, fontsize=11)
            # axes[i,j].text(0.02, 0.95,'c', weight='bold', transform=axes[i,j].transAxes)
         
            plotmin = 0 # 50 for JI
            plotmax = 350 # 700 for JI
            axes[i,j].plot([plotmin,plotmax],[plotmin, plotmax], color='k', linestyle=':')

            #modify the legend handle to not plot error bars
            handles, labels = axes[i,j].get_legend_handles_labels()  # get handles
            handles = [h[0] for h in handles]   # remove the errorbars
            axes[i,j].legend(handles, labels, loc=2) #bbox_to_anchor=(1, 1), 

            axes[i,j].set_ylim(plotmin,plotmax)
            axes[i,j].set_xlim(plotmin,plotmax)
            axes[i,j].set_aspect('equal')
    

    fig.text(0.52, 0.04,'Measured Water Depth (m)', ha='center', va='center', fontsize=12)
    fig.text(0.03, 0.5,'Inferred Water Depth (m)', ha='center', va='center', rotation='vertical', fontsize=12)
            
    fig.suptitle(plot_title, fontsize=14)

#     if nrows<2:
#         fig.subplots_adjust(top=0.87, bottom=0.13, wspace=0.6, left=0.08, right=0.99) #for rec plot: (top=0.85, bottom=0.15, wspace=0.6, left=0.15, right=0.95)
#     else:        
#         fig.subplots_adjust(hspace=0.3, wspace = 0.2, top=0.87)

    plt.show()

    if save==True:
        fig_name = 'measured_vs_inferred_' + berg_data.fjord.iloc[0] +'.png'
        fig_save_path = '/Users/jessica/projects/bathymetry_from_bergs/prelim_results/'
        fig.savefig(fig_save_path+fig_name, format='png', dpi=800)

   
    
#     meas_src = {'bmach': {'color': 'green',
#                          'shape': 's'
#                          }
#                 'ibcao': {'color': 'blue',
#                          'shape': 'D'
#                          }
#                }
    
#     medmaxhue = {'green': {'med': (0,.5,0), #dark green
#                            'max': (0.2,0.9,0.2), #light green
#                           }
#                  'blue': {'med': (0,.5,0), #dark green
#                           'max': (0.2,0.9,0.2), #light green
#                           }
#                 }
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    

# def contour_plot():



# #     import matplotlib.pyplot as plt
# # import pandas
# # import numpy as np
# # from scipy.interpolate import griddata
# # import ogr,gdal
# # import os
# # from matplotlib.colors import LinearSegmentedColormap


# # %%
# ###universal variables and import of inferred water depths from csv
# #fjord = 'JI'
# #meas_px_size = 20

# fjord = 'UP'
# meas_px_size = 25

# filepath = '/Users/jessica/GreenlandBathymetry/StrandedBergs/' + fjord + '/'
# # %% function to get shapefile points

# # def get_points_fr_shp(file_path, file_name, depth_field):

# #     #open shapefile and iterate through features
# #     driver = ogr.GetDriverByName('ESRI Shapefile')
# #     ds = driver.Open(file_path+file_name, 0) #0 is read only; 1 is writeable
# #     layer = ds.GetLayer()
# # #    layerDefn = layer.GetLayerDefn()
# #     featureCount = layer.GetFeatureCount()
    
# #     print "Number of features in %s: %d" % (os.path.basename(file_name),featureCount)
    
# #     x = np.empty(featureCount)
# #     y = np.empty(featureCount)
# #     d = np.empty(featureCount) 
# #     i = 0;
    
# #actually get x and y coordinates for it here...
# #     for feature in layer:
# #         geom=feature.GetGeometryRef()
# # #        print geom
# #         x[i] = geom.GetPoint()[0]
# #         y[i] = geom.GetPoint()[1]

        
# #         if depth_field=='land':
# #             d[i] = 0
        
# #         elif depth_field=='depth':
# #             dep = layer.GetLayerDefn().GetFieldDefn(2).GetName()
# #             d[i] = feature.GetField(dep)
        
# #         elif depth_field=='bed_JIarea':
# #             dep = layer.GetLayerDefn().GetFieldDefn(1).GetName()
# #             d[i] = feature.GetField(dep)

    
# #         i = i+1    
        
# #     if fjord=='UP':
# #         d = -d
    
# #     pts = pandas.DataFrame(np.column_stack([x,y,d]), columns=['east','north','depth'])
    
    
# #     return pts


# # %% Note: this may produce wacky results the next time it's run for UP, since I had to modify it to get inner rings that have no data for JI
# def get_poly_pts(file_path, file_name):

#     #open shapefile and iterate through features
#     driver = ogr.GetDriverByName('ESRI Shapefile')
#     ds = driver.Open(file_path+file_name, 0) #0 is read only; 1 is writeable
#     layer = ds.GetLayer()
# #    layerDefn = layer.GetLayerDefn()
#     featureCount = layer.GetFeatureCount()
    
#     print "Number of features in %s: %d" % (os.path.basename(file_name),featureCount)
    
#     i = 0;
#     pts = pandas.DataFrame(data=None, columns=['east','north'])
# #actually get x and y coordinates for it here...
#     for feature in layer:
#         geom=feature.GetGeometryRef()

# #        print geom
        
#         #get all the inner rings for the feature, which are regions of nodata within the data coverage polygon (outer ring)
#         ringCount=geom.GetGeometryCount()
#         print "Number of rings in feature: %d" % (ringCount)
#         for ring in geom:
# #            print ring
 
#             points = ring.GetPointCount()
#             x = np.empty(points)
#             y = np.empty(points)

#             for p in xrange(points):
#                 x[p], y[p], z = ring.GetPoint(p)
            
#             pts.set_value(i,'east',x)
#             pts.set_value(i,'north',y)

#             i = i+1    
    
#     return pts

# # %% get figure boundary points (extent)
    
# fp = '/Users/jessica/GreenlandBathymetry/RegionMaps/bathy_contour_maps/contour_rois/'
# fn = fjord + '_contour_roi.shp'
# #border_pts = get_points_fr_shp(fp, fn, 'land')

# driver = ogr.GetDriverByName('ESRI Shapefile')
# ds = driver.Open(fp+fn, 0) #0 is read only; 1 is writeable
# layer = ds.GetLayer()
# extent = layer.GetExtent()


# # # %%get land mask raster for masking grid
# # fp = '/Users/jessica/GreenlandBathymetry/RegionMaps/bathy_contour_maps/land_points/'
# # fn = fjord + '_contour_land_mask.tif'
# # ds = gdal.Open(fp+fn)     
# # ds_array = ds.GetRasterBand(1).ReadAsArray() #comes in such that land is >0 and water is 0
# # mask = np.zeros_like(ds_array)
# # mask[ds_array==0] = 1

# # ds=None
# # ds_array=None
    
   
# # %%get land boundary-forcing points

# fp = '/Users/jessica/GreenlandBathymetry/RegionMaps/bathy_contour_maps/land_points/'
# fn = fjord + '_contour_land_points.shp'
# land_pts = get_points_fr_shp(fp, fn, 'land')


# # %% get measured pts

# fp = '/Users/jessica/GreenlandBathymetry/RegionMaps/bathy_contour_maps/observations/'
# fn = fjord + '_contour_roi_obs.shp'

# meas_pts = get_points_fr_shp(fp, fn, 'depth')


# # %% get border of measured pts to show extent

# fp = '/Users/jessica/GreenlandBathymetry/RegionMaps/bathy_contour_maps/observations/'
# fn = fjord + '_contour_roi_obs_extent.shp'

# meas_extent = get_poly_pts(fp, fn)


# # %% DELETE: get raster of measured pts to show extent
# #fp = '/Users/jessica/GreenlandBathymetry/RegionMaps/bathy_contour_maps/observations/'
# #fn = fjord + '_contour_roi_obs_extent.tif'
# #ds = gdal.Open(fp+fn)     
# #ds_array = ds.GetRasterBand(1).ReadAsArray()
# #meas_extent = np.zeros_like(ds_array)
# #meas_extent[ds_array>0] = 1
# #
# #ds=None
# #ds_array=None

# # %% get BedMachine border forcing points (for no data regions)

# fp = '/Users/jessica/GreenlandBathymetry/RegionMaps/bathy_contour_maps/BedMachinev3_borders/'
# fn = fjord + '_BedMach_border_pts.shp'

# border_pts = get_points_fr_shp(fp, fn, 'bed_JIarea')
# border_pts['depth'] = -border_pts.depth



# # %%get my [inferred] points
# #rerun variable declaration at top of process_all_bergs to ensure correct variables are set

# #DEM method ones
# deminfidx = berg_data['meas_depth_med'].isnull()
# deminf_pts = berg_data.loc[deminfidx,['date','iceberg','east','north','draft_filt_med']]
# deminf_pts['depth'] = deminf_pts.draft_filt_med
# #print deminf_pts

# #remove DEM points for floating icebergs (variables declared in process_all_bergs)
# d=0
# for date in dates:
#     if float_bergs[d]:
# #        print 'found some floaters ' + str(float_bergs[d])
#         for bg in float_bergs[d]:
# #            print bg
#             deminf_pts.drop(deminf_pts[(deminf_pts.date==date) & (deminf_pts.iceberg==bg)].index, inplace=True)
            
#     d=d+1
    
# #print deminf_pts

# #w-d method ones
# wdinfidx = wdberg_df['meas_depth_med'].isnull()
# wdinf_pts = wdberg_df.loc[wdinfidx,['date','iceberg','east','north','wd_draft_med_js']]
# wdinf_pts['depth'] = wdinf_pts.wd_draft_med_js
# #print wdinf_pts

# inf_pts = deminf_pts.append(wdinf_pts)

# deminf_pts=None; wdinf_pts=None

# print inf_pts


# # %%
# ###create a contour plot (with lines and shading) of bathymetry, with locations of input points overlain
# ###Currently, this plots inferred (left), inferred only where there are measurements (center),
# ###and measured (right) values on the same colorscale

# #loc = ['Ilulissat Isfjord']
# #letters=['a','b']

# loc = ['Naajarsuit Fjord']
# letters=['c','d']

# plot_title = 'Comparison of Measured and Inferred Bathymetry Values'


# plt.close()

# fig, axes = plt.subplots(1,2, figsize=(8,5)) #8,3

# #define range of colors for plotting
# v=np.linspace(0,600,13)
# bathy_cmap = LinearSegmentedColormap.from_list('bathy_cmap', [(0.85,0.9,0.98),(0,0,.5)], 12)   #plt.cm.Blues

# #define grid
# xmin = min(extent[0:2])
# xmax = max(extent[0:2])
# xi = np.linspace(xmin, xmax, np.round(abs((xmin-xmax)/meas_px_size)))
# ymin = min(extent[2:4])
# ymax = max(extent[2:4])
# yi = np.linspace(ymax, ymin, np.round(abs((ymin-ymax)/meas_px_size)))
# xi,yi = np.meshgrid(xi,yi)

# #print xmin
# #print xmax
# #print ymin
# #print ymax


# #apply land mask to grid
# #note that the above grids have been set up so that the land mask array is north side up
# #(so 0,0 is actually the northwestern most point). Be wary of modifying the above or you might get some things in the wrong places.

# #print np.shape(xi)
# xi = xi * mask
# yi = yi * mask

# #print np.round(abs((xmin-xmax)/meas_px_size))
# #print np.round(abs((ymin-ymax)/meas_px_size))


# #first plot: measured data with land forcing, contoured
# x0 = pandas.concat([meas_pts.loc[:,'east'], land_pts.loc[:,'east'], border_pts.loc[:,'east']], axis=0, ignore_index=False)  #[meas[0],meas[1],wdmeas[0]], axis=0, ignore_index=False).astype(float)
# y0 = pandas.concat([meas_pts.loc[:,'north'], land_pts.loc[:,'north'], border_pts.loc[:,'north']], axis=0, ignore_index=False)
# z0 = pandas.concat([meas_pts.loc[:,'depth'], land_pts.loc[:,'depth'], border_pts.loc[:,'depth']], axis=0, ignore_index=False)
# zi0 = griddata((x0,y0), z0, (xi, yi), method='linear')
# #zi = griddata((meas_pts.loc[:,'east'], meas_pts.loc[:,'north']), meas_pts.loc[:,'depth'], (xi, yi), method='cubic')
# #zi0 = griddata((x0.ravel(), y0.ravel()), z0.ravel(), (xi, yi), method='nearest')


# CS = axes[0].contour(xi,yi,zi0,v,linewidths=0.3,colors=[(0.95,0.95,0.95)])
# CS = axes[0].contourf(xi,yi,zi0,v,cmap=bathy_cmap)


# #second plot:
# x2 = pandas.concat([meas_pts.loc[:,'east'], land_pts.loc[:,'east'], border_pts.loc[:,'east'], inf_pts.loc[:,'east']], axis=0, ignore_index=False)  #[meas[0],meas[1],wdmeas[0]], axis=0, ignore_index=False).astype(float)
# y2 = pandas.concat([meas_pts.loc[:,'north'], land_pts.loc[:,'north'], border_pts.loc[:,'north'], inf_pts.loc[:,'north']], axis=0, ignore_index=False)
# z2 = pandas.concat([meas_pts.loc[:,'depth'], land_pts.loc[:,'depth'], border_pts.loc[:,'depth'], inf_pts.loc[:,'depth']], axis=0, ignore_index=False)
# zi2 = griddata((x2,y2), z2, (xi, yi), method='linear')
# #zi2 = griddata((x2.ravel(), y2.ravel()), z2.ravel(), (xi, yi), method='nearest')


# CS2 = axes[1].contour(xi,yi,zi2,v,linewidths=0.3,colors=[(0.95,0.95,0.95)])
# CS2 = axes[1].contourf(xi,yi,zi2,v,cmap=bathy_cmap)

# axes[1].scatter(inf_pts.loc[:,'east'],inf_pts.loc[:,'north'],marker='o',c='k',s=5)


# # draw colorbar
# #x=plt.colorbar(ticks=v) #this is what was in the example I found, but it doesn't work (can't find mappable - I'm guessing it might be because of my multiple axes?)
# cbar = plt.colorbar(CS, cax=plt.axes([0.89, 0.1, 0.02, 0.75]), label='m below mean sea level') #these parameters are: left, bottom, width, height
# cbar.ax.invert_yaxis() 


# landcmap = LinearSegmentedColormap.from_list('land_cmap', [(0.65,0.45,0.35),(1,1,1)], 2)
# #extentcmap = LinearSegmentedColormap.from_list('observations_cmap', [(0,0,0,0),(0,0,0,0.3)], 2)


# if loc==['Ilulissat Isfjord']:
#     xmax = xmax-1000

# for n in range(0,2):
#     axes[n].imshow(mask, extent=(xmin, xmax, ymin, ymax), cmap=landcmap)
# #    axes[n].imshow(meas_extent, extent=(xmin, xmax, ymin, ymax), cmap=extentcmap, zorder=30)
# #    axes[n].scatter(meas_extent.loc[:,'east'], meas_extent.loc[:,'north'], marker='s', c='k', s=0.6)
#     for datarow in meas_extent.itertuples(index=True, name='Pandas'):
#         axes[n].plot(meas_extent.loc[datarow.Index,'east'], meas_extent.loc[datarow.Index,'north'], marker='None', linestyle='-', c='k', linewidth=1)
#     axes[n].axis('equal')
#     axes[n].set_ylim(ymin,ymax)
#     axes[n].set_xlim(xmin,xmax)



# #turn off y axis labels
# axes[1].yaxis.set_ticklabels([])
# #axes[2].yaxis.set_ticklabels([])

# #label each plot
# axes[0].set_title('measured gridpoints', fontsize=11)
# axes[0].text(0.02, 0.95, letters[0], weight='bold', transform=axes[0].transAxes, zorder=10)
# #axes[1].set_title('Bedmachine', fontsize=11)
# axes[1].set_title('measured+inferred', fontsize=11)
# axes[1].text(0.02, 0.95, letters[1], weight='bold', transform=axes[1].transAxes, zorder=10)

# fig.text(0.5, 0.03,'Easting (m)', ha='center', va='center', fontsize=12)
# fig.text(0.03, 0.5,'Northing (m)', ha='center', va='center', rotation='vertical', fontsize=12)

# plt.suptitle(loc[0]+ ' Bathymetry Contours', fontsize=14)
# fig.subplots_adjust(hspace=0.3, wspace = 0.14, top=0.87, left=0.14, right=0.87, bottom=0.1)


# plt.show()

# # # %%###export figure to png
# # fig_name = fjord+'_gridded_bathy_contours_border,linear.png'
# # fig_save_path = '/Users/jessica/Figures/GreenlandFjordBathymetry/results/'
# # fig.savefig(fig_save_path+fig_name, format='png', dpi=1200)

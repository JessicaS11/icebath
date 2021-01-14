import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy
import scipy.optimize as opt
import scipy.stats as stats


###function used to fit line to data    
def scatter_linear_fit(indep, dep, intercept):
#    if intercept is not None:
#        print 'using intercept given'
#        depths=(indep,dep)
#        slope=opt.fsolve(func, x0=1, args=depths)[0]
#             
#    else:
#        print 'fitting slope and intercept'
#        #fit eqn to data
#        fit = np.polyfit(indep, dep, 1, full=True)
#    #    print fit
#        slope = fit[0][0]
#        intercept = fit[0][1]
        
        
    #find intercept given a fixed slope of 1
    slope=1
    intercept = np.mean(dep-slope*indep)
        
    #extract info needed to plot fit line
    linx = np.linspace(min(indep)-200, max(indep)+400, 50)
    liny = [slope*xx + intercept for xx in linx]
        
    fit_y = [slope*xx + intercept for xx in indep]
        
    #code from stackoverflow for confidence intervals here!
    #I used this code: https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
    #and tried to confirm the math/formula with this: https://rpubs.com/aaronsc32/regression-confidence-prediction-intervals
    #the major question was whether or not you can create an artificial dataset (here linx and liny)
    #and then calculate the confidence intervals on that, since in both their cases they used
    #indep instead of linx in the conf_int, but this obviously results in an inconsistent size issue
    #if you want linx to extend beyond the end of the dataset...
    #this link: https://www.datascience.com/blog/learn-data-science-intro-to-data-visualization-in-matplotlib
    #looks like it does the same as what's below (at least as far as figures) but using scipy's stats library
    # Statistics
    n = dep.size                              # number of observations
    m = 2 #fit.size                                    # number of parameters
    DF = n - m                                    # degrees of freedom
    t = stats.t.ppf(0.95, n - m)                  # used for CI and PI bands
    
    # Estimates of Error in Data/Model
    resid = dep - fit_y                           
    chi2 = np.sum((resid/fit_y)**2)             # chi-squared; estimates error in data
    chi2_red = chi2/(DF)                          # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2)/(DF))        # standard deviation of the error
    conf_int = t*s_err*np.sqrt(1/n + (linx-np.mean(indep))**2/np.sum((linx-np.mean(indep))**2))
    lowy = liny-conf_int
    upy = liny+conf_int
    
    #calculate RMSE
    rmse=np.sqrt(((fit_y-dep)**2).mean())
    
    #calculate mean absolute error
    mae=(np.absolute(fit_y-dep)).mean()
         
#    print linx
#    print np.array(fit_y-dep)
#    print np.sqrt(((fit_y[0:10]-dep[0:10])**2).mean())
#    print np.sqrt(((fit_y[10:20]-dep[10:20])**2).mean())
#    print np.sqrt(((fit_y[20:30]-dep[20:30])**2).mean())
#    print np.sqrt(((fit_y[30:40]-dep[30:40])**2).mean())
#    print np.sqrt(((fit_y[40:]-dep[40:])**2).mean())
    
    #print statistics and create figures to show residuals are non-biased and normal
#    print rmse, slope, intercept, mae
    print('RMSE (slope=1, varying intercept):' + str(rmse))
    print('intercept:' + str(intercept))
    print('RMSE (slope=1, intercept=0):' + str(np.sqrt(((indep-dep)**2).mean())))
#    plt.figure(plt.gcf().number+1)
#    plt.scatter(fit_y,resid)
##    plt.hist(resid, np.arange(-150,250,25))
#    stats.probplot(resid,plot=plt)
#    plt.show()
    
    #return values needed to plot the data
    return linx, liny, lowy, upy, rmse, slope, intercept




# ###function used to create scatter plots comparing data, including a curve fit to the data
# ##(2019-03-27) from looking back at the code, the above function is what's actually used to fit curves and plot those fits;
# ##this function appears to be used solely to extract the appropriate error values and not to do any curve fitting or error analysis
# def scatter_with_equal_line(IB, indep, dep, exclude):
#     #extract data from dataframe, sort it, and remove rows with NA
#     IBsort = IB.sort_values(indep).dropna(axis=0, how='any')

#     if exclude is not None:
# #        idx = np.array(exclude)
# #        mask = (IBsort.image.values == idx[:, None, 0]) & (IBsort.berg.values == idx[:, None, 1])
# #        IBsortexcl = IBsort[mask.any(0)]
# #        IBsort = IBsort[~mask.any(0)]
#         idx = np.zeros(IB.shape) 
# #        IBsort['image'] = IBsort['image'].str.strip()
# #        IBsort['berg'] = IBsort['berg'].str.strip()
#         IB['image'] = IB['image'].str.strip()
#         IB['berg'] = IB['berg'].str.strip()
#         print(exclude)
#         for date, berg in exclude:
#             idx |= (IBsort.image == date) & (IBsort.berg == berg)
#             print(IBsort.image == date)
#             print(IBsort.berg == berg)
#             print(idx)
# #            IBsortexcl = IBsort[IBsort.image==date]

#         IBexcl = IB[idx]
#         IB = IB[~idx]
#         IBsort = IB.sort_values(indep).dropna(axis=0, how='any')
#         IBsortexcl = IBexcl.sort_values(indep).dropna(axis=0, how='any')

#         IBsortexcl = IBsort[idx]
#         IBsort = IBsort[~idx]
#         print(IBsortexcl)
#         print(IBsort)
        
#         meas_excl = IBsortexcl.loc[:,indep]
#         infer_excl = IBsortexcl.loc[:,dep]
    
#     else:
#         meas_excl = []
#         infer_excl = []
            
#     meas = IBsort.loc[:,indep]
#     infer = IBsort.loc[:,dep]
    
#     print('note errors are hard-coded in for max or filtered mean values ONLY')
#     #a few alternative ways to handle errors (that try to be more robust to variable inputs) are in versions of this function prior to 30 May 2018
#     try:
#         meas_err = IBsort.loc[:,'meas_depth_mad']
#     except KeyError:
#         meas_err = [0 for m in range(0,len(meas))]

#     if 'max' in dep:
#         print('using max error values')
#         try:
#             infer_err = pandas.Series([(IBsort.loc[i,'draft_filt_med_err0'],IBsort.loc[i,'draft_filt_med_err1']) for i in IBsort.index])
#         except KeyError:
#             try:
#                 infer_err = IBsort.loc[:,'draft_filt_med_err0']
#             except KeyError:
#                 infer_err = [0 for m in range(0,len(infer))]
# #        print infer_err
#     else:
#         try:
#             infer_err = IBsort.loc[:,'draft_err']
#         except KeyError:
#             infer_err = [0 for m in range(0,len(infer))]
            
#     #get info needed to plot a 1-1 line. This used to actually fit a curve, so if
#     #that functionality is desired again see compare_inferred-measured.py
#     slope=1
#     intercept=0

#     #extract info needed to plot fit line
#     x = [min(meas)-100, max(meas)+250]
# #    print x
#     y = [slope*xx + intercept for xx in x]
# #    print y
    

#     #calculate RMSE
#     rmse=np.sqrt(((meas-infer)**2).mean())
#     #NOTE - this is an inaccurate computation of RMSE (because it takes the measured vs inferred rather than modeled vs inferred)
#     #however, although the value is returned it is not used in plotting or data analysis, but rather is leftover old code that wasn't modified...

    
#     #return values needed to plot the data
#     return meas, meas_err, infer, infer_err, x, y, rmse, slope, meas_excl, infer_excl

def make_plot(berg_data):

    cols_dep = ['filtered_draft_med','filtered_draft_max']
    cols_indep = ['meas_depth_med','meas_depth_med']
    col_labels = ['Median and Maximum']

    plot_title = 'Comparison of Measured and Freeboard-\ninferred Bathymetry Values'
    fig_name = 'measuredmed_vs_inferred.png'


    nrows=1 #2
    ncols=len(col_labels) #3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4.5,nrows*4.5), squeeze=False) #fig size is width, ht in inches; orig 10,5

    meas=[]; meas_err=[]; infer=[]; infer_err=[]; x=[]; y=[]; rmse=[]; slp=[] #; meas_excl=[]; infer_excl=[]
    # wdmeas=[]; wdmeas_err=[]; wdinfer=[]; wdinfer_err=[]; wdx=[]; wdy=[]; wdrmse=[]; wdslp=[]; wdmeas_excl=[]; wdinfer_excl=[]


    for i in range(0,nrows):
            
        #get DEM method data for JI
        for k in range(0,len(cols_indep)):
    #        print k
            print(cols_indep[k] +' vs ' + cols_dep[k])
    #        meas[k], meas_err[k], infer[k], infer_err[k], x[k], y[k], rmse[k], slp[k], meas_excl[k], infer_excl[k] = \
            # m,me,f,fe,es,wi,r,s,mex,iex = scatter_with_equal_line(berg_data, cols_indep[k], cols_dep[k], exclude=None)
            
            #comment these lines out if using uniform error bars
            # fe1=pandas.Series(fe)[0] #fe.apply(pandas.Series)[0]
            # fe2=pandas.Series(fe)[1] #fe.apply(pandas.Series)[1]
            # fe=np.array([fe1,fe2])
            
            # meas.append(m); meas_err.append(me)
            # infer.append(f); infer_err.append(fe)
            # x.append(es); y.append(wi)
            # rmse.append(r); slp.append(s)
            #meas_excl.append(mex); infer_excl.append(iex)
            
            meas=-berg_data.meas_depth_med
            meas_err = berg_data.meas_depth_err
            infer = berg_data.filtered_draft_med
            infer_err = berg_data.filtered_draft_err

            # need to reconcile this with below, where it's using [n] to get the first "set" of data but ends up just plotting the first data point


            k=None

    # #    #get DEM method data for UP
    #     for q in range(0,len(cols_indep)):
    # #        print j
    #         print cols_indep[q] +' vs ' + cols_dep[q]
    # #        meas[k], meas_err[k], infer[k], infer_err[k], x[k], y[k], rmse[k], slp[k], meas_excl[k], infer_excl[k] = \
    #         m,me,f,fe,es,wi,r,s,mex,iex = scatter_with_equal_line(UP_berg_data, cols_indep[q], cols_dep[q], exclude=None)
            
    #         #comment these lines out if using uniform error bars
    #         fe1=fe.apply(pandas.Series)[0]
    #         fe2=fe.apply(pandas.Series)[1]
    #         fe=np.array([fe1,fe2])
            
    #         wdmeas.append(m); wdmeas_err.append(me)
    #         wdinfer.append(f); wdinfer_err.append(fe)
    #         wdx.append(es); wdy.append(wi)
    #         wdrmse.append(r); wdslp.append(s)
    #         wdmeas_excl.append(mex); wdinfer_excl.append(iex)

        # print(meas)
        # print(len(meas))
        # print(meas_err)
        # print(len(meas_err))

        #plot data on first figure
        j=0
        symbol = ['s','s']
        wdsymbol = ['D','D'] #,'s','s']
        color = [(0,.5,0),(0.2,0.9,0.2)]  #dark green, light green
    #    color = [(0,.5,0),(0.2,0.3,0.8)]  #dark green, dark blue 0-0-.4
    #    wdcolor = [(0.2,0.9,0.2)] #,(0.53,0.8,0.98)]  #light green, light blue
    #    JIcolor = [(0.53,0.8,0.98),(0.2,0.3,0.8)]  #light blue, dark blue
    #    UPcolor = [(0.2,0.9,0.2),(0,.5,0)]  #light green, dark green
        facecolor = [(0,.6,0),(0.2,0.9,0.2)]
        lbl=['median','maximum']
        fjdlbl=['II ','NJ ']
        for n in range(0,2):
            #plot data (JI DEM)
            
            axes[i,j].errorbar(meas[n],infer[n], xerr=meas_err[n], yerr=infer_err[n], fmt=symbol[n], \
                color=color[n], capsize=2, label=fjdlbl[0] + lbl[n])
    #            color=color[n], capsize=2, label=lbl[n]+'\n(%0.2f)'% rmse[n])
    #            markerfacecolor=facecolor[n], markeredgecolor=edgecolor[n], ecolor=edgecolor[n], capsize=2)

    #        axes[i,j].scatter(meas_excl[n], infer_excl[n], color='red')
            
    #         axes[i,j].errorbar(wdmeas[n],wdinfer[n], xerr=wdmeas_err[n], yerr=wdinfer_err[n], fmt=wdsymbol[n],\
    #             ecolor=color[n], markeredgecolor=color[n], markerfacecolor=facecolor[n], capsize=2, label=fjdlbl[1] + lbl[n])
    # #                markerfacecolor=facecolor[n], markeredgecolor=edgecolor[n], ecolor=edgecolor[n], capsize=2, markersize=12)
        
        

        ## don't delete
        '''
            linx, liny, lowy, upy, rmsefit, slpfit, interceptfit = scatter_linear_fit(pandas.concat([meas[n]]), pandas.concat([infer[n]]), intercept=None)
            axes[i,j].plot(linx, liny, color=color[n])
            axes[i,j].fill_between(linx, lowy, upy, color=color[n], alpha=0.2)
    #        axes[i,j].text(0.42, 0.07-0.05*n,'$RMSE=%0.2f$, $slope=%0.2f$'% (rmsefit, slpfit), color=color[n], transform=axes[i,j].transAxes)
            axes[i,j].text(0.41, 0.07-0.05*n,'$RMSE=%0.2f$, $int=%0.2f$'% (rmsefit, interceptfit), color=color[n], transform=axes[i,j].transAxes)


        linx=None; liny=None; lowy=None; upy=None; rmsefit=None; slpfit=None; interceptfit=None

        '''
        
        
    #    axes[i,j].text(.8*max(meas[n]), 0.35*min(infer[n]),'$RMSE = %0.2f$'% rmse[0], fontsize=10, color=color[0]) #orig 1.15 max
    #    axes[i,j].text(.92*min(meas[n]), 1.05*max(infer[n]),'$RMSE = %0.2f$'% rmse[1], fontsize=10, color=color[1]) #orig 1.15 median
    #    axes[i,j].text(.8*max(meas[n]), 0.25*min(infer[n]),'$RMSE = %0.2f$'% wdrmse[0], fontsize=10, color=wdcolor[0]) #orig 1.15 max
    #    axes[i,j].text(.92*min(meas[n]), .95*max(infer[n]),'$RMSE = %0.2f$'% wdrmse[1], fontsize=10, color=wdcolor[1]) #orig 1.15 median
        n=None
    #    axes[i,j].set_title(col_labels[j], fontsize=11)
    #    axes[i,j].text(0.02, 0.95,'c', weight='bold', transform=axes[i,j].transAxes)
        
        
        # ToDo: set a min value that doesn't cut off data!
        ymax=600  #620 for JI, 1050 for UP
        axes[i,j].plot([0,ymax],[0, ymax], color='k', linestyle=':')

        #modify the legend handle to not plot error bars
        handles, labels = axes[i,j].get_legend_handles_labels()  # get handles
        handles = [h[0] for h in handles]   # remove the errorbars
        axes[i,j].legend(handles, labels, loc=2) #bbox_to_anchor=(1, 1), 

        
        axes[i,j].set_ylim(90,ymax) #25, 620=ymax
        axes[i,j].set_xlim(90,ymax) #110, 425)
        axes[i,j].set_aspect('equal')

        

        
        meas=None
        infer=None
        meas_err=None
        infer_err=None
        x=None
        y=None

    fig.text(0.52, 0.04,'Measured Water Depth (m)', ha='center', va='center', fontsize=12)
    fig.text(0.03, 0.5,'Inferred Water Depth (m)', ha='center', va='center', rotation='vertical', fontsize=12)
            
    fig.suptitle(plot_title, fontsize=14)

    if nrows<2:
        fig.subplots_adjust(top=0.87, bottom=0.13, wspace=0.6, left=0.08, right=0.99) #for rec plot: (top=0.85, bottom=0.15, wspace=0.6, left=0.15, right=0.95)
    else:        
        fig.subplots_adjust(hspace=0.3, wspace = 0.2, top=0.87) #orig (before legend between plots) wspace = 0.2

    plt.show()

    # %%export figure to png
    # fig_save_path = '/Users/jessica/Figures/GreenlandFjordBathymetry/results/'
    # fig.savefig(fig_save_path+fig_name, format='png', dpi=800)
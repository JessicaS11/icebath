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

    return linx, liny, lowy, upy, rmse, slope, intercept


# ToDo: functionize this (or at least parts of it)
def meas_vs_infer_plot(berg_data):

    cols_dep = ['filtered_draft_med','filtered_draft_max']
    cols_indep = ['meas_depth_med','meas_depth_med']
    col_labels = ['Median and Maximum']

    plot_title = 'Comparison of Measured and Freeboard-\ninferred Bathymetry Values'
    # fig_name = 'measuredmed_vs_inferred.png'

    berg_data = berg_data.dropna(axis='index', subset=cols_dep+cols_indep)

    sz=6.5 #4.5
    nrows=1
    ncols=len(col_labels)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*sz,nrows*sz), squeeze=False) #fig size is width, ht in inches; orig 10,5

    meas=[]; meas_err=[]; infer=[]; infer_err=[]

    for i in range(0,nrows):
            
        #get DEM method data
        for k in range(0,len(cols_indep)):
            print(cols_indep[k] +' vs ' + cols_dep[k])
            
            meas.append(-berg_data[cols_indep[k]])
            meas_err.append(berg_data.meas_depth_err)
            infer.append(berg_data[cols_dep[k]])
            infer_err.append(berg_data.filtered_draft_err)

            k=None

        #plot data on first figure
        j=0
        symbol = ['s','s']
        # wdsymbol = ['D','D']
        color = [(0,.5,0),(0.2,0.9,0.2)]  #dark green, light green
        # facecolor = [(0,.6,0),(0.2,0.9,0.2)]
        lbl=['median','maximum']
        fjdlbl=['II ','NJ ']
        for n in range(0,2):
            #plot data (JI DEM)
            
            axes[i,j].errorbar(meas[n],infer[n], xerr=meas_err[n], yerr=infer_err[n], fmt=symbol[n], \
                color=color[n], capsize=2, label=fjdlbl[0] + lbl[n])
    #            color=color[n], capsize=2, label=lbl[n]+'\n(%0.2f)'% rmse[n])
    #            markerfacecolor=facecolor[n], markeredgecolor=edgecolor[n], ecolor=edgecolor[n], capsize=2)

            
            linx, liny, lowy, upy, rmsefit, slp, interceptfit = scat_lin_fit_sl1(pandas.concat([meas[n]]), pandas.concat([infer[n]]))            
            
            axes[i,j].plot(linx, liny, color=color[n])
            axes[i,j].fill_between(linx, lowy, upy, color=color[n], alpha=0.2)
            axes[i,j].text(0.55, 0.07-0.05*n,'$RMSE=%0.2f$, $int=%0.2f$'% (rmsefit, interceptfit), color=color[n], transform=axes[i,j].transAxes)
            # above x needs to be 0.41 with smaller figure
        linx=None; liny=None; lowy=None; upy=None; rmsefit=None; slp=None; interceptfit=None
     
        n=None
    #    axes[i,j].set_title(col_labels[j], fontsize=11)
    #    axes[i,j].text(0.02, 0.95,'c', weight='bold', transform=axes[i,j].transAxes)
         
        plotmin = 50
        plotmax = 575
        axes[i,j].plot([plotmin,plotmax],[plotmin, plotmax], color='k', linestyle=':')

        #modify the legend handle to not plot error bars
        handles, labels = axes[i,j].get_legend_handles_labels()  # get handles
        handles = [h[0] for h in handles]   # remove the errorbars
        axes[i,j].legend(handles, labels, loc=2) #bbox_to_anchor=(1, 1), 

        axes[i,j].set_ylim(plotmin,plotmax)
        axes[i,j].set_xlim(plotmin,plotmax)
        axes[i,j].set_aspect('equal')
    
        meas=None
        infer=None
        meas_err=None
        infer_err=None

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
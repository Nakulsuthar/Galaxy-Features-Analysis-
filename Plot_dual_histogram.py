import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from Initial import * 
from Stat_test_1 import * 


# FINAL 

# this file calls for the function described below and 
# it also computes w1 and w2 which are weighted bins to control for any two variables 
# it computes these values from stat_test_1 file which was provided by our coordinator 


def plot_mass_z_2(table_1, table_2,
    z_col, mass_col, z_limits,
    mass_limits, bins_z, bins_mass, 
    outfile=None):
    """
    This function plots the stellar mass redshift distributions for any two subsamples with 
    marginal histograms on its side as one plot. 

    This panel shows: 
    A scatter plot of stellar mass against redshift for any two subsets which is the main panel
    Top histogram of redshift to see the spread of redshift across the two subsets
    Right histogram of stellar mass distribution for the two subsets 

    This function is applied under the mass completeness limit

    Paramaters: 
    table_1: table 1 containing the first subset 
    table_2: table 2 containing the second subset 
    z_col, mass_col: column names to extract specific data 
    z_limits, mass_limits: mass and redshift limits applied. takes in both min and max value 
    bins_z, bins_mass: number of bins used for both histograms 
    outfile: name if provided saves the figure to the given file path. 


    """

    # applying the volumetric limit to the plot 
    z_min, z_max = z_limits
    mass_min, mass_max = mass_limits

    #masking the two subset such that the volumetric limit is applied (yes the limit is already applied in inital.py but that was done in the end when i realised how to optimise code)
    table_1_selection = ((table_1[z_col]>= z_min)&(table_1[z_col]<= z_max)&(table_1[mass_col]>= mass_min)& (table_1[mass_col]<= mass_max))
    table_2_selection = ((table_2[z_col]>= z_min)&(table_2[z_col]<= z_max)&(table_2[mass_col]>= mass_min)& (table_2[mass_col]<= mass_max))

    tab1 = table_1[table_1_selection]
    tab2 = table_2[table_2_selection]

    # creating a 2x2 figure using gridspec
    fig = plt.figure(figsize=(6.2, 6),constrained_layout=True)  
    gs = GridSpec(
    2, 2,
    width_ratios=[4, 1.2],
    height_ratios=[1.2, 4],
    hspace=0.0,
    wspace=0.0
    )
    plt.subplots_adjust(left=0.26, bottom=0.12)
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_hist_top_z  = fig.add_subplot(gs[0, 0], sharex=ax_scatter) # to ensure the axis of each histogram is matched to the axis of the scatter plot
    ax_hist_right_M  = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    ax_scatter.scatter(tab1[z_col],tab1[mass_col],marker='s',facecolors='none',edgecolors='forestgreen',alpha=0.25, s=35, linewidths=1.0, label='NO-RING-GALAXIES',zorder=3)
    ax_scatter.scatter(tab2[z_col],tab2[mass_col],marker='^',facecolors='none',edgecolors='royalblue',alpha=0.8, s=35, linewidths=1.0, label='RING-GALAXIES',zorder=2)

    _,z_bins,_ = ax_hist_top_z.hist(tab1[z_col],bins=bins_z,histtype='step',density=True,linewidth=1.8, color='royalblue')
    _,z_bins_,_ = ax_hist_top_z.hist(tab2[z_col],bins=bins_z,histtype='step',density=True,linestyle='--',linewidth=1.8, color='forestgreen')

    _,m_bins,_ = ax_hist_right_M.hist(tab1[mass_col],bins=bins_mass,histtype='step',density=True,linewidth=1.8,color='royalblue',orientation='horizontal')
    _,m_bins,_ = ax_hist_right_M.hist(tab2[mass_col],bins=bins_mass,histtype='step',density=True,linestyle='--',linewidth=1.5,color='forestgreen',orientation='horizontal')
    
    ax_scatter.set_xlim(z_min, z_max)
    ax_scatter.set_ylim(9.23, mass_max) # give some empty space at the bottom

    ax_hist_top_z.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False
)

    # formatting
    ax_hist_right_M.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False
)


    ax_scatter.set_xlabel(r"redshift $z$", fontsize=16,labelpad=10)
    ax_scatter.set_ylabel(r"$log_{10}$ Stellar Mass $M_* \; [M_\odot]$", fontsize=14,labelpad=12)

    #more formatting
    plt.setp(ax_hist_top_z.get_xticklabels(), visible=False)
    plt.setp(ax_hist_right_M.get_yticklabels(), visible=False)

    ax_hist_top_z.tick_params(axis='y', direction='in')
    ax_hist_right_M.tick_params(axis='x', direction='in')

    for ax in [ax_scatter, ax_hist_top_z, ax_hist_right_M]:
        ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True,
        labelsize=11
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax_scatter.legend(
    loc='upper left',
    frameon=True,
    fontsize=11
    )

    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        plt.close()
    else:
        plt.show()


    # important to return bin values to get weighted bins
    return {
    "z_bins": z_bins,
    "m_bins": m_bins,
}


    
# call 

sample1 = Ringed_Disc_Barred_Table
sample2 = No_ringed_Disc_Barred_Table

all_bins = plot_mass_z_2(table_1=sample1, table_2=sample2,
 z_col="redshift_UVISTA_r", mass_col="lp_mass_med",
  z_limits=(0.2,1.0), mass_limits=(9.5,12), bins_z=30, bins_mass=30, 
  outfile="Gallery/test.png")



# formatting data to find weighted bins 
# extracting all the bins first from the histogram in the function above

z_bins = all_bins["z_bins"]
m_bins = all_bins["m_bins"]
#print(z_bins.shape)         # sanity tests 

Bins_array = np.array([z_bins,m_bins])
#print(Bins_array.shape)       # sanity tests


table_1_array = np.array([
    np.array(sample1["redshift_UVISTA_r"]),
    np.array(sample1["lp_mass_med"])
])

table_2_array = np.array([
    np.array(sample2["redshift_UVISTA_r"]),
    np.array(sample2["lp_mass_med"])
])

    
# call function from stat_test.py
w1, w2, used_bins = weight_dist_dd(table_1_array,table_2_array,bins=Bins_array,return_bins=True,renorm=False)

# extract used bins and unpack them as mass and redshift bins
used_redshift_bin = used_bins[0]
used_mass_bin = used_bins[1]


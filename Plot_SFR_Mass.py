import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from Initial import * 
from Stat_test_1 import * 
from Plot_dual_histogram import *
#FINAL

def plot_mass_z_3(table_1, table_2, contour_table, 
    mass_col, sfr_col,
    mass_limits, sfr_limits, 
    bins_mass, bins_sfr, 
    outfile=None,w1=w1,w2=w2):
    """
    This function plots the stellar mass redshift distributions of any two subsamples with 
    marginal histograms on its side as one plot. 

    This panel shows: 
    A scatter plot of the stellar mass redshift dostributions for any two subsamples with 
    a 90% contour of a background table (normally the entire sample). Top histogram of redshift 
    and a right histogram of stellar mass to see the effective distributions of each subsample. 
    These histograms are applied weight such that the stellar mass distributions are matched. 

    Parameters: 
    table_1: table 1 containing the first subset 
    table_2: table 2 containing the second subset 
    contour_table: table 3 containing the background 90% contour
    mass_col, sfr_col: column names to extract specific data 
    mass_limits, sfr_limits: mass and redshift limits applied. takes in both min and max value.
    bins_mass, bins_sfr: number of bins for the marginal histograms. (normally the same)
    outfile: name if provided saves the figure to the given file path.
    w1,w2: weights for table 1 and table 2. extracted from plot_dual_histogram.py
    """
    mass_min, mass_max = mass_limits
    sfr_min, sfr_max = sfr_limits

    #convienience
    tab1 = table_1
    tab2 = table_2
    
    # set up figure
    fig = plt.figure(figsize=(5.5, 5.3))
    gs = GridSpec(
    2, 2,
    width_ratios=[4, 1.2],
    height_ratios=[1.2, 4],
    hspace=0.0,
    wspace=0.0
    )

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_hist_top_z  = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_hist_right_M  = fig.add_subplot(gs[1, 1], sharey=ax_scatter)


    # contour formatting to obtain the 90% contour of a background table
    if contour_table is not None:

        contour_sel = (
        (contour_table[mass_col] >= mass_min) &
        (contour_table[mass_col] <= mass_max) &
        (contour_table[sfr_col] >= sfr_min) &
        (contour_table[sfr_col] <= sfr_max)
        )

        contour = contour_table[contour_sel]

        z_vals = np.array(contour[mass_col])
        m_vals = np.array(contour[sfr_col])

        # 2D density grid
        H, xedges, yedges = np.histogram2d(
        z_vals,
        m_vals,
        bins=[bins_mass, bins_sfr],
        range=[[mass_min, mass_max], [sfr_min, sfr_max]]
        )

    # smooth to get natural shape
        H = gaussian_filter(H, sigma=1.5)


        H_flat = H.flatten()
        H_sorted = np.sort(H_flat)[::-1]   # descending
        cumsum = np.cumsum(H_sorted)
        cumsum /= cumsum[-1]

        level_90 = H_sorted[np.searchsorted(cumsum, 0.90)]    # 90 percent contour

        X = 0.5 * (xedges[1:] + xedges[:-1])
        Y = 0.5 * (yedges[1:] + yedges[:-1])

        ax_scatter.contour(
        X,
        Y,
        H.T,
        levels=[level_90],     
        colors='grey',
        linestyles='dotted',
        linewidths=1.6,
        zorder=1,
        )

        contour_proxy = Line2D(
        [0], [0],
        color='grey',
        linestyle='dotted',
        linewidth=2,
        label='FULL SAMPLE (90% CONTOUR)'
        )


    ax_scatter.scatter(tab2[mass_col],tab2[sfr_col],marker='o',facecolors='none',edgecolors='#3a8f3a',alpha=0.35, s=28, linewidths=1.2, label='RING-BARRED-GALAXIES',zorder=3)
    ax_scatter.scatter(tab1[mass_col],tab1[sfr_col],marker='^',facecolors='royalblue',edgecolors='white',alpha=0.75, s=28, linewidths=0.5, label='NO-RING-BARRED-GALAXIES',zorder=3)

    _,z_bins,_ = ax_hist_top_z.hist(tab1[mass_col],bins=bins_mass,histtype='step',density=True,linewidth=1.8, color='royalblue',weights=w1)
    _,z_bins_,_ = ax_hist_top_z.hist(tab2[mass_col],bins=used_mass_bin,histtype='step',density=True,linestyle='--',linewidth=1.8, color='forestgreen',weights=w2)
    ax_hist_top_z.hist(contour[mass_col],bins=bins_mass,histtype='step',density=True,linestyle=':',linewidth=1.8, color='grey')

    _,m_bins,_ = ax_hist_right_M.hist(tab1[sfr_col],bins=bins_sfr,histtype='step',density=True,linewidth=1.8,color='royalblue',orientation='horizontal',weights=w1)
    _,m_bins,_ = ax_hist_right_M.hist(tab2[sfr_col],bins=bins_sfr,histtype='step',density=True,linestyle='--',linewidth=1.5,color='forestgreen',orientation='horizontal',weights=w2)
    ax_hist_right_M.hist(contour[sfr_col],bins=bins_sfr,histtype='step',density=True,linestyle=':',linewidth=1.5,color='grey',orientation='horizontal')
    
    ax_scatter.set_xlim(mass_min, mass_max)
    ax_scatter.set_ylim(sfr_min, sfr_max)

    #formatting
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

    #MS Line from aird et al
    z_med = np.median(Initial_Table['redshift_UVISTA_r'])

    logM = Initial_Table['lp_mass_med']
    logM_range = np.linspace(logM.min(),logM.max(),len(Initial_Table)) 
    logSFR_line = (-7.6 + (0.76 * logM_range ) + 2.95 * (np.log10(1+z_med)))


    ax_scatter.plot(
    logM_range,
    logSFR_line,
    color='black',
    linestyle='--',
    linewidth=2,
    label=f'MS (z={z_med:.2f})',
    zorder=10
    )

    

    ax_scatter.set_xlabel(r'$\log Stellar Mass(M_\star\,[M_\odot])$', fontsize=14)
    ax_scatter.set_ylabel(r'$\log (\mathrm{SFR}\,[M_\odot\,\mathrm{yr}^{-1}])$', fontsize=14)

    # formatting
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

    #handling legend 
    handles, labels = ax_scatter.get_legend_handles_labels()

    seen = set()
    new_handles = []
    new_labels = []

    for h, l in zip(handles, labels):
        if l not in seen and l != "":
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)


    new_handles.append(contour_proxy)
    new_labels.append('FULL SAMPLE (90% CONTOUR)')

    ax_scatter.legend(
    new_handles,
    new_labels,
    loc='upper left',
    frameon=True,
    framealpha=0.85,
    fontsize=8
    )

    if outfile is not None:
        plt.savefig(outfile ,bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    return {
    "z_bins": z_bins,
    "m_bins": m_bins,
}


# call 

sample1 = sample1
sample2 = sample2

all_bins = plot_mass_z_3(table_1=sample1, table_2=sample2, contour_table=Initial_Table,
    mass_col="lp_mass_med", sfr_col="lp_SFR_med",
    mass_limits=(9.2,11.7), sfr_limits=(-3,5.5),
    bins_mass=30, bins_sfr=30,
    outfile="Gallery/test.png",
    w1=w1,w2=w2)

#note that in order to obtain valid bins (w1,w2) for table_1 and table_@
#one must change and ensure that sample 1 and sample 2 in both (Plot_dual_histogram.py and Plot_SFR_Mass.py) must be the same 

# if they are not the same,
# then they are not valid and it will cause an error. 

#so if someone wishes to change the input data for table_1 and table_2, they should
#change sample1 and sample2 in plot_dual_histogram.py
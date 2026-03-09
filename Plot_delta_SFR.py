
import numpy as np
import matplotlib.pyplot as plt
from Initial import * 
from Stat_test_1 import * 
from matplotlib.lines import Line2D

#FINAL

def delta_sfr(Table1, ax, colors):
    """
    Plotting function that plots a singular histogram of delta_SFR

    Paramaters:
    Table1: table 1 for which the histogram is plotted
    ax: Axis on which the histogram will be plotted 
    colors: color that will be used on the histogram

    """

    # Computing Delta SFR using the equation from aird et al
    # adding a new columns in order to use it later for plotting
    # the new columns are called 'logSFR_cut' which is the value of the main sequence line given redshift and mass
    # and 'Delta_logSFR' which is the offset value from the MS line and the actual value

    Table1['logSFR_cut'] = (
        -7.6
        + 0.76 * Table1['lp_mass_med']
        + 2.95 * np.log10(1 + Table1['redshift_UVISTA_r'])
    )

    Table1['Delta_logSFR'] = (
        Table1['lp_SFR_med']
        - Table1['logSFR_cut']
    )

    bins = np.linspace(-2, 2, 20)

    ax.hist(
        Table1['Delta_logSFR'],
        bins=bins,
        histtype='step',
        linewidth=1.8,
        color=colors,
        density=True
    )

    ax.set_title(None)

    # Region dividers to distinguish different star forming regions 
    # values of these dividers are taken from aird et al

    for x in [-1.3, -0.4, 0.4]:
        ax.axvline(x, linestyle=':', color='0.5', linewidth=1.2)

    ax.set_xlim(-2.5, 2)
    ax.set_ylim(0, 1.2)

def plot_2x2(samples):
    """
    This function creates a 2x2 panel figure showing delta log SFR distributions
    for four galaxy subsamples.

    Parameters:
    samples: a list containing four galaxy tables in order to be plotted


    """

    # 2x2 fig setup

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 8),
        sharex=True,
        sharey=True
    )

    titles = [
        "Ringed Barred",
        "Ringed Unbarred",
        "No-Ring Barred",
        "No-Ring Unbarred"
    ]
    colors = [
        "darkred",
        "royalblue",
        "darkgreen",
        "purple"
    ]

    for ax, sample, title, colors in zip(axes.flat, samples, titles, colors):
        delta_sfr(sample, ax, title,colors)

    # universal axis labels only on bottom left panels
    axes[1,0].set_xlabel(r'$\Delta \log(\mathrm{SFR})$',fontsize=18)
    axes[1,0].set_ylabel('Normalised Count',fontsize=18)

    ax00 = axes[0,0]
    
    # vertical position of text for each star forming regimes
    ypos = 0.88

    ax00.text(-1.8, ypos, 'Quiescent',
          transform=ax00.get_xaxis_transform(),
          ha='center', va='center',
          fontsize=12)

    ax00.text(-0.8, ypos, 'Sub-\nSeq',
          transform=ax00.get_xaxis_transform(),
          ha='center', va='center',
          fontsize=12)

    ax00.text(0.0, ypos, 'On-\nSeq',
          transform=ax00.get_xaxis_transform(),
          ha='center', va='center',
          fontsize=12)

    ax00.text(1.0, ypos, 'Starburst',
          transform=ax00.get_xaxis_transform(),
          ha='center', va='center',
          fontsize=12)

    # common legend 
    legend_elements = [
    Line2D([0], [0], color='darkred', lw=2, label='Ringed Barred'),
    Line2D([0], [0], color='darkgreen', lw=2, label='No-Ring Barred'),
    Line2D([0], [0], color='royalblue', lw=2, label='Ringed Unbarred'),
    Line2D([0], [0], color='purple', lw=2, label='No-Ring Unbarred')
    ]

    fig.legend(
    handles=legend_elements,
    loc='center',
    bbox_to_anchor=(0.77, 0.9),
    ncol=2,
    frameon=True,
    fontsize=12
    )

    plt.tight_layout()
    plt.savefig(
    "Gallery/2x2_final.png",
    dpi=300,
    bbox_inches='tight'
)

# samples to be plotted
samples = [
    Ringed_Disc_Barred_Table,
    Ringed_Disc_Unbarred_Table,
    No_ringed_Disc_Barred_Table,
    No_ringed_Disc_Unbarred_Table
]

# call 
plot_2x2(samples)

 





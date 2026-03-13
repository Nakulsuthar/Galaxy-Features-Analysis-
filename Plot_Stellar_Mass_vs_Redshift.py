import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import Table
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from Initial import * 

# FINAL 
def plot_mass_z_1(
    background_table, contour_table,
    z_col="redshift_UVISTA_r", mass_col="lp_mass_med",
    z_min = 0.0, z_max=2.6, dz=0.2,
    mass_min=6.0, mass_max=12.0,
    smooth_sigma=1.6,
    outfile=None):
    """
    This function plots stellar mass as a function of redshift. 

    The plot shows many things such as: 

    A smooth 2D histogram of the entire dataset within z_min, z_max and mass_min, mass_max
    A contour of all the clean disc galaxies sampled from initial.py 
    A mass complete region seen in red defined by limiting mass and redshift in the function itself. All the 
    clean disc galaxies seen in this region are seen by red points. 
    
    Parameters: 
    background_table: Table containing the full galaxy sample used to construct the 2D histogram in the background 
    contour_table: Table containing the subsample for which the contours are plotted in blue 
    z_col: column name used for redshift. ("redshift_UVISTA_r")
    mass_col: column name used for stellar mass. ("lp_mass_med")
    z_min, z_max: min and max value of redshift limits for the plot 
    dz: redshift bin width when constructing the contour histogram 
    mass_min, mass_max: min and max value of stellar mass limits for the plot 
    smooth_sigma: guassian smoothing sigma value applied to the contour histogram 
    outfile: name if provided saves the figure to the given file path.

    """

    # Extracting mass and redshift column values
    # then i am defining binning for contours and background 2d histogram
    # the background hist uses finer bins for smoother density map 
    z_all = background_table[z_col]
    m_all = background_table[mass_col] 

    z_cnt = contour_table[z_col]
    m_cnt = contour_table[mass_col]

    z_bins = np.arange(z_min, 2.6 + dz, dz)
    m_bins = np.linspace(mass_min, mass_max, 120)

    z_bins_bg = np.linspace(z_min, z_max, 90)
    m_bins_bg = np.linspace(mass_min, mass_max, 90)

    H_bg, _, _ = np.histogram2d(
        z_all, m_all,
        bins=[z_bins_bg, m_bins_bg]
    )
    H_bg = H_bg.T
    H_bg_smooth = gaussian_filter(H_bg, sigma=0.8)

    H_cnt, _, _ = np.histogram2d(
        z_cnt, m_cnt,
        bins=[z_bins, m_bins]
    )
    H_cnt = H_cnt.T  # to match the plotting orientation 
    H_cnt_smooth = gaussian_filter(H_cnt, sigma=smooth_sigma)


    fig, ax = plt.subplots(figsize=(8, 7))

    # Background density plotting code 
    ax.imshow(
        np.log10(H_bg_smooth + 1),
        origin="lower",
        extent=[z_bins_bg[0], z_bins_bg[-1], m_bins_bg[0], m_bins_bg[-1]],
        cmap="Greys",
        aspect="auto"
    )

    # Contours code 
    # computing contour levels from percentile of the smooth distribution 
    levels = np.percentile(
        H_cnt_smooth[H_cnt_smooth > 0],
        [60, 80, 95]
    )

    ax.contour(
        z_bins[:-1],
        m_bins[:-1],
        H_cnt_smooth,
        levels=levels,
        colors="navy",
        linewidths=2.2
    )

   # defining a hard redshift cut at z=1.0. 
   # defining a mass cut at 9.5 
    zmin_sel = 0.2
    zmax_sel = 1.0
    cosmos = contour_table  #convienience


    Mlim = 9.5
   # volume cut within redshift and mass limits 
    mass_complete = (
        (cosmos[z_col] >= zmin_sel) &
        (cosmos[z_col] <= zmax_sel) &
        (cosmos[mass_col] >= Mlim)
    )

    ax.scatter(
        cosmos[z_col][mass_complete],
        cosmos[mass_col][mass_complete],
        s=15,
        color="#e74c3c",
        alpha=0.60,
        linewidths=0,
        label=r"mass-complete"
    )

    ax.plot(
        [zmin_sel, zmax_sel],   # horizontal red dotted line that starts from z_min and ends at z_max 
        [Mlim, Mlim],           # and is constant at Mlim
        color="red",
        linestyle="dotted",
        linewidth=1.5
    )

    ax.plot(
        [zmin_sel, zmin_sel],  #left vertical red dotted line that starts at Mlim and ends at Mmax 
        [Mlim, mass_max],      # and is constant at z_min
        color="red",
        linestyle="dotted",
        linewidth=1.5
)

    ax.plot(
        [zmax_sel, zmax_sel],   #right vertical red dotted line that starts at Mlim and ends at Mmax 
        [Mlim, mass_max],       # and is constant at z_max
        color="red",
        linestyle="dotted",
        linewidth=1.5
    )

    #creating a legend using line2D
    legend_elements = [

    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor='#e74c3c',
        markersize=6,
        alpha=0.25,
        label=r'Mass-complete'
    ),

    Line2D(
        [0], [0],
        linestyle='dotted',
        color='#c0392b',
        linewidth=2.0,
        label=r'Sample limits'
    ),
    Line2D(
        [0], [0],
        color='navy',
        linewidth=2.2,
        label=r'Discs Galaxies'
    )
    ]

    ax.legend(
    handles=legend_elements,
    loc='lower right',
    frameon=False,
    fontsize=11
    )

    # Axes formatting
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(mass_min, mass_max)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    ax.tick_params(axis="both", which="major", length=6, direction="in")
    ax.tick_params(axis="both", which="minor", length=3, direction="in")
    ax.tick_params(top=True, right=True)

    ax.set_xlabel(r"redshift $z$", fontsize = 16)
    ax.set_ylabel(r"log Stellar Mass $M_* \,[M_\odot]$", fontsize = 16)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        plt.close()
    else:
        plt.show()



Initial_Table_total_3 = Table.read("Matched_GZH_UVISTA.fits")
background = Initial_Table_total_3


# call 

plot_mass_z_1(
    background_table=background,
    contour_table=Clean_Disc_Galaxies_Table,
    z_min = 0.0,
    z_max=2.0,
    dz=0.2,
    outfile="Gallery/volume_sample.png"
)

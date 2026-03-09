import numpy as np
import matplotlib.pyplot as plt
from Initial import * 
from Fraction_delta_Binning import *
from scipy.stats import gaussian_kde
import scipy.stats.distributions as dist
from matplotlib.colors import TwoSlopeNorm
from Stat_test_1 import * 



def compute_fractions(arr):
    """
    Computes the fraction of galaxies in different star forming regions

    Parameters:
    arr: array containing delta log SFR values of a sample

    Returns:
    tuple of different fractional values of each star forming regions 
    returns NaN values if the sample size is smaller than 10
    """
    total = len(arr)
    if total < 10:
        return np.nan, np.nan, np.nan, np.nan

    # using the boundary values from aird et al
    f_quenched = np.sum(arr < -1.3) / total
    f_subseq   = np.sum((arr >= -1.3) & (arr < -0.4)) / total
    f_ms       = np.sum((arr >= -0.4) & (arr < 0.4)) / total
    f_starburst= np.sum(arr >= 0.5) / total

    return f_quenched, f_subseq, f_ms, f_starburst

def compute_sample_fraction(sample,m_min,m_max,z_min,z_max):
    """
    computes star forming region fractions within a mass redshift bin

    This function selects galaxies within a specified mass and redshift limit, 
    computes the delta log SFR value for each and computes what star forming region 
    they lie in using the function compute_fractions()

    Parameters: 
    sample: galaxy sample for which the fractions are needed to be calcualted 
    m_min, m_max: lower and upper stellar mass limits 
    z_min, z_max: lower and upper redshift limits

    Returns:
    tuple of fractions of galaxies in each star forming regions

    """

    # limiting the sample to defined mass and redshift bin 
    # using the same method used for applying the volumetric limit in initial.py

    z_min = z_min 
    z_max = z_max

    mask = (
    (sample['redshift_UVISTA_r'] >= z_min) &
    (sample['redshift_UVISTA_r'] < z_max)
    )

    sample = sample[mask]

    m_min = m_min 
    z_min = z_min

    mask = (
        (sample['lp_mass_med'] >= m_min) & 
        (sample['lp_mass_med'] < m_max
        )
    )

    sample = sample[mask].copy()
    n = len(sample)

    # applying a minimum bin value such that the error in fraction is not very high
    # a reasonable error of +- 15% was picked for these fractions 
    # this value of min_n = 10 was picked using beta distributions at 68% confidence 
    # the code for this is seen at the bottom of this script

    min_n = 10  
    
    if n < min_n:
        return np.nan, np.nan, np.nan, np.nan

    sample['logSFR_cut'] = -7.6 + (0.76 * sample['lp_mass_med']) + (2.95*np.log10(1+sample['redshift_UVISTA_r']))
    sample['Delta_logSFR'] =  sample['lp_SFR_med'] - sample['logSFR_cut'] 

    # call for previous function 
    f1 = compute_fractions(sample['Delta_logSFR'])

    return f1[0], f1[1], f1[2], f1[3]



def add_sample_contour(ax, sample, mass_bins, z_bins):
    """ 
    Adds a KDE density contour of a galaxy sample to a plot. 

    This function estimates the two dimensional density distribution
    of galaxies in stellar mass-redshift space using a Gaussian kernel
    density estimator since the sample size is quite small when the subsets
    are used. Hence the use of KDE. 

    Parameters:
    ax: axis on which the contour will be drawn 
    sample: galaxy table for which the contour will be drawn 
    mass_bins, z_bins: mass and redshift bins

    """

    mass = sample['lp_mass_med']
    z    = sample['redshift_UVISTA_r']

    mask = (
        np.isfinite(mass) &
        np.isfinite(z)
    )

    mass = mass[mask]
    z    = z[mask]

    if len(mass) < 15:
        return

    values = np.vstack([mass, z])

    kde = gaussian_kde(values, bw_method=0.55)

    mass_grid = np.linspace(mass_bins[0], mass_bins[-1], 200)
    z_grid    = np.linspace(z_bins[0], z_bins[-1], 200)
    M, Z = np.meshgrid(mass_grid, z_grid)

    coords = np.vstack([M.ravel(), Z.ravel()])
    density = kde(coords).reshape(M.shape)

    levels = np.percentile(density, [60, 80, 95])

    ax.contour(
        M,
        Z,
        density,
        levels=levels,
        colors='black',
        linewidths=1.3,
        alpha=0.9
    )


mass_bins = np.linspace(9.5, 11.5, 5)     
z_bins    = np.linspace(0.2, 1.0, 5)


def fractions_array(sample, mass_bins, z_bins):
    """
    makes an array of fq, fms, and fsb for the sample provided in the bins given.
    computes the fractional values using function from delta binning file. 
    basically converts indiviual fractional value to a group fractional value. 

    Parameters: 
    sample: galaxy sample used to compute the fraction 
    mass_bins, z_bins: mass and redshift bins 
    
    Returns:
    tuple of four 2D arrays corresponding to fraction in each star forming region

    """
    n_mass = len(mass_bins) - 1
    n_z    = len(z_bins) - 1

    fq  = np.zeros((n_mass, n_z))
    fsubs  = np.zeros((n_mass, n_z))
    fms = np.zeros((n_mass, n_z))
    fsb = np.zeros((n_mass, n_z))

    for i in range(n_mass):
        for j in range(n_z):

            m_min = mass_bins[i]
            m_max = mass_bins[i+1]
            z_min = z_bins[j]
            z_max = z_bins[j+1]

            f_quenched, f_subs, f_ms, f_sb = compute_sample_fraction(
                sample, m_min, m_max, z_min, z_max
            )

            fq[i, j]  = f_quenched
            fsubs[i, j]  = f_subs
            fms[i, j] = f_ms
            fsb[i, j] = f_sb

    return fq, fsubs, fms, fsb


def plot_final(samples, mass_bins, z_bins):
    """
    Plots a 4x4 plot of star formation region fractions for multiple galaxy sample

    This function generates a 4x4 panel figure showing the fractions of galaxies in 
    different star forming regions across a mass redshift distributions. 
    Rows correspond to star-formation regimes:
        1. Quenched
        2. Sub-main-sequence
        3. Main sequence
        4. Starburst

    Columns correspond to galaxy samples:
        - Ring-Bar
        - Ring-Unbar
        - No-Ring-Bar
        - No-Ring-Unbar
     
    KDE density contours of each sample are overlaid o the fraction maps to indicate 
    the distributions of galaxies

    Parameters:
    samples: a list of 4 galaxy tables containig the subsamples to be plotted
    mass_bins, z_bins: stellar mass and redshift bins 
    """

    fig, axes = plt.subplots(
        4, 4,
        figsize=(12, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    # to store all mapping data 
    maps = [] 

    for sample in samples:
        fq, fsub, fms, fsb = fractions_array(sample, mass_bins, z_bins)
        maps.append((fq, fsub, fms, fsb))
    fq_all  =  np.concatenate([m[0].flatten() for m in maps])
    fsub_all = np.concatenate([m[0].flatten() for m in maps])
    fms_all =  np.concatenate([m[1].flatten() for m in maps])
    fsb_all =  np.concatenate([m[2].flatten() for m in maps])

    # giving max and min values for the colormap
    vmin_q, vmax_q   = np.nanmin(fq_all),  np.nanmax(fq_all)
    vmin_sub, vmax_sub   = np.nanmin(fsub_all),  np.nanmax(fsub_all)
    vmin_ms, vmax_ms = np.nanmin(fms_all), np.nanmax(fms_all)
    vmin_sb, vmax_sb = np.nanmin(fsb_all), np.nanmax(fsb_all)

    #making individual colormaps
    for col, (fq, fsub, fms, fsb) in enumerate(maps):

        im_q = axes[0, col].pcolormesh(
            mass_bins, z_bins, fq.T,
            cmap='summer_r',
            vmin=vmin_q, vmax=0.8
        )

        im_q = axes[1, col].pcolormesh(
            mass_bins, z_bins, fsub.T,
            cmap='summer_r',
            vmin=vmin_q, vmax=0.8
        )

        im_ms = axes[2, col].pcolormesh(
            mass_bins, z_bins, fms.T,
            cmap='summer_r',
            vmin=vmin_ms, vmax=0.8
        )

        im_sb = axes[3, col].pcolormesh(
            mass_bins, z_bins, fsb.T,
            cmap='summer_r',
            vmin=vmin_sb, vmax=0.8
        )

        current_sample = samples[col]

        for row in range(4):
            add_sample_contour(
            axes[row, col],
            current_sample,
            mass_bins,
            z_bins
        )

    col_titles = ["Ring-Bar", "Ring-Unbar",
                  "No-Ring-Bar", "No-Ring-Unbar"]

    for col in range(4):
        axes[0, col].set_title(col_titles[col],fontsize=14)

    #positioning and formatting
    colorbar_a = fig.colorbar(im_q,  ax=axes[0, :], shrink=0.8)
    colorbar_b = fig.colorbar(im_ms, ax=axes[1, :], shrink=0.8)
    colorbar_c = fig.colorbar(im_ms, ax=axes[2, :], shrink=0.8)
    colorbar_d = fig.colorbar(im_sb, ax=axes[3, :], shrink=0.8)
    colorbar_a.set_label(r"f_quenched",fontsize=18)
    colorbar_b.set_label(r"f_subSq",fontsize=18)
    colorbar_c.set_label(r"f_MainSq",fontsize=18)
    colorbar_d.set_label(r"f_starburst",fontsize=18)

    for ax in axes[:-1, :].flatten():
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
        ax.set_xticks([9.5,10.0,10.5,11.0,11.5])

    for ax in axes[:, 1:].flatten():
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
        ax.set_yticks([0.2,0.4,0.6,0.8,1.0])

        fig.supxlabel(r"log Stellar Mass $M_* \,[M_\odot]$", fontsize=22)
        fig.supylabel(r"redshift $z$", fontsize=22)



    #plt.tight_layout()
    #plt.show()
    plt.savefig(
    "Gallery/test.png",
    dpi=300,
    bbox_inches='tight'
)

samples = [
    Ringed_Disc_Barred_Table,
    Ringed_Disc_Unbarred_Table,
    No_ringed_Disc_Barred_Table,
    No_ringed_Disc_Unbarred_Table
]

plot_final(samples, mass_bins, z_bins)


# new code for new plot 
# these next lines of code will plot a new graph that basically subtracts histogram 
# from the previous plots


def fraction_diff_plot(samples1, mass_bins, z_bins):
    """
    Plot differences in star formation fractions between galaxy categories.

    This function visualizes the differences in fractional populations
    across the stellar mass-redshift plane between:

    - Ring vs No-Ring galaxies
    - Bar vs Unbar galaxies

    The resulting maps of these compared subsamples show what are the likely 
    properties of a galaxy given the star forming status. eg, if it has a ring.


    Parameters: 
    samples1: list containing the galaxy tables used for comparison
    mass_bins, z_bins: mass and redshift bins 

    """ 

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    maps = []

    for sample1 in samples1:
        fq1, fsub1, fms1, fsb1 = fractions_array(sample1, mass_bins, z_bins)
        maps.append((fq1, fsub1, fms1, fsb1))

    # Unpack
    (fqA, fsubA, fmsA, fsbA) = maps[0]  # Ring
    (fqB, fsubB, fmsB, fsbB) = maps[1]  # Bar
    (fqC, fsubC, fmsC, fsbC) = maps[2]  # NoRing
    (fqD, fsubD, fmsD, fsbD) = maps[3]  # Unbar

    delta_fq_ring  = fqA   - fqC
    delta_fq_bar    = fqB   - fqD

    delta_fsub_ring = fsubA - fsubC
    delta_fsub_bar   = fsubB - fsubD
    all_deltas = np.concatenate([
        delta_fq_ring.flatten(),
        delta_fq_bar.flatten(),
        delta_fsub_ring.flatten(),
        delta_fsub_bar.flatten()
    ])

    vmax = np.nanmax(np.abs(all_deltas))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


    im0 = axes[0,0].pcolormesh(
        mass_bins, z_bins, delta_fq_ring,
        cmap='PRGn', norm=norm, shading='auto'
    )

    im1 = axes[1,0].pcolormesh(
        mass_bins, z_bins, delta_fq_bar,
        cmap='PRGn', norm=norm, shading='auto'
    )

    im2 = axes[0,1].pcolormesh(
        mass_bins, z_bins, delta_fsub_ring,
        cmap='PRGn', norm=norm, shading='auto'
    )

    im3 = axes[1,1].pcolormesh(
        mass_bins, z_bins, delta_fsub_bar,
        cmap='PRGn', norm=norm, shading='auto'
    )

    add_sample_contour(axes[0,0],samples1[0],mass_bins,z_bins)
    add_sample_contour(axes[0,1],samples1[0],mass_bins,z_bins)
    add_sample_contour(axes[1,0],samples1[1],mass_bins,z_bins)
    add_sample_contour(axes[1,1],samples1[1],mass_bins,z_bins)

    axes[0,0].set_title(
        r'$\Delta f_{\rm quenched}$' + '\n(Ring - NoRing)'
    )

    axes[1,0].set_title(
        r'$\Delta f_{\rm quenched}$' + '\n(Bar - Unbar)'
    )

    axes[0,1].set_title(
        r'$\Delta f_{\rm sub}$' + '\n(Ring - NoRing)'
    )

    axes[1,1].set_title(
        r'$\Delta f_{\rm sub}$' + '\n(Bar - Unbar)'
    )


    axes[1,0].set_xlabel(r'$\log M_* \,[M_\odot]$',fontsize=20)

    axes[1,0].set_ylabel('Redshift z',fontsize=20)

    mass_ticks = np.arange(9.5, 11.6, 0.5)

    for ax in axes.flatten():
        ax.set_xticks(mass_ticks)

    z_ticks = np.arange(0.2, 1.2, 0.2)

    for ax in axes.flatten():
        ax.set_yticks(z_ticks)

    cbar = fig.colorbar(im0, ax=axes, shrink=0.9)
    cbar.set_label(r'$\Delta f$ (Category2 - Category1)', fontsize=14)

    plt.savefig('Gallery/Test5.png',dpi=300)


#finding minimum bin size using beta distribution

def new_function (n,c=0.68):
    k = n // 2 
    lower = dist.beta.ppf((1-c)/2.,k+1,n-k+1)
    upper = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)

    return (upper - lower)/2

#print errors for n=0 to n=10 bin size

for i in range(10):
    n = new_function(i)

#found +-15% at n=10, hence n=10 was chosen as minimum bin size


#samples1 = [Ringed_Disc_Unbarred_Table, Ringed_Disc_Barred_Table,No_ringed_Disc_Unbarred_Table,No_ringed_Disc_Barred_Table]
samples1 = [ring_galaxies,Clean_Disc_Barred_Galaxies_Table,no_ring_galaxies_1,Clean_Disc_Unbarred_Galaxies_Table]
fraction_diff_plot(samples1, mass_bins, z_bins)
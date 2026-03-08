from astropy.table import Table
import warnings
from astropy.utils.exceptions import AstropyWarning
import numpy as np 

# FINAL 
warnings.simplefilter("ignore", AstropyWarning)
Initial_Table_total = Table.read("Matched_GZH_UVISTA.fits")

# mask initial table so that the entire subset selection is already masked
# volume cut 

z_min = 0.2
z_max = 1.0
m_min = 9.5
m_max = 15.0

mask = (
    (Initial_Table_total['redshift_UVISTA_r'] >= z_min) &
    (Initial_Table_total['redshift_UVISTA_r'] <= z_max) & 
    (Initial_Table_total['lp_mass_med'] >= m_min) & 
    (Initial_Table_total['lp_mass_med'] <= m_max)
)

Initial_Table = Initial_Table_total[mask]


#   RING YES/NO SUBSETS
ring_galaxies_1 = Table.read("ring_galaxies_final.fits")
no_ring_galaxies_1 = Table.read("discs_ring_no.fits")

mask2 = (
    (ring_galaxies_1['redshift_UVISTA_r'] >= z_min) &
    (ring_galaxies_1['redshift_UVISTA_r'] <= z_max) & 
    (ring_galaxies_1['lp_mass_med'] >= m_min) & 
    (ring_galaxies_1['lp_mass_med'] <= m_max))

ring_galaxies = ring_galaxies_1[mask2]


mask3 = (
    (no_ring_galaxies_1['redshift_UVISTA_r'] >= z_min) &
    (no_ring_galaxies_1['redshift_UVISTA_r'] <= z_max) & 
    (no_ring_galaxies_1['lp_mass_med'] >= m_min) & 
    (no_ring_galaxies_1['lp_mass_med'] <= m_max))

no_ring_galaxies = no_ring_galaxies_1[mask3]

ring_mask = np.isin(
    Initial_Table["ID"],
    ring_galaxies["ID"]
)

no_ring_mask = np.isin(
    Initial_Table["ID"],
    no_ring_galaxies["ID"]
)

#only to get plot no 1
#Initial_Table = Initial_Table_total

# creating subsamples

Barred = Initial_Table["t03_bar_a01_bar_fraction"] >= 0.5
Barred_Table = Initial_Table[Barred]

Unbarred = Initial_Table["t03_bar_a01_bar_fraction"] < 0.5
Unbarred_Table = Initial_Table[Unbarred]

Featured = Initial_Table["t01_smooth_or_features_a02_features_or_disk_weighted_fraction"] >= 0.4
Featured_Table = Initial_Table[Featured]
#print(len(Featured_Table))

Not_Artifact = Initial_Table["t01_smooth_or_features_a03_star_or_artifact_weighted_fraction"] <= 0.35
Not_Artifact_Table = Initial_Table[Not_Artifact]
#print(len(Not_Artifact_Table))

Not_Edge_On = (Initial_Table["t02_edgeon_a02_no_weighted_fraction"] >= 0.545) & (Initial_Table["t02_edgeon_total_weight"] >= 8)
Not_Edge_On_Table = Initial_Table[Not_Edge_On]
#print(len(Not_Edge_On_Table))

Not_Clumpy = (Initial_Table["t12_clumpy_a02_no_weighted_fraction"] >= 0.3) & (Initial_Table["t12_clumpy_total_weight"] >= 8)
Not_Clumpy_Table = Initial_Table[Not_Clumpy]
#print(len(Not_Clumpy_Table))

AGN = Initial_Table["is_xray_source"] 
AGN_Table = Initial_Table[AGN]


Is_Not_Odd = ~Initial_Table["all_odd"]
Is_Not_Odd_Table = Initial_Table[Is_Not_Odd]
#print(len(Is_Not_Odd_Table))


Clean_Disc_Galaxies = (
    (Featured) & 
    (Not_Clumpy) & 
    (Not_Edge_On) & 
    (Not_Artifact) & 
    (Is_Not_Odd)
)

Clean_Disc_Galaxies_Table = Initial_Table[Clean_Disc_Galaxies]



Clean_Disc_Barred_Galaxies = (
    (Clean_Disc_Galaxies) & 
    (Barred)
)
Clean_Disc_Barred_Galaxies_Table = Initial_Table[Clean_Disc_Barred_Galaxies]


Clean_Disc_Unbarred_Galaxies = (
    (Clean_Disc_Galaxies) & 
    (Unbarred)
)
Clean_Disc_Unbarred_Galaxies_Table = Initial_Table[Clean_Disc_Unbarred_Galaxies]

#creating merged subsamples

Ringed_Disc_Barred = ((Clean_Disc_Barred_Galaxies) & (ring_mask))
Ringed_Disc_Barred_Table = Initial_Table[Ringed_Disc_Barred]

Ringed_Disc_Unbarred = ((Clean_Disc_Unbarred_Galaxies) & (ring_mask))
Ringed_Disc_Unbarred_Table = Initial_Table[Ringed_Disc_Unbarred]

No_ringed_Disc_Barred = ((Clean_Disc_Barred_Galaxies) & (no_ring_mask))
No_ringed_Disc_Barred_Table = Initial_Table[No_ringed_Disc_Barred]

No_ringed_Disc_Unbarred = ((Clean_Disc_Unbarred_Galaxies) & (no_ring_mask))
No_ringed_Disc_Unbarred_Table = Initial_Table[No_ringed_Disc_Unbarred]


obv_merge_ring = ring_galaxies['obvious_merger']
obv_merge_ring_table = ring_galaxies[obv_merge_ring]


#subsample size

#print("max redshift",max(Clean_Disc_Barred_Galaxies_Table['redshift_UVISTA_r']))
#print("disc bar",len(Clean_Disc_Barred_Galaxies_Table))
#print("disc ubar",len(Clean_Disc_Unbarred_Galaxies_Table))

#print("ring bar",len(Ringed_Disc_Barred_Table))
#print("ring unbar",len(Ringed_Disc_Unbarred_Table))
#print("no ring bar",len(No_ringed_Disc_Barred_Table))
#print("no ring unbar",len(No_ringed_Disc_Unbarred_Table))


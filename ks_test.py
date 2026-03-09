import numpy as np 
from Initial import *
from Stat_test_1 import * 

_1 = Ringed_Disc_Barred_Table              # ring bar 
_2 = Ringed_Disc_Unbarred_Table            # ring unbar 
_3 = No_ringed_Disc_Barred_Table           # no ring bar 
_4 = No_ringed_Disc_Unbarred_Table         # no ring unbar 


Sample1 = _2
Sample2 = _3

#creating bins within the volumetric limit 
m_bins = np.linspace(9.5,13.0,20) 
r_bins = np.linspace(0.2,1.0,20)

all_bins = m_bins, r_bins

Sample1_array = np.array([
    np.array(Sample1["lp_mass_med"]),
    np.array(Sample1["redshift_UVISTA_r"])
])

Sample2_array = np.array([
    np.array(Sample2["lp_mass_med"]),
    np.array(Sample2["redshift_UVISTA_r"])
])
# running this line of code to get the weighted bins to run the KS test later on
w1, w2= weight_dist_dd(Sample1_array,Sample2_array,all_bins,return_bins=False,renorm=False)

# weighted ks test for delta log sfr between two samples

# finding delta log sfr for the two samples 
Sample1['logSFR_cut'] = -7.6 + (0.76 * Sample1['lp_mass_med']) + (2.95*np.log10(1+Sample1['redshift_UVISTA_r']))
Sample1['Delta_logSFR'] =  Sample1['lp_SFR_med'] - Sample1['logSFR_cut'] 

Sample2['logSFR_cut'] = -7.6 + (0.76 * Sample2['lp_mass_med']) + (2.95*np.log10(1+Sample2['redshift_UVISTA_r']))
Sample2['Delta_logSFR'] =  Sample2['lp_SFR_med'] - Sample2['logSFR_cut'] 

Sample1_delsfr = np.array(Sample1['Delta_logSFR'])
Sample2_delsfr = np.array(Sample2['Delta_logSFR'])


ks, p_ks, sig_ks = ks_weighted(Sample1_delsfr,Sample2_delsfr,w1,w2,return_dist=False)


print("ks value",ks)
print("p value",p_ks)
print("sigma",sig_ks)


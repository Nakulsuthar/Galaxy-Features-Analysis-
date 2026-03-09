# Code by Brooke Simmons, last updated 18 February 2025
# don't use this as a model for how to document your code!

import sys, os
import numpy as np
import scipy.stats.distributions as dist
from scipy import special



# have a fraction value of samples from a distribution, and need uncertainties?
# (Ewan Cameron 2011) <-- paper to read and cite about this
# the first 2 functions below will give you lower and upper bounds of a particular 
# confidence level (not uncertainties, but the actual value at those bounds, dx_lo = x - x_lo etc)
# and the next function will do this if you pass the distributions.

# given subpop count k (e.g. n_spirals), sample size n (e.g. n_all), confidence level c (e.g. 0.68 or 0.95), the basic code is:
#p_lower = dist.beta.ppf((1-c)/2.,  k+1,n-k+1)
#p_upper = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)
def p_lower(c, n, k):
    return dist.beta.ppf((1-c)/2.,  k+1,n-k+1)

def p_upper(c, n, k):
    return dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)

def get_errors_on_fractions(subpophist, allpophist, n_random=10000):
    # compute uncertainties using bayesian binomial confidence intervals
    # let's compute this for both 1 sigma and 2 sigma, so both 0.68 and 0.95
    subfrac_upper_1sig = subpophist * 0.0
    subfrac_lower_1sig = subpophist * 0.0
    subfrac_upper_2sig = subpophist * 0.0
    subfrac_lower_2sig = subpophist * 0.0
    c1 = 0.68
    c2 = 0.95

    if n_random > 0:
        subfrac_dists = np.zeros((len(subpophist), n_random))
    else:
        subfrac_dists = []

    for i, allcount in enumerate(allpophist):
        subcount = subpophist[i]
        subfrac_lower_1sig[i] = dist.beta.ppf((1-c1)/2.,   subcount+1, allcount-subcount+1)
        subfrac_upper_1sig[i] = dist.beta.ppf(1-(1-c1)/2., subcount+1, allcount-subcount+1)
        subfrac_lower_2sig[i] = dist.beta.ppf((1-c2)/2.,   subcount+1, allcount-subcount+1)
        subfrac_upper_2sig[i] = dist.beta.ppf(1-(1-c2)/2., subcount+1, allcount-subcount+1)

        if n_random > 0:
            # randomly populate the full beta function so we can do better errors later
            subfrac_dists[i] = np.random.beta(subcount+1, allcount-subcount+1, n_random)

    return subfrac_dists, subfrac_lower_1sig, subfrac_upper_1sig, subfrac_lower_2sig, subfrac_upper_2sig



# this still works, but you can also use weight_dist_d() below
def weight_dist(arr1, arr2, bins=None, return_bins=True, renorm=False):
    # this will take 2 arrays of values from 2 different samples
    # e.g. redshift distributions from 2 samples
    # and return weights for each value such that the weighted
    # distributions of each sample will match.

    # note the bins need to be the same for both datasets
    # so either supply 1 number or 1 array of bin edges
    # also if you have fully specified the bins you don't need them returned
    # but if we've had to figure them out, you do need them returned
    # if you want to make sure you aren't under-weighting (e.g. if a whole dataset is
    # much larger than the other so you might always be able select >1 object
    # in dataset 1 for each object in dataset 2) then you can re-normalise to make
    # sure the max weight of both datasets is 1
    # Note, if the distributions cross this won't make a difference

    # weight arrays
    w1 = np.zeros_like(arr1)
    w2 = np.zeros_like(arr2)

    # if bins not specified, guess at them ourselves
    if bins is None:
        minsize = np.amin([    len(arr1),     len(arr2)])
        themin  = np.amin([np.amin(arr1), np.amin(arr2)])
        themax  = np.amax([np.amax(arr1), np.amax(arr2)])

        # on average 5 data points per bin, but at least 3 bins pls, max value is last bin edge
        bins = np.linspace(themin, themax, int(np.amax([(minsize/5)+1, 3])), endpoint=True)

    else:
        pass
        # because the np.histogram function can deal with distinguishing between number of bins or specific bins itself
        # so we don't have to

    hist1, thebins = np.histogram(arr1, bins=bins)  # returns (counts_arr, bins_arr)
    # use the bins from hist1 to make hist2
    # note: for these purposes, any values of hist2 outside the minmax range of hist1 should have 0 weight
    # which is taken care of by the zeros_like initialisation of w1 and w2 above
    # so it's fine for them to be excluded below
    hist2, thebins = np.histogram(arr2, bins=thebins)

    # now step through the bins and assign weights
    for i_bin in range(len(thebins)-1):
        # zero-"index"ing these because they're indices and not values
        # and if I get them mixed up below I want this to error that there's no b2 or whatever
        b0 = thebins[i_bin]
        b1 = thebins[i_bin+1]

        count1 = hist1[i_bin]
        count2 = hist2[i_bin]

        # don't miss any values and don't double-count
        if i_bin == 0:
            in_bin1 = (arr1 >= b0) & (arr1 <= b1)
            in_bin2 = (arr2 >= b0) & (arr2 <= b1)
        else: 
            in_bin1 = (arr1 >  b0) & (arr1 <= b1)
            in_bin2 = (arr2 >  b0) & (arr2 <= b1)


        # don't divide by 0 in the rest of the if/else
        if (count1 == 0) | (count2 == 0):
            w1[in_bin1] = 0.0
            w2[in_bin2] = 0.0

        elif count1 < count2:
            # weight count2 values so the sum of wt2 in this bin equals count1
            # wt_fac will always be < 1
            wt_fac = float(count1)/float(count2)
            w1[in_bin1] = 1.0
            w2[in_bin2] = wt_fac

        else:
            # weight count1 values so the sum of wt1 in this bin equals count2
            # wt_fac will always be <= 1
            wt_fac = float(count2)/float(count1)
            w1[in_bin1] = wt_fac
            w2[in_bin2] = 1.0


    # now the weights should be determined
    # we can optionally re-normalise to make sure we are getting max value out of the datasets
    if renorm:
        if (np.sum(w1) > 0.00000) & (np.amax(w1) < 1.0):
            w1 /= np.amax(w1)

        if (np.sum(w2) > 0.00000) & (np.amax(w2) < 1.0):
            w2 /= np.amax(w2)


    if return_bins:
        return w1, w2, thebins 
    else: 
        return w1, w2



# this is now deprecated -- still works, but you may as well use weight_dist_d() below
def weight_dist_3(arr1, arr2, arr3, bins=None, return_bins=True, renorm=False):
    # this will take 3 arrays of values from 3 different samples
    # e.g. redshift distributions from 3 samples
    # and return weights for each value such that the weighted
    # distributions of each sample will match.

    # note the bins need to be the same for all datasets
    # so either supply 1 number or 1 array of bin edges
    # also if you have fully specified the bins you don't need them returned
    # but if we've had to figure them out, you do need them returned
    # if you want to make sure you aren't under-weighting (e.g. if a whole dataset is
    # much larger than the others so you might always be able select >1 object
    # in dataset 1 for each object in dataset 2) then you can re-normalise to make
    # sure the max weight of all datasets is 1
    # Note, if the distributions cross this won't make a difference

    # weight arrays
    w1 = np.zeros_like(arr1)
    w2 = np.zeros_like(arr2)
    w3 = np.zeros_like(arr3)

    # if bins not specified, guess at them ourselves
    if bins is None:
        minsize = np.amin([    len(arr1),     len(arr2), len(arr3)])
        themin  = np.amin([np.amin(arr1), np.amin(arr2),np.amin(arr3)])
        themax  = np.amax([np.amax(arr1), np.amax(arr2),np.amin(arr3)])

        # on average 5 data points per bin, but at least 3 bins pls, max value is last bin edge
        bins = np.linspace(themin, themax, int(np.amax([(minsize/5)+1, 3])), endpoint=True)

    else:
        pass
        # because the np.histogram function can deal with distinguishing between number of bins or specific bins itself
        # so we don't have to

    hist1, thebins = np.histogram(arr1, bins=bins)  # returns (counts_arr, bins_arr)
    # use the bins from hist1 to make hist2
    # note: for these purposes, any values of hist2 outside the minmax range of hist1 should have 0 weight
    # which is taken care of by the zeros_like initialisation of w1 and w2 above
    # so it's fine for them to be excluded below
    hist2, thebins = np.histogram(arr2, bins=thebins)
    hist3, thebins = np.histogram(arr3, bins=thebins)

    # now step through the bins and assign weights
    for i_bin in range(len(thebins)-1):
        # zero-"index"ing these because they're indices and not values
        # and if I get them mixed up below I want this to error that there's no b2 or whatever
        b0 = thebins[i_bin]
        b1 = thebins[i_bin+1]

        count1 = hist1[i_bin]
        count2 = hist2[i_bin]
        count3 = hist3[i_bin]

        # don't miss any values and don't double-count
        if i_bin == 0:
            in_bin1 = (arr1 >= b0) & (arr1 <= b1)
            in_bin2 = (arr2 >= b0) & (arr2 <= b1)
            in_bin3 = (arr3 >= b0) & (arr3 <= b1)
        else: 
            in_bin1 = (arr1 >  b0) & (arr1 <= b1)
            in_bin2 = (arr2 >  b0) & (arr2 <= b1)
            in_bin3 = (arr3 >  b0) & (arr3 <= b1)


        # don't divide by 0 in the rest of the if/else
        if (count1 == 0) | (count2 == 0)  | (count3 == 0):
            w1[in_bin1] = 0.0
            w2[in_bin2] = 0.0
            w3[in_bin3] = 0.0


        elif count1 < count2 and count1 < count3 :
            # weight count2 values so the sum of wt2 in this bin equals count1
            # wt_fac will always be < 1
            wt_fac1 = float(count1)/float(count2)
            wt_fac2 = float(count1)/float(count3)
            w1[in_bin1] = 1.0
            w2[in_bin2] = wt_fac1
            w3[in_bin3] = wt_fac2

        elif count3 < count2 and count3 < count1 :
            # weight count2 values so the sum of wt2 in this bin equals count1
            # wt_fac will always be < 1
            wt_fac1 = float(count3)/float(count2)
            wt_fac2 = float(count3)/float(count1)
            w3[in_bin3] = 1.0
            w2[in_bin2] = wt_fac1
            w1[in_bin1] = wt_fac2

        elif count1 == count2 and count1 < count3 :
            # weight count2 values so the sum of wt2 in this bin equals count1
            # wt_fac will always be < 1
            wt_fac2 = float(count1)/float(count3)
            w1[in_bin1] = 1.0
            w2[in_bin2] = 1.0
            w3[in_bin3] = wt_fac2

        elif count1 == count3 and count1 < count2 :
            # weight count2 values so the sum of wt2 in this bin equals count1
            # wt_fac will always be < 1
            wt_fac2 = float(count1)/float(count2)
            w1[in_bin1] = 1.0
            w2[in_bin2] = wt_fac2
            w3[in_bin3] = 1.0
            
        elif count3 == count2 and count3 < count1 :
            # weight count2 values so the sum of wt2 in this bin equals count1
            # wt_fac will always be < 1
            wt_fac2 = float(count3)/float(count1)
            w3[in_bin3] = 1.0
            w2[in_bin2] = 1.0
            w1[in_bin1] = wt_fac2

        elif count3 == count2 and count3 == count1 :
            # weight count2 values so the sum of wt2 in this bin equals count1
            # wt_fac will always be < 1
            wt_fac2 = float(count3)/float(count1)
            w3[in_bin3] = 1.0
            w2[in_bin2] = 1.0
            w1[in_bin1] = 1.0

        else:
            # weight count1 values so the sum of wt1 in this bin equals count2
            # wt_fac will always be <= 1
            wt_fac1 = float(count2)/float(count1)
            wt_fac2 = float(count2)/float(count3)
            w1[in_bin1] = wt_fac1
            w2[in_bin2] = 1.0
            w3[in_bin3] = wt_fac2


    # now the weights should be determined
    # we can optionally re-normalise to make sure we are getting max value out of the datasets
    if renorm:
        if (np.sum(w1) > 0.00000) & (np.amax(w1) < 1.0):
            w1 /= np.amax(w1)

        if (np.sum(w2) > 0.00000) & (np.amax(w2) < 1.0):
            w2 /= np.amax(w2)
        
        if (np.sum(w3) > 0.00000) & (np.amax(w3) < 1.0):
            w3 /= np.amax(w3)


    if return_bins:
        return w1, w2, w3, thebins 
    else: 
        return w1, w2, w3





def weight_dist_d(samples, bins=None, return_bins=True, renorm=False):

    # this will take N arrays as samples = [array1, array2, ..., arrayN]
    # bin those samples (if bins are specified, great; if not, it will try
    # to make the bins for you), and then determine weights for the samples
    # such that the weighted arrays will have matched distributions under
    # those bins.
    # The function will return the array (not a list) of weight arrays.
    # if return_bins == True, it will also return the bins used.
    # if renorm == True, each weight array will be renormalised so that its
    # maximum weight is 1.0 (might be useful if some samples are much larger
    # than others; then again it might not change the weights at all).

    # It is recommended to do the binning yourself and specify the bins --
    # the binning must be the same for all samples, so the bin array is 1
    # array with the bin edges that will be used across all N sample arrays.

    # list of weight arrays - initialise to zero, same shape as samples
    weights = []
    for s in samples:
        weights.append(np.zeros_like(s).astype(float))
    # note: I will use list comprehension more often below -- if I had
    # done this for the above it would have looked like:
    # weights = [np.zeros_like(s).astype(float) for s in samples]


    # if bins not specified, guess at them ourselves
    # I really really suggest you determine the bins yourself
    # or at least say how many bins you want (pass an integer to the function)
    if bins is None:
        lengths = [    len(s) for s in samples]
        themins = [np.amin(s) for s in samples]
        themaxs = [np.amax(s) for s in samples]

        minsize = np.amin(lengths)
        themin  = np.amin(themins)
        themax  = np.amax(themaxs)

        # on average 5 data points per bin, but at least 3 bins pls, max value is last bin edge
        bins = np.linspace(themin, themax, int(np.amax([(minsize/5)+1, 3])), endpoint=True)

    else:
        pass
        # because the np.histogram function can deal with distinguishing between being passed
        # an integer representing the number of bins or the specific bin array, by itself
        # so we don't have to


    # use the bins from hist1 to make all the other histograms
    hist1, thebins = np.histogram(samples[0], bins=bins)  # returns (counts_arr, bins_arr)
    # note: for these purposes, any values of hists outside the minmax range of hist1 should have 0 weight
    # which is taken care of by the zeros_like initialisation of weights above
    # so it's fine for them to be excluded below

    # note this will remake the first histogram as well but that's a minor computational cost
    hists_bins = [np.histogram(s, bins=thebins) for s in samples]
    # this has returned a list of (hist, bins) tuples so extract the histograms

    # we don't need to extract the bins because, by construction, they're all the same, 
    # and already stored in thebins
    hists = [h[0] for h in hists_bins]


    # now step through the bins and assign weights
    for i_bin in range(len(thebins)-1):
        # zero-"index"ing these because they're indices and not values
        # and if I get them mixed up below I want this to error that there's no b2 or whatever
        # bin edges
        b0 = thebins[i_bin]
        b1 = thebins[i_bin+1]

        # what is the number of sources in this bin for each sample?
        counts = np.array([h[i_bin] for h in hists])

        # identify the elements of each sample array that are in this bin
        # don't miss any values and don't double-count
        if i_bin == 0:
            in_bins = [(s >= b0) & (s <= b1) for s in samples]
        else: 
            in_bins = [(s >  b0) & (s <= b1) for s in samples]


        # don't divide by 0 in the rest of the if/else if any of them are zero
        if np.any(counts == 0):
            # there are no counts for at least 1 sample in this bin
            # so just leave the zero-initialised weights as is, as
            # they would all be zero-weighted anyway
            pass

        else:
            # all weights are the counts normalised to make weighted count
            # equal to the min count over all counts
            min_count = np.amin(counts)
            # don't divide by zero though, should already be covered but let's be sure
            # w_facs = min_count/np.array(counts)
            w_facs = np.zeros_like(counts).astype(float)
            w_facs[counts > 0] = min_count/counts[counts > 0]

            # for each sample, assign the weight factor for this sample
            # to the elements of the weight array that correspond to the
            # sources in this specific bin
            #
            for i_s in range(len(samples)):
                in_this_bin = in_bins[i_s]
                weights[i_s][in_this_bin] = w_facs[i_s]


    # by default, return these weights   
    weights_return = weights

    # now the weights should be determined
    # we can optionally re-normalise to make sure we are getting max value out of the datasets
    if renorm:
        # I can't bring myself to overwrite weights,
        # in case this ever needs debugging and I need to compare these
        weights_norm = [w/np.amax(w) for w in weights]
        weights_return = weights_norm


    if return_bins:
        return weights_return, thebins 
    else: 
        return weights_return










def weight_dist_dd(sample1, sample2, bins=None, return_bins=True, renorm=False):
    # this will take 2 arrays of values from 2 different samples
    # e.g. redshift distributions from 2 samples
    # and return weights for each value such that the weighted
    # distributions of each sample will match.
    # if you have multiple dimensions along which to weight,
    # pass each array as a list/array of arrays
    # e.g. sample1 = [mass_arr1, z_arr1] and
    #      sample2 = [mass_arr2, z_arr2]
    # and if passing bins, also 2D bins etc.
    # if you want weighting in 1D just use weight_dist()

    
    # note the bins need to be the same for both datasets
    # so either supply 1 number or 1 array of bin edges
    # also if you have fully specified the bins you don't need them returned
    # but if we've had to figure them out, you do need them returned
    # if you want to make sure you aren't under-weighting (e.g. if a whole dataset is
    # much larger than the other so you might always be able select >1 object
    # in dataset 1 for each object in dataset 2) then you can re-normalise to make
    # sure the max weight of both datasets is 1
    # Note, if the distributions cross this won't make a difference

    # avoid errors if these are lists, or Series, or whatever
    sample1 = np.array(sample1, dtype=object)
    sample2 = np.array(sample2, dtype=object)

    # we just care about how many dimensions there are in the sample,
    # not what the sample sizes are
    ndim = np.shape(sample1)[0]

    # weight arrays
    # the weights are for the sample, not the individual distributions
    # so e.g. if sample1 = ([mass1, z1])  
    # the weights are an array with the same dimension as mass1 or z1
    # (which themselves need to have matching dimensions)
    w1 = np.zeros_like(sample1[0])
    w2 = np.zeros_like(sample2[0])

    # if bins not specified, guess at them ourselves
    if bins is None:
        bins = []
        for i_dim in range(ndim):
            minsize = np.amin([    len(sample1[i_dim]),     len(sample2[i_dim])])
            themin  = np.amin([np.amin(sample1[i_dim]), np.amin(sample2[i_dim])])
            themax  = np.amax([np.amax(sample1[i_dim]), np.amax(sample2[i_dim])])

            # on average 5 data points per bin, but at least 3 bins pls, max value is last bin edge
            bins.append(np.linspace(themin, themax, int(np.amax([(minsize/5)+1, 3])), endpoint=True))
        bins = np.array(bins)

    else:
        pass
        # because the np.histogram function can deal with distinguishing between number of bins or specific bins itself
        # so we don't have to

    # histogramdd wants an array of coordinates, e.g. ([x1, y1], [x2, y2],...)
    # not an array of ([x_all, y_all])
    # so you need to pass the transpose
    hist1, thebins = np.histogramdd(sample1.T, bins=bins)  # returns (counts_arr, bins_arr)
    # use the bins from hist1 to make hist2
    # note: for these purposes, any values of hist2 outside the minmax range of hist1 should have 0 weight
    # which is taken care of by the zeros_like initialisation of w1 and w2 above
    # so it's fine for them to be excluded below
    hist2, thebins = np.histogramdd(sample2.T, bins=thebins)

    nbins_tot = 1
    for ii in range(np.array(thebins).shape[0]):
        nbins_tot *= len(thebins[ii]-1)

    bin_id1    = []
    bin_id2    = []
    bin_unique = []
    for i_dim in range(ndim):
        # assign bin numbers to each data point in each dimension
        # note the digitize function single-indexes
        # e.g. if it returns j_bin, then the value is between
        # bin edges j_bin-1 and j_bin
        bin_id1.append(np.digitize(sample1[i_dim], thebins[i_dim]))
        bin_id2.append(np.digitize(sample2[i_dim], thebins[i_dim]))

        # bin_unique.append(np.unique(np.append(np.array(bin_id1[i_dim]), np.array(bin_id2[i_dim]))))

    # avoid any list vs array errors downline
    bin_id1    = np.array(bin_id1)
    bin_id2    = np.array(bin_id2)

    # the unique combination of bin matches is in the transpose
    # ie not ([bin_ids_x, bin_ids_y, bin_ids_z])
    # but ([bin_id_x1, bin_id_y1, bin_id_z1], [bin_id_x2, bin_id_y2, bin_id_z2],...)
    # and I'll need it in the loop below so do that once and save it
    bin_id1_T = bin_id1.T
    bin_id2_T = bin_id2.T

    # however many arrays per sample, the bins are the same 
    # for samples 1 and 2
    # so I can append the used bin combinations for both samples
    # then figure out what unique bin combinations are actually used
    # then get weights from them individually
    # this feels a bit janky but until I figure out the more
    # pythonic version I'm going to run with it

    # the axis=0 keeps it from going multiple levels into the arrays
    # which would just return a bunch of single integers
    all_bin_ids = np.unique(np.append(bin_id1_T, bin_id2_T, axis=0), axis=0)


    # now step through the bins and assign weights
    for this_bin in all_bin_ids:

        # figure out which points are in this bin
        in_bin1 = np.all(bin_id1_T == this_bin, axis=1)
        in_bin2 = np.all(bin_id2_T == this_bin, axis=1)

        count1 = np.sum(in_bin1)
        count2 = np.sum(in_bin2)


        # don't divide by 0 in the rest of the if/else
        if (count1 == 0) | (count2 == 0):
            w1[in_bin1] = 0.0
            w2[in_bin2] = 0.0

        elif count1 < count2:
            # weight count2 values so the sum of wt2 in this bin equals count1
            # wt_fac will always be < 1
            wt_fac = float(count1)/float(count2)
            w1[in_bin1] = 1.0
            w2[in_bin2] = wt_fac

        else:
            # weight count1 values so the sum of wt1 in this bin equals count2
            # wt_fac will always be <= 1
            wt_fac = float(count2)/float(count1)
            w1[in_bin1] = wt_fac
            w2[in_bin2] = 1.0


    # now the weights should be determined
    # we can optionally re-normalise to make sure we are getting max value out of the datasets
    if renorm:
        if (np.sum(w1) > 0.00000) & (np.amax(w1) < 1.0):
            w1 /= np.amax(w1)

        if (np.sum(w2) > 0.00000) & (np.amax(w2) < 1.0):
            w2 /= np.amax(w2)


    if return_bins:
        return w1, w2, thebins 
    else: 
        return w1, w2



def pick_sample(weights):
    # Send this function a list/array of weights and it will send back a
    # list of True/False values where the probability of any given value
    # being True is the value of the weight. 
    # this has the effect of picking a subsample based on the weighting
    # which you've already determined in a different function, e.g. to
    # match subsample distributions along some axis.

    # for each element of the weight array, generate a random number
    # where the value has a uniform probability of being anything between 
    # 0 and 1
    the_randoms = np.random.uniform(low=0.0, high=1.0, size=len(weights))

    # use the weights, which are also between 0 and 1 (but inclusive of 1)
    # to select the sample.
    # you can convince yourself that if e.g. the weight value for a given
    # element is 0.5, then the uniform random number will have a 50% chance
    # of being selected, so if you have a bin where all the weights are 0.5
    # then overall you will select half of the sample as True
    # which is what you want when picking a sample.
    # And if the weight is higher, you want to be more likely to pick,
    # which is how this inequality works out.
    # if the weight is 1, every sample will be picked because
    # np.random.uniform doesn't include the high value (but does include
    # the low value). If your weight is 0, you could in theory choose a
    # galaxy but that is highly unlikely.
    in_sample = the_randoms <= np.array(weights)

    return in_sample
    # yes, this is 3 lines of code but >20 lines of description. That's how coding is sometimes.





# the intersection of the 2 distributions, equivalent to the intersection of the integral under 2 curves
# where the curves are the shape of the distributions
# but note: you pass the actual arrays of values (x only) that make the distributions, not (x, y) curves
def prob_dist_overlap(arr1, arr2, getsigma=False):

    # provides weights needed such that weighted histograms of arr1 and arr2 are statistically indistinguishable
    # which, in practice, is the overlap of the distributions
    w1, w2, thebins = weight_dist(arr1, arr2, renorm=True)

    # technically we don't want to double count the overlap area in either numerator or denominator
    # but if your statistics depend on quibbling about this, your significance is marginal at best
    # so report that and don't overegg the results
    overlap_area = 0.5*(np.sum(w1) + np.sum(w2))
    total_area = float(len(arr1) + len(arr2)) - np.sum(w1)

    f_overlap = overlap_area / total_area

    if getsigma:
        # the special.erfc(x) is the complementary error function, where x = sigma/sqrt(2)
        # i.e. the table of p-value-to-sigma at https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_tolerance_intervals is actually a table of erf(x), erfc(x), 1./erfc(x) for sigma values from 1 to 6
        # and special.erfcinv(alpha)*np.sqrt(2.) will return the significance level in sigma.
        # however, WARNING - this sigma is only valid assuming the distributions are normal
        # which a lot of this utils file assumes they aren't.
        # so use as guidance but not gospel, and very much with caution.
        sigma_overlap = special.erfcinv(f_overlap)*np.sqrt(2.)
        return f_overlap, sigma_overlap
    else:
        return f_overlap




# calculate percentiles for weighted distributions.
# what to do if there's loads of zero-weighted stuff in here? think we have to remove those
# 
# x == rank
# p == pctile_val
# N == dimension of array
# the method numpy uses for straight percentile:
# x = p(N-1) + 1
# then linearly interpolate... 
# we have to do something similar except with weights
# there's an extension for weighted percentiles in wikipedia
# https://en.wikipedia.org/wiki/Percentile#Weighted_percentile
# I have tested it for various things and it seems to work fine

def percentile_wt(sample, weights, pct, verbose=False):

    # these can be passed as lists, arrays, pd.Series, etc -- we need them as arrays
    sample  = np.array(sample)
    weights = np.array(weights)

    if not ((pct >= 0.0) & (pct <= 100.0)):
        print("Percentile requested must be between 0 and 100")
        return np.nan 
    else:
        pass


    count_this = float(np.sum(weights))
    if count_this <= 0.0:
        print("Sum of weights is 0 or negative, something has gone very wrong")
        return np.nan

    else:
        zero_weights = weights <= 0.0
        if (sum(zero_weights) > 0) & verbose:
            print("Removing zero-weight points before computing")
        ssample = sample[np.invert(zero_weights)].copy()
        wweights = weights[np.invert(zero_weights)].copy()
        j_sorted = ssample.argsort()
        sample_sort = ssample[j_sorted]
        weight_sort = weights[j_sorted]
        _c = np.cumsum(weight_sort)

        p_rank = _c/float(_c[-1])

        x = pct/100.*(sum(weight_sort)-1) + 1.

        x_n = x/_c[-1]

        # searchsorted returns the index immediately *before* which you'd insert a 
        # new value to maintain the order
        j_which = np.searchsorted(_c, x, side='left') - 1

        if j_which < 0:
            return sample_sort[0]
        else:

            #print(j_which, "==========")
            #print("_c", _c, "x", x, "x_n", x_n, "p_rank", p_rank, "sample", sample)
            if j_which == len(sample_sort)-1:
                if verbose:
                    print("Returning highest value (index %d)" % j_which)

                return sample_sort[j_which]
            else:        
                return sample_sort[j_which] + ((x_n-p_rank[j_which])/(p_rank[j_which+1]-p_rank[j_which]))*(sample_sort[j_which+1]-sample_sort[j_which])


        # # linearly interpolate
        # cdist_tot   = rank[j_which+1] - rank[j_which]
        # cdist_left  = pct - rank[j_which]
        # cdist_right = rank[j_which+1] - pct

        # # whichever has the shortest distance gets the largest weight, so swap left & right distances
        # return (cdist_right/cdist_tot)*sample_sort[j_which] + (cdist_left/cdist_tot)*sample_sort[j_which+1]



def get_stats_indices():
# This function gives you the indices of arrays/dictionaries
# you'll need to use to retrieve particular values after you've
# called get_basic_stats().

    n_val_basicstats = 15
    i_mean   = 0
    i_median = 1
    i_count  = 2
    i_16p    = 3
    i_25p    = 4
    i_75p    = 5
    i_84p    = 6
    i_var    = 7
    i_varmed = 8
    i_05p    = 9  #0.5th pctile
    i_5p     = 10
    i_95p    = 11
    i_995p   = 12 # 99.5th pctile
    i_025p   = 13 # 2.5th percentile
    i_975p   = 14 # 97.5th percentile
    # it irks me that these are out of order, but the point of using these variables
    # is that you don't have to remember the index -- just use the variable for the
    # value you want.

    # also keep it in a dictionary just because.
    i_stats = {'mean':    0,
               'median':  1,
               'count':   2,
               '16p':     3,
               '25p':     4,
               '75p':     5,
               '84p':     6,
               'var':     7,
               'varmed':  8,
               '05p':     9,
               '5p':     10,
               '95p':    11,
               '995p':   12,
               '025p':   13,
               '975p':   14
               }

    return n_val_basicstats, i_mean, i_median, i_count, i_16p, i_25p, i_75p, i_84p, i_var, i_varmed, i_05p, i_5p, i_95p, i_995p, i_025p, i_975p, i_stats





# calculate some basic stats about a sample
def get_basic_stats(sample, weights=None, verbose=False):

    n_val_basicstats, i_mean, i_median, i_count, i_16p, i_25p, i_75p, i_84p, i_var, i_varmed, i_05p, i_5p, i_95p, i_995p, i_025p, i_975p, i_stats = get_stats_indices()

    basic_stats = np.zeros(n_val_basicstats)

    if weights is None:
        themedian  = np.median(sample)
        count_this = len(sample)

        basic_stats[i_count] = count_this

        if count_this > 0:
            basic_stats[i_mean] = np.mean(sample)
            basic_stats[i_median] = themedian
            if count_this > 2:
                basic_stats[i_05p]  = np.percentile(sample,  0.5)
                basic_stats[i_5p]   = np.percentile(sample,  5)
                basic_stats[i_16p]  = np.percentile(sample, 16)
                basic_stats[i_25p]  = np.percentile(sample, 25)
                basic_stats[i_75p]  = np.percentile(sample, 75)
                basic_stats[i_84p]  = np.percentile(sample, 84)
                basic_stats[i_95p]  = np.percentile(sample, 95)
                basic_stats[i_995p] = np.percentile(sample, 99.5)
                basic_stats[i_025p] = np.percentile(sample,  2.5)
                basic_stats[i_975p] = np.percentile(sample, 97.5)
                basic_stats[i_var] = np.var(sample)

            # try to estimate the variance on the median, which isn't built into numpy
            # https://en.wikipedia.org/wiki/Median#The_sample_median
            # we will use the asymptotic approximation, which will tend to overestimate
            #    the variance, especially when the sample size is small
            # these seem really small when plotted though
            if count_this > 10:
                # we want at least 4 gals per bin on average
                this_nbins = int(min((count_this / 4, 12)))
                # we need to get the PDF for this bin and get the value of it at the median
                sample_hist = np.histogram(sample, bins=this_nbins, density=True)
                # identify all bins with x values <= the median, then take the last one
                medbins = sample_hist[0][sample_hist[1][:-1] <= themedian]
                pdf_at_median = medbins[-1]

            else:
                # not really sure what to do here b/c we don't really know the distribution
                #pdf_at_median = 0.5
                # this goes from 0.4 at n=1 to 0.6 at n=10, but it's kind of arbitrary
                # just trying to characterize doing better at the median with more points
                pdf_at_median = (1./30.)*float(count_this - 1) + 0.4

            basic_stats[i_varmed] = 1./(4.*float(count_this)*pdf_at_median**2)


        return basic_stats

    else: 
        # there are weights, we need to use them to compute everything
        count_this = float(np.sum(weights))
        basic_stats[i_count] = count_this

        # some things are built into numpy, others not so much

        if count_this > 0:

            basic_stats[i_mean] = avg = np.average(sample, weights=weights)
            basic_stats[i_var]  = np.sqrt(np.average((sample-avg)**2, weights=weights))
    
            if len(sample[weights >= 0.0] > 2):
                basic_stats[i_05p]  = percentile_wt(sample, weights, 0.5, verbose=verbose)
                basic_stats[i_5p]   = percentile_wt(sample, weights, 5, verbose=verbose)
                basic_stats[i_16p]  = percentile_wt(sample, weights, 16, verbose=verbose)
                basic_stats[i_25p]  = percentile_wt(sample, weights, 25, verbose=verbose)
                basic_stats[i_median]  = themedian = percentile_wt(sample, weights, 50, verbose=verbose)
                basic_stats[i_75p]  = percentile_wt(sample, weights, 75, verbose=verbose)
                basic_stats[i_84p]  = percentile_wt(sample, weights, 84, verbose=verbose)
                basic_stats[i_95p]  = percentile_wt(sample, weights, 95, verbose=verbose)
                basic_stats[i_995p] = percentile_wt(sample, weights, 99.5, verbose=verbose)
                basic_stats[i_025p] = percentile_wt(sample, weights,  2.5, verbose=verbose)
                basic_stats[i_975p] = percentile_wt(sample, weights, 97.5, verbose=verbose)
 

            # try to estimate the variance on the median, which isn't built into numpy
            # https://en.wikipedia.org/wiki/Median#The_sample_median
            # we will use the asymptotic approximation, which will tend to overestimate
            #    the variance, especially when the sample size is small
            # these seem really small when plotted though
            if len(sample[weights >= 0.0]) > 10:
                # we want at least 4 gals per bin on average
                this_nbins = int(min((len(sample[weights >= 0.0]) / 4, 12)))
                # we need to get the PDF for this bin and get the value of it at the median
                sample_hist = np.histogram(sample, bins=this_nbins, density=True, weights=weights)
                # identify all bins with x values <= the median, then take the last one
                medbins = sample_hist[0][sample_hist[1][:-1] <= themedian]
                pdf_at_median = medbins[-1]

            else:
                # not really sure what to do here b/c we don't really know the distribution
                #pdf_at_median = 0.5
                # this goes from 0.4 at n=1 to 0.6 at n=10, but it's kind of arbitrary
                # just trying to characterize doing better at the median with more points
                pdf_at_median = (1./30.)*float(len(sample[weights >= 0.0]) - 1) + 0.4

            basic_stats[i_varmed] = 1./(4.*float(len(sample[weights >= 0.0]))*pdf_at_median**2)


        return basic_stats





def bin_array(the_array, the_bin_boundaries, list_of_other_arrays=None, weights=None):
# binning in 1D - uses functions above
# usually you want to bin a bunch of things (like stellar mass, BH mass, SFR, X-ray luminosity, etc)
# according to a different quantity (like redshift)
# so in that case you'd call
# bin_array(redshift, redshift_bin_boundaries, list_of_other_arrays=[stellar_mass, bh_mass, SFR, Xray_lum, etc])
# but if you just want to get stats and bin 1 array, then go for it, the list (and weight array) is optional

    # in case the function was passed a list or tuple instead of an array
    # if it's already an array this won't do anything
    the_array = np.array(the_array)

    # some built-in indices we will find it useful to have
    n_val_basicstats, i_mean, i_median, i_count, i_16p, i_25p, i_75p, i_84p, i_var, i_varmed, i_05p, i_5p, i_95p, i_995p, i_025p, i_975p, i_stats = get_stats_indices()

    # there are 1 fewer bins than bin boundaries, so get arrays corresponding to the left
    # and right edges of each bin, and the centers, which have the correct dimensions
    bins_lo  = np.array(the_bin_boundaries[:-1])
    bins_hi  = np.array(the_bin_boundaries[1:])
    bins_ctr = 0.5*(bins_lo + bins_hi)

    # define the arrays/lists we need
    array_binned = []
    stats_t      = []
    if list_of_other_arrays is not None:
        array_binned_list_t_v = []
        array_binned_list_t   = []
        array_binned_list     = []
        stats_list_t_v        = []
        stats_list_t          = []
    else:
        array_binned_list = None
        stats_list_t      = None


    # sort values into each bin
    for _i in range(len(bins_lo)):
        # pick out values from the array that are in this bin

        # if it's the last bin, it's ok if the value equals the bin limit
        # otherwise one test should be "or equals" and the other not, to
        # prevent double-counting
        if (_i == (len(bins_lo)-1)):
            in_bin = (the_array >= bins_lo[_i]) & (the_array <= bins_hi[_i])
        else:
            in_bin = (the_array >= bins_lo[_i]) & (the_array < bins_hi[_i])

        # set a subarray
        a_bin = the_array[in_bin]

        # save the individual sub-arrays in a list, which we may need later
        array_binned.append(a_bin)

        # get stats
        a_stats = get_basic_stats(a_bin, weights=weights)
        # save stats for each bin to a list (we'll rearrange this later)
        stats_t.append(a_stats)

        # now do the same on every array in the list of arrays, if there is one
        if list_of_other_arrays is not None:
            array_binned_list_t_v = []
            # stats_list_t_v        = []
            for _v, vals in enumerate(list_of_other_arrays):
                vals_arr = np.array(vals)
                vals_bin = vals_arr[in_bin]

                array_binned_list_t_v.append(vals_bin)

                v_stats = get_basic_stats(vals_bin, weights=weights)

                stats_list_t_v.append(v_stats)

            # this is a list of lists, bit gross but we'll sort it out later
            array_binned_list_t.append(array_binned_list_t_v)
            stats_list_t.append(stats_list_t_v)




    # right now if we wanted e.g. the median values for each bin we'd need to get them with
    # stats_t[0][i_median], stats_t[1][i_median], etc.
    # when we'd like to have stats[i_median] contain an array with all the bins
    # i.e., we need the transpose.

    the_stats = np.array(stats_t).T 


    # do the same on every array in the list of arrays, if there is one
    if list_of_other_arrays is not None:
        array_binned_list = np.array(array_binned_list_t).T

        the_stats_list = []
        for _v, vals in enumerate(list_of_other_arrays):
            # we need to create a list analogous to the_stats that corresponds to 
            # the original list that was input
            the_stats_list.append(np.array(stats_list_t[_v]).T)
    else:
        the_stats_list = None

    # ok, send both the list of binned array values and the stats for each bin back
    return array_binned, the_stats, array_binned_list, the_stats_list




def get_sigma(p_value):
    # the special.erfc(x) is the complementary error function, where x = sigma/sqrt(2)
    # i.e. the table of p-value-to-sigma at 
    # https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_tolerance_intervals 
    # is actually a table of erf(x), erfc(x), 1./erfc(x) for sigma values from 1 to 6
    # and special.erfcinv(alpha)*np.sqrt(2.) will return the significance level in sigma.
    # 
    # however, WARNING - this sigma is only valid assuming the distributions are Normal
    # which a lot of this utils file assumes they aren't.
    # so use as guidance but not gospel, and very much with caution.
    return special.erfcinv(p_value)*np.sqrt(2.)




"""ks test"""


import numpy as np
from scipy import special
from scipy.stats import kstwobign


def ks_weighted(arr1_all, arr2_all, w1_all, w2_all, return_dist=False):

    '''
    Given 2 arrays and their weights, returns Kolmogorov-Smirnov statistic and significance.

    This differs from the usual K-S test in that it computes a weighted K-S statistic and
    assumes the size of each sample is equal to the sum of the weights, not the length of
    the array. It's not exactly standard statistical practice to do this, so use with 
    caution, but it doesn't seem like a completely ridiculous idea, either.

        Parameters:
            arr1_all   (array): a data sample with values to be weighted by w1_all
            arr2_all   (array): a data sample with values to be weighted by w2_all
            w1_all     (array): weights for arr1_all
            w2_all     (array): weights for arr2_all
            return_dist (bool): True if the array of all K-S distances should be returned,
                                default is False (mostly only useful for debugging)

            Note: the arrays should be np.array() but other data types based on that should
                  work too, e.g. pd.Series, Astropy Table columns, etc. -- but if you get
                  an error on those, wrap your inputs with np.array().

        Returns:
            ks:     the weighted, 2-sided K-S statistic
            p_ks:   the p-value based on the weighted K-S statistic
            sig_ks: the significance level (in sigma) assuming p-values are distributed Normally

            if return_dist == True, also:
                dist_arr: array of KS-distances in raw format (sorted by increasing data sample value)
                          seriously this is not useful statistically, it's just for debugging

    '''

    # drop dead weight
    arr1 = np.array(arr1_all[w1_all > 0.0])
    arr2 = np.array(arr2_all[w2_all > 0.0])
    w1   = np.array(  w1_all[w1_all > 0.0])
    w2   = np.array(  w2_all[w2_all > 0.0])

    # get effective lengths of the weighted arrays
    n1 = np.sum(w1)
    n2 = np.sum(w2)
 
    # this is used below in the k-s calculation
    # (weighted sample sizes)
    ct = np.sqrt((n1+n2)/(n1*n2))

    # we want to sort the arrays, and the weights
    i1 = arr1.argsort()
    i2 = arr2.argsort()

    # sort arrays and weights in increasing order
    arr1_s = np.array(arr1[i1])
    w1_s   = np.array(  w1[i1])
    arr2_s = np.array(arr2[i2])
    w2_s   = np.array(  w2[i2])

    # make combined arrays but track which element comes from what, then sort them again
    both   = np.concatenate([arr1_s, arr2_s])
    both_w = np.concatenate([  w1_s,   w2_s])
    track  = np.concatenate([np.zeros(len(arr1_s), dtype=int), np.ones(len(arr2_s), dtype=int)])

    i_both   = both.argsort()
    both_s   = np.array(  both[i_both])
    both_w_s = np.array(both_w[i_both])
    track_s  = np.array( track[i_both])

    # go through array, once, computing the distance as we go, and track the max distance between cumulative curves
    # (which are both stored in the same array)
    # both cumulative curves start at 0 so the distance starts at 0
    # also cumulative curves always increase
    the_dist = 0.0
    dist_arr = np.zeros_like(both_s)
    max_dist = 0.0
    for j, this_which in enumerate(track_s):
        # the key here is the distance between curves goes up if array A has a new value,
        # and then if B has a new value that curve increases too so the curves get closer together
        # (the distance goes down).
        # it doesn't matter which is curve A and which is curve B, just that one increments 
        # and the other decrements.
        # if we were doing a regular K-S without weights, each new value for a given array changes
        # the distance between curves by 1 count. 
        # (with weighted, it only changes the distance by that object's weight.)
        # And also, these are cumulative curves, so each curve is divided by the total counts in that array
        # (which in the weighted case means the sum of the weights)
        # as a check, the distances should start at 0 and end at 0 (because the cumulative fractional
        # histograms both start at 0.0 and end at 1.0)
        if this_which == 0:
            the_dist += both_w_s[j]/n1
        else:
            the_dist -= both_w_s[j]/n2

        dist_arr[j] = the_dist
        if np.abs(the_dist) > max_dist:
            max_dist = np.abs(the_dist) 

    # the max dist over the whole cumulative curves is the K-S distance
    ks = max_dist
    # p-value (which also cares about the sample sizes)
    p_ks   = special.kolmogorov(float(ks)/float(ct))
    # scipy.stats.ks_2samp uses this instead?
    p_ksalt   = kstwobign.sf(((1./ct) + 0.12 + (0.11 * ct)) * ks)
    #print(p_ksalt)

    # what's the significance assuming a normal distribution? (1 = 1 sigma, 2. = 2 sigma, 3. = 3 sigma result etc.)
    sig_ks = special.erfcinv(p_ks)*np.sqrt(2.)



    if return_dist:
        return ks, p_ks, sig_ks, dist_arr
    else: 
        return ks, p_ks, sig_ks




# I don't really trust this one as you can get a different result with ks_w(x, y) and ks_w(y, x) which is NOT right
# Though note even the built-in scipy one seems to potentially have this problem because of how it sorts and how it computes distances
# so... I'm basically making an executive decision that it shouldn't matter and using the above
def ks_weighted_old(arr1_all, arr2_all, w1_all, w2_all, return_dist=False):

    # use with e.g.
    # print("K-S %.2e, p-value %.2e, i.e. %.1f sigma" % (ks, p_ks, sig_ks))

    # drop dead weight
    arr1 = arr1_all[w1_all > 0.0]
    arr2 = arr2_all[w2_all > 0.0]
    w1   =   w1_all[w1_all > 0.0]
    w2   =   w2_all[w2_all > 0.0]
    
    n1 = np.sum(w1)
    n2 = np.sum(w2)
    # this is used below in the k-s calculation
    # (weighted sample sizes)
    ct = np.sqrt((n1+n2)/(n1*n2))

    '''
    i1 = arr1.argsort()
    i2 = arr2.argsort()

    # sort arrays and weights in increasing order
    arr1_s = np.array(arr1[i1])
    w1_s   = np.array(  w1[i1])
    arr2_s = np.array(arr2[i2])
    w2_s   = np.array(  w2[i2])
    '''
    
    '''
    # from https://stackoverflow.com/a/40059727
    # uh, except it doesn't work? Dunno, but it's giving the wrong answers
    # I think this is because the answer on that page assumes, effectively, that you're already binned?
    # not sure, life's too short, my way takes 11 ms so wtf stop worrying about it
    data = np.concatenate([arr1_s, arr2_s])
    cwei1 = np.hstack([0, np.cumsum(w1_s)/sum(w1_s)])
    cwei2 = np.hstack([0, np.cumsum(w2_s)/sum(w2_s)])
    cdf1we = cwei1[[np.searchsorted(arr1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(arr2, data, side='right')]]
    '''

    data = np.concatenate([arr1, arr2])

    
    #bins = np.linspace(np.min(data), np.max(data), 10*len(data))
    # if your data has ridiculous outliers this might need to be refined
    n_bins = 10*len(data)
    # histograms
    # this is where this is going wrong. You shouldn't have to bin anything!
    h1 = np.histogram(arr1, weights=w1, bins=n_bins)
    h2 = np.histogram(arr2, weights=w2, bins=n_bins)
    # cumulative + normalized
    cdf1we = np.hstack([0.0, np.cumsum(h1[0])/sum(h1[0])])
    cdf2we = np.hstack([0.0, np.cumsum(h2[0])/sum(h2[0])])

    # K-S distance
    ks = np.max(np.abs(cdf1we - cdf2we))
    # p-value
    p_ks   = special.kolmogorov(ks/ct)
    # scipy.stats.ks_2samp uses this instead?
    p_ksalt   = kstwobign.sf(((1./ct) + 0.12 + (0.11 * ct)) * ks)
    print(p_ksalt)
    # what's the significance assuming a normal distribution? (1 = 1 sigma, 2. = 2 sigma, 3. = 3 sigma result etc.)
    sig_ks = special.erfcinv(p_ks)*np.sqrt(2.)

    if return_dist:
        return ks, p_ks, sig_ks, np.abs(cdf1we - cdf2we)
    else: 
        return ks, p_ks, sig_ks





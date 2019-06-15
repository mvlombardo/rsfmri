"""
Compute multi-scale entropy

Yang, A. C., Hong, C.-J., Liou, Y.-J., Huang, K.-L., Huang, C.-C., Liu, M.-E.,
Lo, M.-T., Huang, N. E., Peng, C.-K., Lin, C.-P., Tsai, S.-J. (2015). Decreased
resting-state brain activity complexity in schizophrenia characterized by both
increased regularity and randomness. Human Brain Mapping, 36, 2174-2186.


Example usage for running on parcellated data

imgfile=~/data/rest_pp01.nii.gz
maskfile=~/data/mask01.nii.gz
atlasfile=~/data/MMP_HCP01.nii.gz
outfile=~/data/rest_pp01

python msent.py -i $imgfile --mask $maskfile -o $outname -a $atlasfile -m 2 -r 0.15 -s 5


Example usage for running on voxel-wise data

imgfile=~/data/rest_pp01.nii.gz
maskfile=~/data/mask01.nii.gz
outfile=~/data/rest_pp01

python msent.py -i $imgfile --mask $maskfile -o $outname -m 2 -r 0.15 -s 5

Works with python 3.6
requires sampen library
"""

# import libraries
import numpy as np
import pandas as pd
# import scipy as sp
import sampen as se
import nibabel as nib
from optparse import OptionParser
# from numba import jit


# function to parse input arguments
def parse_args():
    """
    Parse arguments.
    """

    parser=OptionParser()

    parser.add_option('-i',"--input", \
                      dest='imgfile', \
                      help="Filename of the input rsfMRI 4D time-series dataset", \
                      default=None)
    parser.add_option('-o',"--output", \
                      dest='outname', \
                      help="Output filename prefix", \
                      default=None)
    parser.add_option('',"--mask", \
                      dest='maskfile', \
                      help="Filename of brain mask", \
                      default=None)
    parser.add_option('-a',"--atlas", \
                      dest='atlasfile', \
                      help="Filename of atlas file to use if parcellating data", \
                      default=None)
    parser.add_option('-v',"--verbose", \
                      dest='verbose', \
                      action='store_true', \
                      help="Use if you want verbose feedback while script runs.", \
                      default=False)
    parser.add_option('-m',"--mparam", \
                      dest='m',\
                      help="m parameter", \
                      default=2)
    parser.add_option('-r',"--rparam", \
                      dest='r',\
                      help="r parameter", \
                      default=0.15)
    parser.add_option('-s',"--scales", \
                      dest='scales',\
                      help="Number of scales", \
                      default=5)

    (options,args) = parser.parse_args()

    return(options)


# function to load data
def load_data(imgfile, maskfile, atlasfile, verbose):
    """
    Load data
    """

    if verbose:
        print("Loading data")

    # load main 4D time-series data
    img = nib.load(imgfile)
    imgdata = img.get_data()

    # get voxel size
    voxsize = np.array(img.get_header().get_zooms()[0:3]).reshape(1,3)

    # find how many timepoints there are
    numtps = imgdata.shape[-1]

    # load mask image
    maskdata = nib.load(maskfile)

    # make mask a boolean array
    mask = maskdata.get_data().astype(bool)

    # get number of voxels
    numvoxels = np.nansum(mask)

    if atlasfile is not None:
        # load atlas image
        atlas = nib.load(atlasfile)
        atlasdata = atlas.get_data()

        # find unique region labels in atlas
        regions = np.unique(atlasdata)

        # exclude region with label 0
        regions = regions[regions!=0]

        # count how many unique regions there are
        nregions = len(regions)
    else:
        nregions = numvoxels
        regions = None
        atlasdata = None

    # save results into datadict
    datadict = {"imgdata":imgdata, \
                "mask":mask, \
                "atlasdata":atlasdata, \
                "regions":regions, \
                "nregions":nregions, \
                "numtps":numtps, \
                "voxsize":voxsize}

    return(datadict)


# function for parcellating data
def parcellate_data(imgdata, atlasdata, numtps, regions, nregions, verbose):
    """
    Parcellate data
    """

    if verbose:
        print("Parcellating data")

    # pre-allocate parc_data array
    parc_data = np.zeros((nregions,numtps))

    # loop over regions
    for reg_idx, region in enumerate(regions):
        # make an ROI mask
        roimask = atlasdata==region

        #loop over time points
        for ivol in range(0,numtps):
            # grab specific 3D time point volume
            tmp_data = imgdata[:, :, :, ivol]

            # extract voxels within the ROI
            roidata = tmp_data[roimask]

            # calculate the mean for that ROI and timepoint
            parc_data[reg_idx, ivol] = np.nanmean(roidata)

    return(parc_data)


# function to standardize data
def standardize_data(data, n, verbose):
    """
    Standardize data
    """

    if verbose:
        print("Standardizing data")

    # compute the mean over time
    time_mean = np.nanmean(data, axis = 1).reshape(n,1)

    # compute the standard deviation over time
    time_std = np.nanstd(data, axis = 1).reshape(n,1)

    # take data and minus off the mean, then divide by standard deviation
    zdata = (data - time_mean)/time_std

    return(zdata)


# function to coarsegrain the time-series
def coarsegraining(data, scale):
    """
    Coarsegrain a time-series by scale
    """

    # length of data
    n = len(data)

    # block length
    block = np.array(np.fix(n/scale), dtype = int)

    # pre-allocate results array
    result = np.zeros((block,1))

    for block_idx, idx in enumerate(np.arange(1,block+1, dtype=int)):
        result[block_idx] = 0

        for factor_idx, factor in enumerate(np.arange(1,scale+1, dtype=int)):
            idx2use = factor_idx + (block_idx*scale)
            result[block_idx] = result[block_idx] + data[idx2use]

        result[block_idx] = result[block_idx]/scale

    return(result)


# function to compute sample entropy
def sample_entropy(data, m, r):
    """
    Compute sample entropy on a time-series.
    m = maximum template length
    r = matching tolerance
    """

    n = len(data)
    lastrun = np.zeros((1,n))
    run = np.zeros((1,n))

    A = np.zeros((m,1), dtype=int)
    B = np.zeros((m,1), dtype=int)
    p = np.zeros((m,1))
    entropy = np.zeros((m,1))

    for idx_py, idx_mat in enumerate(np.arange(1,n, dtype=int)):
        nj = n - idx_mat
        y1 = data[idx_mat-1]

        for jj_py, jj_mat in enumerate(np.arange(1,nj+1, dtype=int)):
            j_py = jj_py + idx_py
            j_mat = jj_mat + idx_mat

            if np.abs(data[j_mat-1]-y1) < r:
                run[:,(jj_mat-1)] = lastrun[:,(jj_mat-1)] + 1
                M1 = np.min(np.concatenate(([m],run[:,(jj_mat-1)])))

                for mi_py, mi_mat in enumerate(np.arange(1,M1+1,dtype=int)):
                    A[(mi_mat-1)] = A[(mi_mat-1)] + 1

                    if j_mat<n:
                        B[(mi_mat-1)] = B[(mi_mat-1)] + 1

            else:
                run[:,(jj_mat-1)] = 0

        for j1_py, j1_mat in enumerate(np.arange(1,nj+1, dtype=int)):
            lastrun[:,(j1_mat-1)] = run[:,(j1_mat-1)]

    N = np.zeros((1,1))
    N[0] = (n*(n-1))/2
    B = np.concatenate((N,B[0:(m-1)])).reshape(A.shape)
    p = np.divide(A,B)
    entropy = -np.log(p)

    return(entropy,A,B)




# function to write out parcellated image
def save_mse_parc_img(result, atlasfile, regions, outname, verbose):
    """
    Save parcellated image
    """

    if verbose:
        print("Saving parcellated mse image to disk")

    atlas = nib.load(atlasfile)
    atlasdata = atlas.get_data()

    # pre-allocate 4d result array
    nscales = result.shape[1]
    mse_dim = list(atlasdata.shape)
    mse_dim.append(nscales)
    mse_dim = tuple(mse_dim)
    parc_img = np.zeros(mse_dim)

    for scale in np.arange(nscales):
        tmp_img = np.zeros(atlasdata.shape)

        # loop over regions
        for reg_idx, region in enumerate(regions):
            # make an ROI mask
            roimask = atlasdata==region

            # fill in region with value
            tmp_img[roimask] = result[reg_idx,scale]

        parc_img[:,:,:,scale] = tmp_img

    # save image to disk
    nib.save(nib.Nifti1Image(parc_img, atlas.get_affine()), outname)


# function to write out parcellated data to a csv
def save_mse_parc_csv(result, outname, regions):
    """
    Save csv for parcellated data
    """

    data2use = {"region":regions}
    tmp_df1 = pd.DataFrame(data2use)
    tmp_df2 = pd.DataFrame(result)
    tmp_df2_colnames = list(tmp_df2.columns.values)
    for idx, cname in enumerate(tmp_df2_colnames):
        tmp_df2_colnames[idx] = "scale_%d" % idx
    tmp_df2.columns = tmp_df2_colnames
    res_df = tmp_df1.join(tmp_df2)
    export_csv = res_df.to_csv(outname, index = None, header = True)



# main function to run on parcellated data
def mse_parc(imgfile, maskfile, atlasfile, outname, m, r, scales, verbose):
    """
    Main function to run all steps for analysis on parcellated data
    """

    # load data
    datadict = load_data(imgfile, maskfile, atlasfile, verbose)
    imgdata = datadict["imgdata"]
    mask = datadict["mask"]
    atlasdata = datadict["atlasdata"]
    regions = datadict["regions"]
    nregions = datadict["nregions"]
    numtps = datadict["numtps"]

    # parcellate data
    parc_data = parcellate_data(imgdata, atlasdata, numtps, regions, nregions, \
                                verbose)
    # standardize data
    parc_zdata = standardize_data(parc_data, nregions, verbose)

    # pre-allocate results array
    nscales = len(scales)
    result = np.zeros((nregions,nscales))

    # loop over regions
    for region in range(0,nregions):
        if verbose:
            print("Working on region %d of %d" % (region+1,nregions))

        for scale_idx, scale in enumerate(scales):
            # do coarsegraining
            s = coarsegraining(parc_zdata[region,:],scale)

            # # compute sample entropy
            # [sampe, A, B] = sample_entropy(s,m+1,r)
            #
            # # grab sample entropy at specific scale
            # result[region, scale_idx] = sampe[m]

            # compute sample entropy
            sampe = se.sampen2(s, mm=m+1, r=r, normalize=False)

            # grab sample entropy at specific scale
            result[region, scale_idx] = sampe[m][1]

    # save parcellated degree centrality image to disk
    if outname is not None:
        fname2save = "%s_mse_parc.nii.gz" % outname
        save_mse_parc_img(result, atlasfile, regions, fname2save, verbose)
        fname2save = "%s_mse_parc.csv" % outname
        save_mse_parc_csv(result, fname2save, regions)

    return(result)


# function to write out voxel-wise image
def save_mse_vox_img(result, imgfile, outname, verbose):
    """
    Save voxel-wise mse image
    """

    if verbose:
        print("Saving voxel-wise mse image")

    img = nib.load(imgfile)
    nib.save(nib.Nifti1Image(result, img.get_affine()), outname)


# main function to run analysis on voxel-wise data
def mse_img(imgfile, maskfile, outname, m, r, scales, verbose):
    """
    Main function to run all steps for analysis on voxel-wise data
    """

    if verbose:
        print("Computing voxel-wise mse map")

    # load data
    atlasfile = None
    datadict = load_data(imgfile, maskfile, atlasfile, verbose)
    imgdata = datadict["imgdata"]
    mask = datadict["mask"]
    numvoxels = datadict["nregions"]
    numtps = datadict["numtps"]
    voxsize = datadict["voxsize"]

    # grab indices of brain voxels within mask
    indices = np.transpose(np.nonzero(mask))
    imgts = imgdata[indices[:,0], indices[:,1], indices[:,2]]

    # standardize data
    imgts = standardize_data(imgts, numvoxels, verbose)

    # pre-allocate 4d result array
    nscales = len(scales)
    mse_dim = list(mask.shape)
    mse_dim.append(nscales)
    mse_dim = tuple(mse_dim)
    result = np.zeros(mse_dim)

    # loop over scales
    for scale_idx, scale in enumerate(scales):
        if verbose:
            print("Working on scale %d" % scale)

        tmp_img = np.zeros(mask.shape)

        # loop over voxels
        for basevoxel in range(0, numvoxels):
            if verbose:
                print("Working on %d voxel of %d voxels" % (basevoxel+1,numvoxels))

            #Get x,y,z coords for the voxel
            x,y,z = indices[basevoxel,:]

            # grab specific voxel's time-series
            ts = np.array(imgts[basevoxel,:]).reshape((numtps,1))

            # do coarsegraining
            s = coarsegraining(ts,scale)

            # # compute sample entropy
            # [sampe, A, B] = sample_entropy(s,m+1,r)
            #
            # # grab sample entropy at specific scale
            # tmp_img[x,y,z] = sampe[m]

            # compute sample entropy
            try:
                sampe = se.sampen2(s, mm=m+1, r=r, normalize=False)
                # grab sample entropy at specific scale
                tmp_img[x,y,z] = sampe[m][1]
            except ZeroDivisionError:
                # grab sample entropy at specific scale
                tmp_img[x,y,z] = float("Inf")

        result[:,:,:,scale_idx] = tmp_img

    # save result to disk
    if outname is not None:
        fname2save = "%s_mse.nii.gz" % outname
        save_mse_vox_img(result, imgfile, fname2save, verbose)

    return(result)


# boilerplate code to call main code for executing
if __name__ == '__main__':

    # parse arguments
    opts = parse_args()

    # main 4D time-series
    imgfile = opts.imgfile

    # mask file
    maskfile = opts.maskfile

    # output file
    outname = opts.outname

    # atlas file
    atlasfile = opts.atlasfile

    # m parameter
    m = np.array(opts.m, dtype = int)

    # r parameter
    r = np.array(opts.r, dtype = float)

    # scales
    scales = np.arange(1,np.array(opts.scales,dtype=int)+1)

    # verbose flag
    verbose = opts.verbose

    # run analysis using atlas file or not
    if atlasfile is None:
        result = mse_img(imgfile, maskfile, outname, m, r, scales, verbose)
    elif atlasfile is not None:
        result = mse_parc(imgfile, maskfile, atlasfile, outname, m, r, scales, verbose)

    print("Done")

"""
Compute weighted global brain connectivity (wGBC) on a 4D rsfMRI time-series
data. Code will work to compute this metric on all voxels or within parcels from
a pre-specified atlas.

Cole, M. W., Pathak, S.,  & Schneider, W. (2010). Identifying the brain's most
globally connected regions. Neuroimage, 49, 3132-3148.

Example usage for running on voxel-wise data

imgfile=~/data/rest_pp01.nii.gz
maskfile=~/data/mask01.nii.gz
outfile=~/data/vox_wgbc01.nii.gz
python wgbc.py -i $imgfile -m $maskfile -o $outfile

Example usage for running on parcellated data

imgfile=~/data/rest_pp01.nii.gz
maskfile=~/data/mask01.nii.gz
atlasfile=~/data/MMP_HCP01.nii.gz
outfile=~/data/parc_wgbc01.nii.gz
python wgbc.py -i $imgfile -m $maskfile -a $atlasfile -o $outfile

Credit to Marco Pagani and Alessandro Gozzi, for a previous version which this
code partly based upon.
"""

# import libraries
import sys
import numpy as np
import nibabel as nib
import math
from optparse import OptionParser
import datetime


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
                      help="Output filename", \
                      default=None)
    parser.add_option('-m',"--mask", \
                      dest='maskfile', \
                      help="Filename of brain mask", \
                      default=None)
    parser.add_option('-a',"--atlas", \
                      dest='atlasfile', \
                      help="Filename of atlas file to use if parcellating data", \
                      default=None)
    parser.add_option('-r',"--radius", \
                      dest='radius', \
                      help="""Radius. If not provided, the script computes
                      global brain connectivity, i.e. it takes one voxel at a
                      time, calculates its correlation voxel with every other
                      voxel in the mask (itself excluded), converts these values
                      to Fisher\'s z and averages them. If the radius is a
                      negative value, the script computes local connectivity,
                      i.e. it takes into account only voxels whose centers are
                      within the absolute value of the  radius. If it is a
                      positive value, it takes into account only voxels in the
                      mask whose centers are further away than the given radius.""", \
                      default=None)

    (options,args) = parser.parse_args()

    return(options)


# function to load data
def load_data(imgfile, maskfile, atlasfile=None):
    """
    Load data
    """

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


# function to standardize data
def standardize_data(data, n):
    """
    Standardize data
    """

    print("Standardizing data")

    # compute the mean over time
    time_mean = np.nanmean(data, axis = 1).reshape(n,1)

    # compute the standard deviation over time
    time_std = np.nanstd(data, axis = 1).reshape(n,1)

    # take  data and minus off the mean, then divide by standard deviation
    zdata = (data - time_mean)/time_std

    return(zdata)


# function for parcellating data
def parcellate_data(imgdata, atlasdata, numtps, regions, nregions):
    """
    Parcellate data
    """

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


# function to compute weighted global brain connectivity for every voxel
def compute_wgbc_img(imgts, mask, radius, indices, numvoxels, numtps):
    """
    Compute voxel-wise weighted global brain connectivity image
    """

    result = np.zeros(mask.shape)

    for basevoxel in range(0, numvoxels):

        print("Working on %d voxel of %d voxels" % (basevoxel,numvoxels))

        x,y,z = indices[basevoxel,:]

        if radius is None:
            subset = np.arange(numvoxels) != basevoxel

        else:
            distance = np.nansum(np.square(((indices-indices[basevoxel,:]) * voxsize)), axis=1)

            if radius > 0:
                subset = distance > radius**2

            else:
                subset = distance <= radius**2
                subset[basevoxel] = False

        rvalues = np.dot(imgts, imgts[basevoxel].T) / numtps

        result[x,y,z] = np.nanmean(rvalues)

    return(result)


# function to compute weighted global brain connectivity for parcellation
def compute_wgbc_parc(parc_zdata, nregions):
    """
    Compute weighted global brain connectivity on parcellated data
    """

    print("Computing wGBC on parcels")

    # compute correlation matrix
    corr_mat = np.corrcoef(parc_zdata)

    # make a mask of the regions to use excluding the seed region itself
    corr_mask = np.eye(nregions).astype(bool)==False

    # pre-allocate result array in memory
    result = np.zeros((nregions,1))

    # loop over regions
    for region in range(0,nregions):

        # compute the mean correlation for seed region of interest
        result[region,] = np.nanmean(corr_mat[region,corr_mask[region,:]])

    return(result)

# function to write out weighted global brain connectivity parcellated image
def save_wgbc_parc_img(result, mask, atlasfile, regions, outname):
    """
    Save parcellated wGBC image
    """

    print("Saving parcellated wGBC image to disk")

    atlas = nib.load(atlasfile)
    atlasdata = atlas.get_data()
    parc_img = np.zeros(atlasdata.shape)

    # loop over regions
    for reg_idx, region in enumerate(regions):

        # make an ROI mask
        roimask = atlasdata==region

        # fill in region with wdc value
        parc_img[roimask] = result[reg_idx]

    # save image to disk
    nib.save(nib.Nifti1Image(parc_img, atlas.get_affine()), outname)


# function to write out weighted global brain connectivity voxel-wise image
def save_wgbc_vox_img(result, imgfile, outname):
    """
    Save voxel-wise wGBC image
    """

    img = nib.load(imgfile)
    nib.save(nib.Nifti1Image(result, img.get_affine()), outname)


# main function to run the weighted global brain connectivity analysis on parcellated data
def dcbc_parc(imgfile, maskfile, atlasfile, outname):
    """
    Main function to run all steps for analysis on parcellated data
    """

    # load data
    datadict = load_data(imgfile, maskfile, atlasfile)
    imgdata = datadict["imgdata"]
    mask = datadict["mask"]
    atlasdata = datadict["atlasdata"]
    regions = datadict["regions"]
    nregions = datadict["nregions"]
    numtps = datadict["numtps"]

    # parcellate data
    parc_data = parcellate_data(imgdata, atlasdata, numtps, regions, nregions)

    # standardize data
    parc_zdata = standardize_data(parc_data, nregions)

    # compute weighted global brain connectivity  on parcellated data
    result = compute_wgbc_parc(parc_zdata, nregions)

    # save parcellated wgbc image to disk
    if outname is not None:
        save_wgbc_parc_img(result, mask, atlasfile, regions, outname)

    return(result)


# main function to run weighted global brain connectivity analysis on voxel-wise data
def dcbc_img(imgfile, maskfile, radius, outname):
    """
    Main function to run all steps for analysis on voxel-wise data
    """

    # load data
    datadict = load_data(imgfile, maskfile, atlasfile=None)
    imgdata = datadict["imgdata"]
    mask = datadict["mask"]
    numvoxels = datadict["nregions"]
    numtps = datadict["numtps"]
    voxsize = datadict["voxsize"]

    indices = np.transpose(np.nonzero(mask))

    imgts = imgdata[indices[:,0], indices[:,1], indices[:,2]]

    imgts = standardize_data(imgts, numvoxels)

    result = compute_wgbc_img(imgts, mask, radius, indices, numvoxels, numtps)

    # save result to disk
    if outname is not None:
        save_wgbc_vox_img(result, imgfile, outname)

    return(result)


# boilerplate code to call main code for executing
if __name__ == '__main__':

    # parse arguments
    opts = parse_args()

    # main 4D time-series
    imgfile = opts.imgfile

    # mask file
    maskfile = opts.maskfile

    # atlas file
    atlasfile = opts.atlasfile

    # output file
    outname = opts.outname

    # radius
    radius = opts.radius

    if atlasfile is None:
        result = dcbc_img(imgfile, maskfile, radius, outname)

    elif atlasfile is not None:
        result = dcbc_parc(imgfile, maskfile, atlasfile, outname)

"""
Compute intrinsic neural timescale (intsc) from time-series data.

Watanabe, T., Rees, G., & Masuda, N. (2019). Atypical intrinsic neural timescale in autism. eLife, 8, e42256.


Example usage for running on parcellated data

imgfile=~/data/rest_pp01.nii.gz
maskfile=~/data/mask01.nii.gz
atlasfile=~/data/MMP_HCP01.nii.gz
outfile=~/data/rest_pp01

python intsc.py -i $imgfile -m $maskfile -o $outname -a $atlasfile --tr 1.302 --nlags 20


Example usage for running on voxel-wise data

imgfile=~/data/rest_pp01.nii.gz
maskfile=~/data/mask01.nii.gz
outfile=~/data/rest_pp01

python intsc.py -i $imgfile -m $maskfile -o $outname --tr 1.302 --nlags 20
"""

# import libraries
import numpy as np
import pandas as pd
import scipy as sp
import math
import nibabel as nib
from optparse import OptionParser



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
    parser.add_option('-m',"--mask", \
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
    parser.add_option('',"--tr", \
                      dest='tr',\
                      help="TR in seconds", \
                      default=None)
    parser.add_option('',"--nlags", \
                      dest='nlags',\
                      help="number of lags", \
                      default=20)

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


# function to find next power of 2
def nextpow2(x):
    """
    Get next power of 2
    """

    result = math.ceil(sp.log2(np.abs(x)))
    result = int(result)
    return(result)


# function to mean center data
def mean_center(data, n, verbose):
    """
    Mean center data
    """

    if verbose:
        print("Mean centering data")

    # compute the mean over time
    time_mean = np.nanmean(data, axis = 1).reshape(n,1)

    # take data and minus off the mean
    result = data - time_mean

    return(result)


# main function for computing intrinsic neural timescale
def compute_intsc(ts, numtps, nlags, tr):
    """
    Main function for computing intrinsic neural timescale
    """

    # find next power of 2
    p2use = nextpow2(numtps)

    # figure out n for FFT
    nFFT = 2**(p2use+1)

    # FFT data
    f = sp.fft(ts, n=nFFT)

    # multiply FFT result by its complex conjugate
    f = np.multiply(f, np.conjugate(f))

    # inverse FFT
    acf = sp.ifft(f)

    # grab nlags
    acf = acf[0,0:nlags+1]

    # normalize
    acf = np.divide(acf,acf[0])

    # convert to real numbers
    acf = np.real(acf)

    # sum positive acf
    positive_acf = np.nansum(acf[acf>0])-acf[0]

    # multiply by TR
    intsc = positive_acf * tr

    return(intsc)


# function to write out intsc parcellated image
def save_intsc_parc_img(result, atlasfile, regions, outname, verbose):
    """
    Save parcellated intsc image
    """

    if verbose:
        print("Saving parcellated intsc image to disk")

    atlas = nib.load(atlasfile)
    atlasdata = atlas.get_data()
    parc_img = np.zeros(atlasdata.shape)

    # loop over regions
    for reg_idx, region in enumerate(regions):
        # make an ROI mask
        roimask = atlasdata==region

        # fill in region with dc value
        parc_img[roimask] = result[reg_idx]

    # save image to disk
    nib.save(nib.Nifti1Image(parc_img, atlas.get_affine()), outname)


# function to write out intsc parcellated data to a csv
def save_intsc_parc_csv(result, outname, regions):
    """
    Save csv for parcellated intsc
    """

    data2use = {"region":regions, "dc":result.reshape(len(regions))}
    res_df = pd.DataFrame(data2use)
    export_csv = res_df.to_csv(outname, index = None, header = True)


# main function to run on parcellated data
def intsc_parc(imgfile, maskfile, atlasfile, outname, nlags, tr, verbose):
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

    # mean center data
    parc_data_mc = mean_center(parc_data, nregions, verbose)

    result = np.zeros((nregions,1))
    # loop over regions
    for region in range(0,nregions):
        if verbose:
            print("Working on region %d of %d" % (region+1,nregions))

        ts = np.array(parc_data_mc[region,:]).reshape((1,numtps))
        result[region,] = compute_intsc(ts, numtps, nlags, tr)

    # save parcellated degree centrality image to disk
    if outname is not None:
        fname2save = "%s_intsc_parc.nii.gz" % outname
        save_intsc_parc_img(result, atlasfile, regions, fname2save, verbose)
        fname2save = "%s_intsc_parc.csv" % outname
        save_intsc_parc_csv(result, fname2save, regions)

    return(result)


# function to write out degree centrality voxel-wise image
def save_intsc_vox_img(result, imgfile, outname, verbose):
    """
    Save voxel-wise intsc image
    """

    if verbose:
        print("Saving voxel-wise intsc image")

    img = nib.load(imgfile)
    nib.save(nib.Nifti1Image(result, img.get_affine()), outname)


# main function to run intsc analysis on voxel-wise data
def intsc_img(imgfile, maskfile, outname, nlags, tr, verbose):
    """
    Main function to run all steps for analysis on voxel-wise data
    """

    if verbose:
        print("Computing voxel-wise intsc map")

    # load data
    atlasfile = None
    datadict = load_data(imgfile, maskfile, atlasfile, verbose)
    imgdata = datadict["imgdata"]
    mask = datadict["mask"]
    numvoxels = datadict["nregions"]
    numtps = datadict["numtps"]
    voxsize = datadict["voxsize"]

    indices = np.transpose(np.nonzero(mask))
    imgts = imgdata[indices[:,0], indices[:,1], indices[:,2]]

    imgts = mean_center(imgts, numvoxels, verbose)

    result = np.zeros(mask.shape)

    for basevoxel in range(0, numvoxels):
        if verbose:
            print("Working on %d voxel of %d voxels" % (basevoxel+1,numvoxels))

        x,y,z = indices[basevoxel,:]
        ts = np.array(imgts[basevoxel,:]).reshape((1,numtps))

        result[x,y,z] = compute_intsc(ts, numtps, nlags, tr)

    # save result to disk
    if outname is not None:
        fname2save = "%s_intsc.nii.gz" % outname
        save_intsc_vox_img(result, imgfile, fname2save, verbose)

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

    # tr
    tr = np.array(opts.tr, dtype = float)

    # nlags
    nlags = np.array(opts.nlags, dtype = int)

    # verbose flag
    verbose = opts.verbose

    if atlasfile is None:
        result = intsc_img(imgfile, maskfile, outname, nlags, tr, verbose)
    elif atlasfile is not None:
        result = intsc_parc(imgfile, maskfile, atlasfile, outname, nlags, tr, verbose)

    print("Done")

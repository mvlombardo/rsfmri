"""
Compute degree centrality (unweighted or weighted) on a 4D rsfMRI time-series
dataset.

Buckner, R.L., Sepulcre, J., Talukdar, T., Krienen, F.M., Liu, H., Hedden, T.,
Andrews-Hanna, J.R., Sperling, R.A., Johnson, K.A., 2009. Cortical hubs revealed
by intrinsic functional connectivity: mapping, assessment of stability, and
relation to Alzheimer's disease. J Neurosci 29, 1860-1873.

Zuo, X.N., Ehmke, R., Mennes, M., Imperati, D., Castellanos, F.X., Sporns, O.,
Milham, M.P., 2012. Network Centrality in the Human Functional Connectome. Cereb
Cortex 22, 1862-1875.


Example usage for running on voxel-wise data

imgfile=~/data/rest_pp01.nii.gz
maskfile=~/data/mask01.nii.gz
outfile=~/data/vox_dc01

# unweighted with r > 0.25 threshold
thresh=0.25
python degcen.py -i $imgfile -m $maskfile -o $outfile -t $thresh

# weighted with r > 0.25 threshold
thresh=0.25
python degcen.py -i $imgfile -m $maskfile -o $outfile -t $thresh -w

# weighted with no threshold
thresh=0.25
python degcen.py -i $imgfile -m $maskfile -o $outfile -w


Example usage for running on parcellated data

imgfile=~/data/rest_pp01.nii.gz
maskfile=~/data/mask01.nii.gz
atlasfile=~/data/MMP_HCP01.nii.gz
outfile=~/data/parc_dc01

# unweighted with r > 0.25 threshold
thresh=0.25
python degcen.py -i $imgfile -m $maskfile -a $atlasfile -o $outfile -t $thresh

# weighted with r > 0.25 threshold
thresh=0.25
python degcen.py -i $imgfile -m $maskfile -a $atlasfile -o $outfile -t $thresh -w

# weighted with no threshold
python degcen.py -i $imgfile -m $maskfile -a $atlasfile -o $outfile -w
"""

# import libraries
import numpy as np
import pandas as pd
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
    parser.add_option('-w',"--weighted", \
                      dest='weighted', \
                      action='store_true', \
                      help="Use if you want to compute weighted degree centrality", \
                      default=False)
    parser.add_option('-m',"--mask", \
                      dest='maskfile', \
                      help="Filename of brain mask", \
                      default=None)
    parser.add_option('-a',"--atlas", \
                      dest='atlasfile', \
                      help="Filename of atlas file to use if parcellating data", \
                      default=None)
    parser.add_option('-t',"--threshold", \
                      dest='threshold', \
                      help="Correlation threshold to use if calculated unweighted degree.", \
                      default=None)
    parser.add_option('-v',"--verbose", \
                      dest='verbose', \
                      action='store_true', \
                      help="Use if you want verbose feedback while script runs.", \
                      default=False)

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


# function to compute degree centrality for every voxel
def compute_dc_img(imgts, mask, indices, numvoxels, numtps, threshold, weighted_flag, verbose):
    """
    Compute voxel-wise degree centrality map
    """

    if verbose:
        print("Computing voxel-wise degree centrality map")

    result = np.zeros(mask.shape)

    for basevoxel in range(0, numvoxels):
        if verbose:
            print("Working on %d voxel of %d voxels" % (basevoxel,numvoxels))

        x,y,z = indices[basevoxel,:]

        rvalues = np.dot(imgts, imgts[basevoxel].T) / numtps

        # compute weighted degree
        if weighted_flag:
            if threshold is None:
                result[x,y,z] = np.nansum(rvalues) - 1
            else:
                voxmask = rvalues > threshold
                result[x,y,z] = np.nansum(rvalues[voxmask]) -1
        # compute unweighted degree
        else:
            voxmask = rvalues > threshold
            result[x,y,z] = np.nansum(voxmask) - 1

    return(result)


# function to compute degree centrality for parcellation
def compute_dc_parc(parc_zdata, nregions, threshold, weighted_flag, verbose):
    """
    Compute degree centrality on parcellated data
    """

    if verbose:
        print("Computing degree centrality on parcels")

    # compute correlation matrix
    corr_mat = np.corrcoef(parc_zdata)

    # make a mask of the regions to use excluding the seed region itself
    corr_mask = np.eye(nregions).astype(bool)==False

    # pre-allocate result array in memory
    result = np.zeros((nregions,1))

    # loop over regions
    for region in range(0,nregions):
        if verbose:
            print("Working on region %d of %d" % (region,nregions))

        rvalues = corr_mat[region,corr_mask[region,:]]

        # compute weighted degree
        if weighted_flag:
            if threshold is None:
                # compute the sum of connection weights
                result[region,] = np.nansum(rvalues)
            else:
                # find regions above some threshold and then sum connection weights
                connection_mask = rvalues > threshold
                result[region,] = np.nansum(rvalues[connection_mask])
        # compute unweighted degree
        else:
            # sum up the number of connections with the seed that pass threshold
            connection_mask = rvalues > threshold
            result[region,] = np.nansum(connection_mask)

    return(result)

# function to write out degree centrality parcellated image
def save_dc_parc_img(result, mask, atlasfile, regions, outname, verbose):
    """
    Save parcellated degree centrality image
    """

    if verbose:
        print("Saving parcellated degree centrality image to disk")

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
    fname2save = "%s.nii.gz" % (outname)
    nib.save(nib.Nifti1Image(parc_img, atlas.get_affine()), fname2save)


# function to write out degree centrality parcellated data to a csv
def save_dc_parc_csv(result, outname, regions):
    """
    Save csv for parcellated degree centrality
    """

    # nreg = result.shape[0]
    # data2use = {"region":np.arange(nreg)+1, "dc":result.reshape(nreg)}
    data2use = {"region":regions, "dc":result.reshape(len(regions))}
    res_df = pd.DataFrame(data2use)
    fname2save = "%s.csv" % (outname)
    export_csv = res_df.to_csv(fname2save, index = None, header = True)


# function to write out degree centrality voxel-wise image
def save_dc_vox_img(result, imgfile, outname, verbose):
    """
    Save voxel-wise degree centrality image
    """

    if verbose:
        print("Saving voxel-wise degree centrality image")

    img = nib.load(imgfile)
    fname2save = "%s.nii.gz" % (outname)
    nib.save(nib.Nifti1Image(result, img.get_affine()), fname2save)


# main function to run the degree centrality analysis on parcellated data
def dc_parc(imgfile, maskfile, atlasfile, outname, threshold, weighted_flag, verbose):
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
    parc_data = parcellate_data(imgdata, atlasdata, numtps, regions, nregions, verbose)

    # standardize data
    parc_zdata = standardize_data(parc_data, nregions, verbose)

    # compute degree centrality on parcellated data
    result = compute_dc_parc(parc_zdata, nregions, threshold, weighted_flag, verbose)

    # save parcellated degree centrality image to disk
    if outname is not None:
        save_dc_parc_img(result, mask, atlasfile, regions, outname, verbose)
        save_dc_parc_csv(result, outname, regions)

    return(result)


# main function to run degree centrality analysis on voxel-wise data
def dc_img(imgfile, maskfile, outname, threshold, weighted_flag, verbose):
    """
    Main function to run all steps for analysis on voxel-wise data
    """

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

    imgts = standardize_data(imgts, numvoxels, verbose)

    result = compute_dc_img(imgts, mask, indices, numvoxels, numtps, threshold, weighted_flag, verbose)

    # save result to disk
    if outname is not None:
        save_dc_vox_img(result, imgfile, outname, verbose)

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

    # weighted flag
    weighted_flag = opts.weighted

    # verbose flag
    verbose = opts.verbose

    # threshold
    if opts.threshold is not None:
        threshold = np.array(opts.threshold, dtype = float)
    else:
        threshold = opts.threshold

    if atlasfile is None:
        result = dc_img(imgfile, maskfile, outname, threshold, weighted_flag, verbose)
    elif atlasfile is not None:
        result = dc_parc(imgfile, maskfile, atlasfile, outname, threshold, weighted_flag, verbose)

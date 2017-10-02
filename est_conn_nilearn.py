# est_conn_nilearn.py
"""
Script that uses nilearn tools to estimate connectivity matrix on rsfMRI data

python est_conn_nilearn.py -d rest_mefc.nii.gz -a HarvardOxford -o connmat_harvoxf.csv -p connmat_harvoxf.pdf
python est_conn_nilearn.py -d rest_mefc.nii.gz -a AAL -o connmat_aal.csv -p connmat_aal.pdf
python est_conn_nilearn.py -d rest_mefc.nii.gz -a HarvardOxford -o connmat_pc_harvoxf.csv -p connmat_pc_harvoxf.pdf --cest partialcorr
"""

# import modules
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from matplotlib import pyplot as plt
from optparse import OptionParser
import pandas as pd



# function to parse input arguments
def parse_args():
    """
    Parse arguments.
    """
    parser=OptionParser()
    parser.add_option('-d',"",dest='data',help="rsfMRI data ex: -d rest_mefc.nii.gz",default=None)
    parser.add_option('-a',"",dest='atlas_name',help="Atlas to use (HarvardOxford or AAL) ex: -a HarvardOxford",default="HarvardOxford")
    parser.add_option('-o',"",dest='csv2save',help="csv filename of file to save connectivity matrix to ex: -o connmat.csv",default=None)
    parser.add_option('-p',"",dest='pdf2save',help="PDF filename for plot to save ex: -p connplot.pdf",default=None)
    parser.add_option('-v',"",action="store_true",dest='verbose',help="turn on verbose output",default=False)
    parser.add_option('--cest',"",dest='cest',help="Connectivity estimator ex: --cest correlation",default="correlation")
    parser.add_option('--cmax',"",dest='cmax',help="Max value for color scaling plot ex: --cmax 0.5",default=0.8)
    parser.add_option('--cmin',"",dest='cmin',help="Min value for color scaling plot ex: --cmin -0.5",default=-0.5)
    parser.add_option('--cmap',"",dest='cmap',help="Colormap to use ex: --cmap RdBu_r",default="RdBu_r")
    parser.add_option('--fig_size',"",dest='fig_size',help="Size of figure to make ex: --fig_size '12,12'",default='12,12')
    (options, args) = parser.parse_args()
    return(options)


# function to load atlas
def get_atlas(atlas_name, verbose = False):
    """
    Get atlas from nilearn.
    Atlases are currently only HarvardOxford and AAL
    """

    if atlas_name == "HarvardOxford":
        dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    elif atlas_name == "AAL":
        dataset = datasets.fetch_atlas_aal()

    atlas_filename = dataset.maps
    labels = dataset.labels

    if verbose:
        print('Atlas ROIs are located in nifti image (4D) at: %s' %
              atlas_filename)  # 4D data

    return((atlas_filename, labels))


# function to grab time series from atlas and vectorize
def prepare_data(atlas_filename, fmri_filenames, standardize_arg = True,
    memory_arg = 'nilearn_cache', verbose_arg = 5):
    """
    Grab time series from atlas regions and vectorize.
    """

    masker = NiftiLabelsMasker(labels_img = atlas_filename,
        standardize = standardize_arg, memory = memory_arg,
        verbose = verbose_arg)

    time_series = masker.fit_transform(fmri_filenames)

    return((masker,time_series))

# function to estimate connectivity
def estimate_connectivity(time_series, measure_type = "correlation"):
    """
    Main function to estimate connectivity from atlas regions
    """

    correlation_measure = ConnectivityMeasure(kind = measure_type)
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    return((correlation_measure, correlation_matrix))


# function to plot the connectivity matrix
def plot_connectivity_matrix(correlation_matrix, labels,
    interp_arg = "nearest", cmap_arg = "RdBu_r", cmax = 0.8, cmin = -0.8,
    fig_size = [12,12]):
    """
    Plot the connectivity matrix.
    """

    plt.figure(figsize = fig_size)
    # Mask the main diagonal for visualization:
    np.fill_diagonal(correlation_matrix, 0)

    plt.imshow(correlation_matrix, interpolation = interp_arg, cmap = cmap_arg,
        vmax = cmax, vmin = cmin)

    # Add labels and adjust margins
    x_ticks = plt.xticks(range(len(labels) - 1), labels[1:], rotation=90)
    y_ticks = plt.yticks(range(len(labels) - 1), labels[1:])
    plt.gca().yaxis.tick_right()
    plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)



# boilerplate code to call main code for executing
if __name__ == '__main__':

    # parse arguments
    opts = parse_args()
    fmri_filenames = opts.data
    atlas_name = opts.atlas_name
    if opts.verbose:
        verbose_arg = 5
    else:
        verbose_arg = 0
    verbose_tf = opts.verbose
    csv2save = opts.csv2save
    pdf2save = opts.pdf2save
    cmap = opts.cmap
    cmax = np.array(opts.cmax, dtype = float)
    cmin = np.array(opts.cmin, dtype = float)
    fig_size = opts.fig_size
    fig_size = np.array(fig_size.split(','), dtype = int)
    if opts.cest == "correlation":
        conn_estimator = "correlation"
    elif opts.cest == "partialcorr":
        conn_estimator = "partial correlation"

    # get atlas
    [atlas_filename, labels] = get_atlas(atlas_name, verbose = verbose_tf)

    # prepare data
    [masker, time_series] = prepare_data(atlas_filename, fmri_filenames,
        standardize_arg = True, verbose_arg = verbose_arg)

    # estimate connectivity
    [corr_meas, corr_matrix] = estimate_connectivity(time_series,
        measure_type = conn_estimator)
    # save connectivity matrix to csv file
    if csv2save is not None:
        labels2use = labels[1:]
        pd_connmat = pd.DataFrame(corr_matrix, columns = labels2use,
            index = labels2use)
        pd_connmat.to_csv(csv2save)

    # make plot of connectivity matrix
    plot_connectivity_matrix(corr_matrix, labels, cmap_arg = cmap, cmax = cmax,
        cmin = cmin, fig_size = fig_size)
    # save plot to pdf
    if pdf2save is not None:
        plt.savefig(pdf2save)
    else:
        plt.show()

# plot_dvars.py
"""
plot_dvars.py - plots dvars estimates from preprocessed & wavelet denoised data

dvars_sm    = filename of dvars from preprocessed data
dvars_noise = filename of dvars from noise removed from wavelet denoising
dvars_wds   = filename of dvars from wavelet denoised data
subid       = subject id
pdf2save    = filename of pdf to save

Example usage:
python plot_dvars.py --dvars_sm rest_sm_dvars.txt
    --dvars_noise rest_noise_dvars.txt --dvars_wds rest_wds_dvars.txt
    --subid 0051456 --pdf2save dvars_plot.pdf
"""

# main function
def plot_dvars(pp_fname, noise_fname, wds_fname, subid, fname2save,
    gridline_width = 0.5):
    """
    Plot DVARS.
    """
    # read in files as np.array
    pp = np.loadtxt(pp_fname)
    noise = np.loadtxt(noise_fname)
    wds = np.loadtxt(wds_fname)

    # make plot
    plt.plot(pp, label = "Raw")
    plt.plot(noise, label = "Noise")
    plt.plot(wds, label = "Wavelet")

    plt.title(subid)
    plt.xlabel('Volume')
    plt.ylabel('DVARS')
    plt.legend()
    plt.grid(linewidth = gridline_width)
    plt.savefig(fname2save)

# function to parse input arguments
def parse_args():
    """
    Parse arguments.
    """
    parser=OptionParser()
    parser.add_option('--dvars_sm',"",dest='dvars_sm_file',help="DVARS on preprocessed data text file. ex: -dvars_sm rest_sm_dvars.txt ",default='')
    parser.add_option('--dvars_noise',"",dest='dvars_noise_file',help="DVARS on noise removed from wavelet denoising. ex: -dvars_noise rest_noise_dvars.txt ",default='')
    parser.add_option('--dvars_wds',"",dest='dvars_wds_file',help="DVARS on wavelet denoised data. ex: -dvars_wds rest_wds_dvars.txt ",default='')
    parser.add_option('--subid',"",dest='subid',help="Subject ID ex: -subid 0051456",default='')
    parser.add_option('--pdf2save',"",dest='pdf2save',help="PDF filename to save ex: -0 fd_dvars_plot.pdf",default='')
    (options,args) = parser.parse_args()
    return(options)


# boilerplate code to call main code for executing
if __name__ == '__main__':

    # import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from optparse import OptionParser,OptionGroup

    # parse arguments
    options = parse_args()

    # call main function
    plot_dvars(options.pp_file, options.noise_file, options.wds_file,
        options.subid, options.pdf2save)

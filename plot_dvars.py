# plot_dvars.py
"""
plot_dvars.py - plots dvars estimates from preprocessed & wavelet denoised data

python plot_dvars.py dvars_pp_file dvars_wds_file subid fname2save

dvars_pp_file = filename of dvars from preprocessed data
dvars_noise_file = filename of dvars from noise removed from wavelet denoising
dvars_wds_file = filename of dvars from wavelet denoised data
subid = subject id
fname2save = filename of pdf to save

Example usage:
python plot_dvars.py rest_sm_dvars.txt rest_noise_dvars.txt rest_wds_dvars.txt
    0051456 dvars_plot.pdf
"""

# main function
def plot_dvars(pp_fname, noise_fname, wds_fname, subid, fname2save,
    gridline_width = 0.5):

    # import libraries
    import numpy as np
    import matplotlib.pyplot as plt

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


# grab arguments
import sys
pp_file = sys.argv[1]
noise_file = sys.argv[2]
wds_file = sys.argv[3]
subid2use = sys.argv[4]
pdf2save = sys.argv[5]

# call main function
plot_dvars(pp_file, noise_file, wds_file, subid2use, pdf2save)

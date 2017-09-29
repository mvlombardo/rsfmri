# plot_fd_dvars.py
"""
plot_fd_dvars.py - plots framewise displacement and dvars as pdf plot

python plot_fd_dvars.py fd_file subid fname2save

fd_file = filename of fd.txt
subid = subject id
fname2save = filename of pdf to save

Example usage:
python plot_fd_dvars.py --fd motion_fd.txt --dvars_sm rest_sm_dvars.txt
    --dvars_noise rest_noise_dvars.txt --dvars_wds rest_wds_dvars.txt
    --subid 0051456 --pdf2save fd_plot.pdf
"""

# function to plot fd
def plot_fd(fname, subid, gridline_width = 0.5):

    # read in fd file as np.array
    fd = np.loadtxt(fname)

    # make plot
    plt.subplot(2,1,1).plot(fd)
    plt.title(subid)
    plt.xlabel('Volume')
    plt.ylabel('FD (mm)')
    plt.grid(linewidth = gridline_width)
    plt.tight_layout()


# function to plot dvars
def plot_dvars(pp_fname, noise_fname, wds_fname, subid, gridline_width = 0.5):
    # read in files as np.array
    pp = np.loadtxt(pp_fname)
    noise = np.loadtxt(noise_fname)
    wds = np.loadtxt(wds_fname)

    # make plot
    plt.subplot(2,1,2).plot(pp, label = "Raw")
    plt.subplot(2,1,2).plot(noise, label = "Noise")
    plt.subplot(2,1,2).plot(wds, label = "Wavelet")

    plt.title(subid)
    plt.xlabel('Volume')
    plt.ylabel('DVARS')
    plt.legend()
    plt.grid(linewidth = gridline_width)
    plt.tight_layout()

# function to parse input arguments
def parse_args():
    # from optparse import OptionParser,OptionGroup
    parser=OptionParser()
    parser.add_option('--fd',"",dest='fd_file',help="Framewise displacement text file ex: --fd rest_motion_fd.txt",default='')
    parser.add_option('--dvars_sm',"",dest='dvars_sm_file',help="DVARS on preprocessed data text file. ex: --dvars_sm rest_sm_dvars.txt ",default='')
    parser.add_option('--dvars_noise',"",dest='dvars_noise_file',help="DVARS on noise removed from wavelet denoising. ex: --dvars_noise rest_noise_dvars.txt ",default='')
    parser.add_option('--dvars_wds',"",dest='dvars_wds_file',help="DVARS on wavelet denoised data. ex: --dvars_wds rest_wds_dvars.txt ",default='')
    parser.add_option('--subid',"",dest='subid',help="Subject ID ex: --subid 0051456",default='')
    parser.add_option('--pdf2save',"",dest='pdf2save',help="PDF filename to save ex: --pdf2save fd_dvars_plot.pdf",default='')
    (options,args) = parser.parse_args()
    return(options)


# grab arguments
# import sys
# fd_file = sys.argv[1]
# subid2use = sys.argv[2]
# pdf2save = sys.argv[3]
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

# call main function
options = parse_args()
plot_fd(options.fd_file, options.subid)
plot_dvars(options.dvars_sm_file, options.dvars_noise_file,
    options.dvars_wds_file, options.subid)
# save figure as pdf
plt.savefig(options.pdf2save)

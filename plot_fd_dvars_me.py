# plot_fd_dvars_me.py
"""
plot_fd_dvars_me.py - plots framewise displacement and dvars for multi-echo data

Example usage:
python plot_fd_dvars_me.py --fd motion_fd.txt --dvars_hik hik_dvars.txt --dvars_tsoc tsoc_dvars.txt --subid 0051456 --pdf2save fd_plot.pdf
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser


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
def plot_dvars(hik_fname, tsoc_fname, subid, gridline_width = 0.5):
    # read in files as np.array
    hik = np.loadtxt(hik_fname)
    tsoc = np.loadtxt(tsoc_fname)

    # make plot
    plt.subplot(2,1,2).plot(hik, label = "HIK")
    plt.subplot(2,1,2).plot(tsoc, label = "TSOC")

    plt.title(subid)
    plt.xlabel('Volume')
    plt.ylabel('DVARS')
    # plt.legend()
    plt.grid(linewidth = gridline_width)
    plt.tight_layout()

# function to parse input arguments
def parse_args():
    # from optparse import OptionParser,OptionGroup
    parser=OptionParser()
    parser.add_option('--fd',"",dest='fd_file',help="Framewise displacement text file ex: --fd rest_motion_fd.txt",default='')
    parser.add_option('--dvars_hik',"",dest='dvars_hik_file',help="DVARS on hik data. ex: --dvars_hik hik_dvars.txt ",default='')
    parser.add_option('--dvars_tsoc',"",dest='dvars_tsoc_file',help="DVARS on tsoc data. ex: --dvars_tsoc tsoc_dvars.txt ",default='')
    parser.add_option('--subid',"",dest='subid',help="Subject ID ex: --subid 0051456",default='')
    parser.add_option('--pdf2save',"",dest='pdf2save',help="PDF filename to save ex: --pdf2save fd_dvars_plot.pdf",default='')
    (options,args) = parser.parse_args()
    return(options)


# boilerplate code to call main code for executing
if __name__ == '__main__':

    # call main function
    options = parse_args()
    plot_fd(options.fd_file, options.subid)
    plot_dvars(options.dvars_hik_file, options.dvars_tsoc_file, options.subid)

    # save figure as pdf
    plt.savefig(options.pdf2save)

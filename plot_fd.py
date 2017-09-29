# plot_fd.py
"""
plot_fd.py - plots framewise displacement as pdf plot

fd_file     = filename of fd.txt
subid       = subject id
pdf2save    = filename of pdf to save

Example usage:
python plot_fd.py --fd rest_motion_fd.txt --subid 0051456 --pdf2save fd_plot.pdf
"""

# main function
def plot_fd(fname, subid, fname2save, gridline_width = 0.5):
    """
    Plot framewise displacement.
    """
    # read in fd file as np.array
    fd = np.loadtxt(fname)

    # make plot
    plt.plot(fd)
    plt.title(subid)
    plt.xlabel('Volume')
    plt.ylabel('FD (mm)')
    plt.grid(linewidth = gridline_width)
    plt.savefig(fname2save)


# function to parse input arguments
def parse_args():
    """
    Parse arguments.
    """
    parser=OptionParser()
    parser.add_option('--fd',"",dest='fd_file',help="Framewise displacement text file ex: -fd rest_motion_fd.txt",default='')
    parser.add_option('--subid',"",dest='subid',help="Subject ID ex: -subid 0051456",default='')
    parser.add_option('--pdf2save',"",dest='pdf2save',help="PDF filename to save ex: -0 fd_dvars_plot.pdf",default='')
    (options,args) = parser.parse_args()
    return(options)


# boilerplate code to call main code for executing
if __name__ == '__main__':

    # import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from optparse import OptionParser

    # parse arguments
    options = parse_args()

    # call main function
    plot_fd(options.fd_file, options.subid, options.pdf2save)

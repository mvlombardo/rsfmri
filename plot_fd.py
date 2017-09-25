# plot_fd.py
"""
plot_fd.py - plots framewise displacement as pdf plot

python plot_fd.py fd_file subid fname2save

fd_file = filename of fd.txt
subid = subject id
fname2save = filename of pdf to save

Example usage:
python plot_fd.py motion_fd.txt 0051456 fd_plot.pdf
"""

# main function
def plot_fd(fname, subid, fname2save, gridline_width = 0.5):

    # import libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # read in fd file as np.array
    fd = np.loadtxt(fname)

    # make plot
    plt.plot(fd)
    plt.title(subid)
    plt.xlabel('Volume')
    plt.ylabel('FD (mm)')
    plt.grid(linewidth = gridline_width)
    plt.savefig(fname2save)


# grab arguments
import sys
fd_file = sys.argv[1]
subid2use = sys.argv[2]
pdf2save = sys.argv[3]

# call main function
plot_fd(fd_file, subid2use, pdf2save)

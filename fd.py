# fd.py
"""
Compute framewise displacement.

For usage see:  python fd.py -h

Example usage:
python fd.py -d rest_motion.1D
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import pandas as pd


# function to parse input arguments
def parse_args():
    """
    Parse arguments.
    """
    parser=OptionParser()
    parser.add_option('-d',"",dest='data',help="File with motion parameters file ex: -d rest_motion.1D",default=None)
    parser.add_option('-p',"",action="store_true",dest='plot',help="make fd plot",default=False)
    (options,args) = parser.parse_args()
    return(options)

# function for making plot
def make_plot(data):
    """
    Make FD plot
    """
    plt.plot(data)
    plt.xlabel("Frame #")
    plt.ylabel("Framewise Displacement (mm)")
    plt.show()

def compute_summary_stats(fd):
    """
    Compute summary stats.
    """
    summary_stats = {"meanFD":fd.mean(), "medianFD":np.median(fd),
        "minFD":fd.min(), "maxFD":fd.max()}
    return(summary_stats)

def write_summary_stats(summary_stats, outname):
    """
    Write summary stats to file.
    """

    outseries = pd.Series(summary_stats)
    outseries.to_csv(outname)


# boilerplate code to call main code for executing
if __name__ == '__main__':

    # Parse arguments
    opts = parse_args()
    data_file = opts.data

    # load data
    m = np.loadtxt(data_file)

    # compute framewise displacement
    dmdt = np.abs(np.diff(m,n=1,axis=0))
    fd=np.sum(dmdt,axis=1)

    # save fd to text file
    outname = '%s_fd.txt' % (data_file.split('.')[0])
    np.savetxt(outname,fd)

    # save summary stats to a file
    summary_stats = compute_summary_stats(fd)
    outname = '%s_fd_summary_stats.csv' % (data_file.split('.')[0])
    write_summary_stats(summary_stats, outname)

    # make plot
    if opts.plot:
        make_plot(fd)

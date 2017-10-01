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
    np.savetxt('%s_fd.txt' % (argv[1].split('.')[0]),fd)

    # make plot
    if opts.plot:
        plt.plot(fd)
        plt.xlabel("Frame #")
        plt.ylabel("Framewise Displacement (mm)")
        plt.show()

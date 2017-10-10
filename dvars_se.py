# dvars_se.py
"""
Compute DVARS on single echo EPI data.

For usage see:  python dvars_se.py -h

Example usage:
python dvars_se.py -d rest_sm.nii.gz
"""

# import modules
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from optparse import OptionParser
import pandas as pd


# function to parse input arguments
def parse_args():
    """
    Parse arguments.
    """
    parser=OptionParser()
    parser.add_option('-d',"",dest='data',help="EPI data to compute DVARS on file ex: -d rest_sm.nii.gz",default=None)
    parser.add_option('-p',"",action="store_true",dest='plot',help="Make DVARS plot",default=False)
    (options,args) = parser.parse_args()
    return(options)

def make_plot(data):
    """
    Make DVARS plot
    """
    plt.plot(data)
    plt.xlabel("Frame #")
    plt.ylabel("DVARS (%x10)")
    plt.show()


def compute_summary_stats(dvars):
    """
    Compute summary stats.
    """
   summary_stats = {"meanDVARS":np.mean(data), "medianDVARS":np.median(data),
        "minDVARS":np.min(data), "maxDVARS":np.max(data)}
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
    nii = data_file
    fn = nib.load(nii)

    # get data and vectorize
    nx,ny,nz,nt = fn.get_data().shape
    data = fn.get_data().reshape([nx*ny*nz,nt]).T

    # compute mean
    d_mu = data.mean(0)

    # compute mask
    d_mask = d_mu!=0

    # grab voxels within mask
    db=data[:,d_mask]

    # compute DVARS
    dbdt = np.abs(np.diff(db,n=1,axis=0))+0.0000001
    dvars = np.sqrt(np.mean(dbdt**2,1))

    # save DVARS to text file
    np.savetxt('%s_dvars.txt' % nii.split('.')[0],dvars)

    # save summary stats to a file
    summary_stats = compute_summary_stats(dvars)
    outname = '%s_dvars_summary_stats.csv' % (data_file.split('.')[0])
    write_summary_stats(summary_stats, outname)

    # make plot
    if opts.plot:
        make_plot(dvars)

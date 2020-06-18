import numpy as np
import os
import shutil
import argparse
from ml_methods import *

my_str = '''
#!/bin/bash
#BSUB -J pylab
#BSUB -o fl.out
#BSUB -e fl.err

# This is a sample script with specific resource requirements for the
# **bigmemory** queue with 64GB memory requirement and memory
# limit settings, which are both needed for reservations of
# more than 40GB.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
# bsub < example_bigmulti_8CPU_64GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q big-multi
#BSUB -n 8
#BSUB -M 10000
#BSUB -R rusage[mem=10000]

# Some important variables to check (Can be removed later)
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR

# Add your job command here
# Load module
module load anaconda/default
source activate dispatcher


cd /PHShome/jjd65/CDIFF/cdiff_metabolomics

python3 ./main_parallelized.py -dtype {0} -lambda {1} -w {pyt2} -reg {3} -hoi {4}

python3 ./main.py -o {0} -dtype {1}
'''

# Make the directories to store the information

#try:
#    os.mkdir(basepath)
#except OSError:
    # Remove the diinconsistent use of tabs and spaces in indentationrectory and then make one
#    shutil.rmtree(basepath)
#    os.mkdir(basepath)

# options = ['auc', 'f1', 'loss']
dtypes = ['week_one', 'all_data']
weights = [True, False]
regularizers = [1,2]
lambda_vector = np.logspace(-3, 2, num=50)

ixx = ml.leave_one_out_cv(data_in, targets)
outer_loops = len(ixx)
for dtype in dtypes:
    ixx = ml.leave_one_out_cv(ml.data_dict[dtype], ml.targets_int[dtype])
    pickle.dump(ixx, open(dtype + "_ixx.pkl", "wb"))
    outer_loops = len(ixx)
    for i in range(outer_loops):
        for lamb in labmda_vector:
            for w in weights:
                for reg in regularizers:
                    fname = 'cdiff_logregnet' + dtype + str(i) + str(lamb).replace('.'.'_') + str(w) + str(reg) + '.lsf'
                    f = open(fname,'w')
                    f.write(my_str.format(dtype, lamb, w, reg, i))
                    f.close()
                    os.system('bsub < {}'.format(fname))
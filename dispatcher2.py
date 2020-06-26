import numpy as np
import os
import shutil
import argparse
import pickle

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

python3 ./main_parallelized.py -dtype {0} -lambda {1} -w {2} -reg {3} -lr {4} -epochs {5}
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
regularizers = [1, None]
epoch_vec = [100,200]
learning_rates = [.1,.01,.001,.0001]
lambda_vector = np.logspace(-1, 3, num=50)

for dtype in dtypes:
    for lamb in lambda_vector:
        for w in weights:
            for reg in regularizers:
                for lr in learning_rates:
                    for epoch in epoch_vec:
                        fname = 'cdiff_logregnet.lsf'
                        f = open(fname,'w')
                        f.write(my_str.format(dtype, lamb, w, reg, lr, epoch))
                        f.close()
                        os.system('bsub < {}'.format(fname))

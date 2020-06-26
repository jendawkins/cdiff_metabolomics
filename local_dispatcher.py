import subprocess
import numpy as np
import os
import shutil
import argparse
import pickle

my_str = '''
python3 ./main_parallelized.py -dtype {0} -lambda {1} -w {2} -reg {3}
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
regularizers = [1, 2]
lambda_vector = np.logspace(-3, 2, num=50)

for dtype in dtypes:
    for lamb in lambda_vector:
        for w in weights:
            for reg in regularizers:
                process = subprocess.Popen(my_str.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

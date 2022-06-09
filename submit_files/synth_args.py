import glob
import os
import pickle
import pdb
import numpy as np
import itertools

script_path = '/home/akumar/nse/neural_control/synth.py'
desc = 'Synthetic fits with polar decomposition ensemble'
dbfile = '/home/akumar/nse/neural_control/LDS_db.dat'

with open(dbfile, 'rb') as f:
    lA = pickle.load(f)
    lB = pickle.load(f)

idxs = list(itertools.product(np.arange(lA).astype(int), np.arange(lB).astype(int)))
nsplits = 500
task_args = [{'dims': np.array([2, 6, 10, 20]), 'dimreduc_args': {'method':'FCCA', 'method_args':{'T':3, 'n_init':5}}}, 
             {'dims': np.array([2, 6, 10, 20]), 'dimreduc_args': {'method':'DCA', 'method_args':{'T':3, 'n_init':5}}}]

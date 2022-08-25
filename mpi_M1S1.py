import numpy as np
import pickle
import scipy
import pdb
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold

import sys
sys.path.append('/home/akumar/nse/neural_control')
from decoders import lr_decoder
from loaders import load_sabes

from mpi4py import MPI

bin_width = [50, 25]
spike_threshold = [0, 100]
data_files = glob('/mnt/Secondary/data/sabes/loco*')
data_files.append('/mnt/Secondary/data/sabes/indy_20160426_01.mat')
regressionlags = np.array([-4, -2, 0, 2, 4])
decoding_window = np.arange(1, 3)

comm = MPI.COMM_WORLD

R2all = []
for i, bw in tqdm(enumerate(bin_width)):
    for data_file in tqdm(data_files):
        if comm.rank == 0:
            datM1 = load_sabes(data_file, bin_width=bw, region='M1', spike_threshold=0)
            datS1 = load_sabes(data_file, bin_width=bw, region='S1', spike_threshold=0)
            X = datS1['spike_rates'].squeeze()
            Y = datM1['spike_rates'].squeeze()
        else:
            X = None
            Y = None

        X = comm.bcast(X)
        Y = comm.bcast(Y)
        n_M1 = Y.shape[-1]

        tasks = np.array_split(np.arange(n_M1), comm.size)[comm.rank]

        r2 = np.zeros((decoding_window.size, 5, regressionlags.size, tasks.size, 2))
        for i2, dw in enumerate(decoding_window):            
            for f, (train_idxs, test_idxs) in enumerate(KFold(n_splits=5, shuffle=False).split(X)):
                for j, rl in enumerate(regressionlags):      
                    for k, nidx in tqdm(enumerate(tasks)):
                        xtrain = X[train_idxs] 
                        xtest = X[test_idxs]
                        ytrain = Y[train_idxs, nidx][:, np.newaxis]
                        ytest = Y[test_idxs, nidx][:, np.newaxis]
                        r2_, _ = lr_decoder(xtest, xtrain, ytest, ytrain, rl, rl, decoding_window=dw, include_velocity=False, include_acc=False)
                        r2[i2, f, j, k, 0] = r2_
                        r2[i2, f, j, k, 1] = nidx

        r2 = comm.gather(r2)
        if comm.rank == 0:
            result = {}
            result['data_file'] = data_file
            result['bin_width'] = bw
            result['r2'] = r2

            with open('/home/akumar/nse/neural_control/data/M1S1/%s_%d.dat' % (data_file.split('/')[-1].split('.mat')[0], bw), 'wb') as f:
                f.write(pickle.dumps(result))

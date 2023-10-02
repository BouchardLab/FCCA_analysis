import pdb
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axisartist.axislines import AxesZero

from dca.methods_comparison import JPCA
from pyuoi.linear_model.var  import VAR
from neurosim.models.var import form_companion

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes, measure_straight_dev

start_times = {'indy_20160426_01': 0,
               'indy_20160622_01':1700,
               'indy_20160624_03': 500,
               'indy_20160627_01': 0,
               'indy_20160630_01': 0,
               'indy_20160915_01': 0,
               'indy_20160921_01': 0,
               'indy_20160930_02': 0,
               'indy_20160930_05': 300,
               'indy_20161005_06': 0,
               'indy_20161006_02': 350,
               'indy_20161007_02': 950,
               'indy_20161011_03': 0,
               'indy_20161013_03': 0,
               'indy_20161014_04': 0,
               'indy_20161017_02': 0,
               'indy_20161024_03': 0,
               'indy_20161025_04': 0,
               'indy_20161026_03': 0,
               'indy_20161027_03': 500,
               'indy_20161206_02': 5500,
               'indy_20161207_02': 0,
               'indy_20161212_02': 0,
               'indy_20161220_02': 0,
               'indy_20170123_02': 0,
               'indy_20170124_01': 0,
               'indy_20170127_03': 0,
               'indy_20170131_02': 0,
               }

if __name__ == '__main__':


    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'


    good_loco_files = ['loco_20170210_03.mat',
    'loco_20170213_02.mat',
    'loco_20170215_02.mat',
    'loco_20170227_04.mat',
    'loco_20170228_02.mat',
    'loco_20170301_05.mat',
    'loco_20170302_02.mat']

    with open('/mnt/Secondary/data/postprocessed/indy_S1_df.dat', 'rb') as f:
        rl2 = pickle.load(f)
    indy_df = pd.DataFrame(rl2)

    data_files = good_loco_files + list(np.unique(indy_df['data_file'].values))

    dpath = '/mnt/Secondary/data/sabes'

    DIM = 6
    inner_reps = 1000
    if not os.path.exists('jpcaAtmp_randomcontrolS1.dat'):
        # Now do subspace identification/VAR inference within these 
        # results = []
        resultsd3 = []
        for i, data_file in tqdm(enumerate(data_files)):
            dat = load_sabes('%s/%s' % (dpath, data_file))
            dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])

            y = np.squeeze(dat['spike_rates'])

            # fit a linear model
            

            for j in tqdm(range(inner_reps)):
                V = scipy.stats.special_ortho_group.rvs(y.shape[-1], random_state=np.random.RandomState(j))
                V = V[:, 0:DIM]
                # Project data
                yproj = y @ V

                # Segment reaches into minimum length 30 timesteps reaches
                yproj = np.array([yproj[t0:t0+20] for t0, t1 in dat['transition_times'] if t1 - t0 > 21])
                # yproj = gaussian_filter1d(yproj, sigma=5)

                result_ = {}
                result_['data_file'] = data_file
                result_['inner_rep'] = j

                jpca = JPCA(n_components=DIM, mean_subtract=False)
                jpca.fit(yproj)
                
                # ypred = yproj[:-1, :] @ jpca.M_skew
                #r2_linear = linmodel.score(yproj[:-1, :], np.diff(yproj, axis=0))
                result_['jeig'] = jpca.eigen_vals_

                yprojcent = np.array([y_ - y_[0:1, :] for y_ in yproj])
                dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(DIM)])
                result_['dyn_range'] = np.mean(dyn_range)
                resultsd3.append(result_)

        with open('jpcaAtmp_randomcontrolS1.dat', 'wb') as f:
            f.write(pickle.dumps(resultsd3))            

    print('Already done!')
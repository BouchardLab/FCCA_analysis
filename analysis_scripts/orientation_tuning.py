import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import sys
import pdb
from scipy.stats import spearmanr

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import cross_validate

from dca.dca import DynamicalComponentsAnalysis
from dca_research.kca import KalmanComponentsAnalysis as KCA
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import glob
import pickle

sys.path.append('/home/akumar/nse/neural_control')
from loaders import load_sabes 
from utils import apply_df_filters, calc_loadings
from decoders import lr_decoder
from segmentation import reach_segment_sabes

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

def fit_DCAnPCAonBin(X):
    """Now just used to calculate PCA and DCA loading"""
    KCAmodel = KCA(d=2, T=3, causal_weights=(0, 1), project_mmse=True)
    PCAmodel = PCA(n_components=2)
    KCAmodel.fit(X)
    extended = X[0]
    for transit in X[1:]:
        extended = np.vstack((extended,transit))
    PCAmodel.fit(extended)
    PCA_loading = calc_loadings(PCAmodel.components_.T)
    KCA_loading = calc_loadings(KCAmodel.coef_)
    return PCA_loading, KCA_loading

def sprVSRsqNsqaured_sum_coef(rsquared, coefs,loading):
    """Given each neuron's loading.
        Get the spearmanr of loading V.S rsqaured of this neruon in the binned model,
        as well as spearmanr of loading V.S squared sum of sin/cos coefficients of this neruon in the model.
        Return correlations"""
    spearmanr_loading_rsquared = spearmanr(loading, rsquared)
    sqaured_sum_coef = np.add(np.square(coefs[:,0]),np.square(coefs[:,1]))
    spearmanr_loading_sqauredsum_coef = spearmanr(loading,sqaured_sum_coef)
    return spearmanr_loading_rsquared.correlation,spearmanr_loading_sqauredsum_coef.correlation

if __name__ == '__main__':

    tau = [0, 2, 4, 6, 8]
    data_path = '/mnt/Secondary/data/sabes'

    df = '/home/akumar/nse/neural_control/data/sabes_decoding_df.dat'
    with open(df, 'rb') as f:
        sabes_df = pickle.load(f)


    data_files = np.unique(sabes_df['data_file'].values)

    # Manual segmentation of orientation
    bins = np.arange(-np.pi,np.pi,.25 * np.pi)

    times_binning_better = 0
    results_list = []

    for i, file_ in tqdm(enumerate(data_files)):
        df = apply_df_filters(sabes_df, data_file=file_)
        dat = load_sabes('%s/%s' % (data_path, file_), bin_width=df.iloc[0]["bin_width"],
                        filter_fn=df.iloc[0]['filter_fn'], filter_kwargs=df.iloc[0]['filter_kwargs'],
                        boxcox=df.iloc[0]['boxcox'], spike_threshold=df.iloc[0]['spike_threshold'])
        
        dat_segmented = reach_segment_sabes(dat, start_times[file_.split(".")[0]])

        spike_rates = dat_segmented['spike_rates']
        spike_rates = spike_rates.reshape(spike_rates.shape[1], -1)
        vels = dat_segmented['vel']

        #||V(t)||
        peak_vels_in_windows = np.array([np.amax(np.absolute(vels[start : end + 1])) \
                                        for start, end in dat_segmented['transition_times']])[:,np.newaxis]
        orientation_in_windows = dat_segmented['transition_orientation']

        #||V(t)||sin[theta(t)]
        #peak_vels_in_windows = normalize(peak_vels_in_windows, axis = 0)
        vel_sin = np.sin(orientation_in_windows)[:,np.newaxis] * peak_vels_in_windows

    #     vel_sin = np.sin(orientation_in_windows)[:,np.newaxis] * peak_vels_in_windows
        #||V(t)||cos[theta(t)]
        vel_cos = np.cos(orientation_in_windows)[:,np.newaxis] * peak_vels_in_windows

    #    vel_cos = normalize(np.cos(orientation_in_windows)[:,np.newaxis] * peak_vels_in_windows, axis = 0)

        #Binning
        binned_indices = np.digitize(orientation_in_windows, bins)
        binned_indices = [np.where(binned_indices == idx) for idx in range(1,9)]

        #To record the r^2 and coeffs indexed by (bin#, neuron#)
        r_squared_bin_neuron = np.zeros((len(binned_indices), len(tau), spike_rates.shape[1]))
        coefficients_bin_neuron = np.zeros((len(binned_indices), len(tau), spike_rates.shape[1], 2))

        for j in range(8):
            binned_idx = binned_indices[j]
            transitions_inbin = np.array(dat_segmented['transition_times'])[binned_idx]
            # Get the features 
            X = np.concatenate((peak_vels_in_windows[binned_idx], \
                                vel_sin[binned_idx], vel_cos[binned_idx]), axis = 1)
            for k, t_ in enumerate(tau):
                for neuron_idx in range(spike_rates.shape[1]):
                    spike_rates_neuron = spike_rates[:,neuron_idx]


                    average_rates_in_windows = np.array([np.average(spike_rates_neuron[max(start_time - t_, 0): \
                                                                                    min(end_time + 1 - t_, spike_rates.shape[0])])
                                                        for start_time, end_time in transitions_inbin])            
        
                    average_rates_in_windows = StandardScaler().fit_transform(average_rates_in_windows.reshape(-1, 1))

                    cvobj = cross_validate(LinearRegression(), X, average_rates_in_windows, cv=5, return_train_score=True)

                    #Getting r sqaured, and put it in the recording array
                    r_squared_bin_neuron[j, k, neuron_idx] = np.mean(cvobj['test_score'])
                    #Do the same for coefficients
                    coefficients_bin_neuron[j, k, neuron_idx] = np.nan


                #Prepare to fit DCA with the bin
                # expanded_transition_times = [list(range(max(0, start - t_), min(end+1 - t_, spike_rates.shape[0]))) 
                #                             for start,end in transitions_inbin]
                # spike_rates_list_transition = [spike_rates[transit] for transit in expanded_transition_times]
                # PCA_loading, KCA_loading = fit_DCAnPCAonBin(spike_rates_list_transition)
                # #Getting the spearmanr correlation for r^2 and sqaured sum of coefficients
                # spr_r2_dca, spr_coef_dca = sprVSRsqNsqaured_sum_coef(r_squared_bin_neuron[j, k], \
                #                                                     coefficients_bin_neuron[j, k],KCA_loading)
                # spr_r2_pca, spr_coef_pca = sprVSRsqNsqaured_sum_coef(r_squared_bin_neuron[j, k], \
                #                                                     coefficients_bin_neuron[j, k],PCA_loading)
            
                # # Now do so for only the top 20% most tuned neurons
                # n20 = int(0.2 * spike_rates.shape[1])
                # top_r2_neurons = np.argsort(r_squared_bin_neuron[j, k])[::-1][0:n20]
                # top_coef_neurons = np.argsort(np.linalg.norm(coefficients_bin_neuron[j, k]))[::-1][0:n20]

                # spr_r2_dca_tr2, spr_coef_dca_tr2 = sprVSRsqNsqaured_sum_coef(r_squared_bin_neuron[j, k, top_r2_neurons],
                #                                                             coefficients_bin_neuron[j, k, top_r2_neurons],
                #                                                             KCA_loading[top_r2_neurons])

                # spr_r2_pca_tr2, spr_coef_pca_tr2 = sprVSRsqNsqaured_sum_coef(r_squared_bin_neuron[j, k, top_r2_neurons], \
                #                                                             coefficients_bin_neuron[j, k, top_r2_neurons],
                #                                                             PCA_loading[top_r2_neurons])
                
                # spr_r2_dca_tc, spr_coef_dca_tc = sprVSRsqNsqaured_sum_coef(r_squared_bin_neuron[j, k, top_coef_neurons], \
                #                                                         coefficients_bin_neuron[j, k, top_coef_neurons],
                #                                                         KCA_loading[top_coef_neurons])
                
                # spr_r2_pca_tc, spr_coef_pca_tc = sprVSRsqNsqaured_sum_coef(r_squared_bin_neuron[j, k, top_coef_neurons], \
                #                                                         coefficients_bin_neuron[j, k, top_coef_neurons],
                #                                                         PCA_loading[top_coef_neurons])

                
                # Append results
                result = {'file': file_, 'bin_idx':j, 'tau': t_, 
                        'tuning_r2':r_squared_bin_neuron, 'theta_coef':coefficients_bin_neuron}
                        # 'PCA_loadings':PCA_loading, 'KCA_loading':KCA_loading,
                        # 'spr_r2_dca':spr_r2_dca, 'spr_coef_dca': spr_coef_dca,
                        # 'spr_r2_pca':spr_r2_pca, 'spr_coef_pca': spr_coef_pca,

                        
                        # # Top r2 neurons
                        # 'spr_r2_dca_tr2':spr_r2_dca_tr2, 'spr_coef_dca_tr2': spr_coef_dca_tr2,
                        # # Top coef neurons
                        # 'spr_r2_dca_tc':spr_r2_dca_tc, 'spr_coef_dca_tc': spr_coef_dca_tc,
                        # 'spr_r2_pca_tr2':spr_r2_pca_tr2, 'spr_coef_pca_tr2': spr_coef_pca_tr2,
                        # 'spr_r2_pca_tc':spr_r2_pca_tc, 'spr_coef_pca_tc': spr_coef_pca_tc}
                        
                        
                results_list.append(result) 

    result_df = pd.DataFrame(results_list)
    with open('orientation_df_unnorm_cv.dat', 'wb') as f:
        f.write(pickle.dumps(result_df))
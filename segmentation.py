import pdb
from copy import deepcopy
import numpy as np
import scipy 

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
               'loco_20170210_03':0, 
               'loco_20170213_02':0, 
               'loco_20170214_02':0, 
               'loco_20170215_02':0, 
               'loco_20170216_02': 0, 
               'loco_20170217_02': 0, 
               'loco_20170227_04': 0, 
               'loco_20170228_02': 0, 
               'loco_20170301_05':0, 
               'loco_20170302_02':0}

def measure_straight_dev(trajectory, start, end):
    # Translate to the origin relative to the 1st target location
    trajectory -= start

    # straight line vector
    straight = end - start
    straight_norm = np.linalg.norm(straight)
    straight /= straight_norm

    if straight[0] == 0:
        perp = np.array([1, 0])
    elif straight[1] == 0:
        perp = np.array([0, 1])
    else:
        # Vector orthogonal to the straight line between targets
        x_orth = np.random.uniform(0, 1)
        y_orth = -1 * (straight[0] * x_orth)/straight[1]
        perp = np.array([x_orth, y_orth])
        perp /= np.linalg.norm(perp)
    
    if np.any(np.isnan(perp)):
        pdb.set_trace()
    
    m = straight[1]/straight[0]
    b = 0

    straight_dev = 0
    for j in range(trajectory.shape[0]):
        
        # transition is horizontal
        if m == 0:
            x_int = trajectory[j, 0]
            y_int = straight[1]
        # transition is vertical
        elif np.isnan(m) or np.isinf(m):
            x_int = straight[0]
            y_int = trajectory[j, 1]
        else:
            m1 = -1/m
            b1 = trajectory[j, 1] - m1 * trajectory[j, 0]
            # Find the intersection between the two lines
            x_int = (b - b1)/(m1 - m)
            y_int = m1 * x_int + b1
        
        straight_dev += np.linalg.norm(np.array([x_int - trajectory[j, 0], y_int - trajectory[j, 1]]))

    # Normalize by the length of straight trajectory
    straight_dev /= straight_norm
    return straight_dev

def reach_segment_sabes(dat, start_time=None, data_file=None, keep_high_error=False, err_thresh=0.9):
    print('Reminder that start times depend on the bin size')
    if start_time is None:
        start_time = start_times[data_file]

    target_locs = []
    time_on_target = []
    valid_transition_times = []

    target_diff = np.diff(dat['target'].T)
    # This will yield the last index before the transition
    transition_times = np.sort(np.unique(target_diff.nonzero()[1]))
    #transition_times = target_diff.nonzero()[1]

    # For each transition, make a record of the location, time on target, and transition_vector
    # Throw away those targets that only appear for 1 timestep
    for i, transition_time in enumerate(transition_times):

        # Only lingers at the target for one timestep
        if i < len(transition_times) - 1:
            if np.diff(transition_times)[i] == 1:
                continue

        target_locs.append(dat['target'][transition_time][:])
        valid_transition_times.append(transition_time)
        
    for i, transition_time in enumerate(valid_transition_times):
            
        if i == 0:
            time_on_target.append(transition_time + 1)
        else:
            time_on_target.append(transition_time - valid_transition_times[i - 1] + 1)
            
    target_locs = np.array(target_locs)
    time_on_target = np.array(time_on_target)
    valid_transition_times = np.array(valid_transition_times)

    # Filter out by when motion starts
    if start_time > valid_transition_times[0]:
        init_target_loc = target_locs[valid_transition_times < start_time][-1]
    else:
        init_target_loc = target_locs[0]

    target_locs = target_locs[valid_transition_times > start_time]
    time_on_target = time_on_target[valid_transition_times > start_time]
    valid_transition_times = valid_transition_times[valid_transition_times > start_time]

    # Velocity profiles
    vel = np.diff(dat['behavior'], axis=0)

    target_pairs = []
    for i in range(1, len(target_locs)):
        target_pairs.append((i - 1, i))

    target_error_pairs = np.zeros(len(target_pairs))

    for i in range(len(target_pairs)):
        
    #    time_win = max(min(10, int(0.05 * time_on_target[i])), 2)
        time_win = 2
        
        # Length of time_win just after target switches
        cursor_0 = dat['behavior'][valid_transition_times[target_pairs[i][0]] + 1:\
                                   valid_transition_times[target_pairs[i][0]] + 1 + time_win]
        # Length of time_win just before target switches again
        cursor_1 = dat['behavior'][valid_transition_times[target_pairs[i][1]] - time_win:\
                                   valid_transition_times[target_pairs[i][1]]]

        target_error_pairs[i] = np.max([np.mean(np.linalg.norm(cursor_0 - target_locs[target_pairs[i][0]])),
                                         np.mean(np.linalg.norm(cursor_1 - target_locs[target_pairs[i][1]]))])

    # Thresholding by error threshold (how far from the start and end targets is the reach)
    err_thresh = np.quantile(target_error_pairs, err_thresh)

    # Throw away trajectories with highly erratic velocity profiles
    # (large number of zero crossings in the acceleration)
    n_zeros = np.zeros(len(target_pairs))
    for i in range(len(target_pairs)):
        acc = np.diff(vel[valid_transition_times[target_pairs[i][0]]:\
                          valid_transition_times[target_pairs[i][1]]], axis=0)    
        n_zeros[i] = (np.diff(np.sign(acc)) != 0).sum()

    # Throw away reaches with highest 10 % of target error and > 200 acceleration zero crossings
    # Pair of target corrdinates
    valid_target_pairs = []
    # How long did the reach take
    reach_duration = []
    # Tuple of indices that describes start and end of reach
    transition_times = []
    transition_vectors = []
    nzeros = []

    indices_kept = []

    for i in range(len(target_error_pairs)): 
        # Keep this transition
        if (target_error_pairs[i] < err_thresh and n_zeros[i] < 200) or keep_high_error:
            valid_target_pairs.append((target_locs[target_pairs[i][0]], target_locs[target_pairs[i][1]]))        
            reach_duration.append(time_on_target[target_pairs[i][1]])
            transition_times.append((valid_transition_times[target_pairs[i][0]] + 1,
                                    valid_transition_times[target_pairs[i][1]]))
            transition_vectors.append(target_locs[target_pairs[i][1]] - target_locs[target_pairs[i][0]])
            indices_kept.append(i)
        else: 
            continue


    target_error_pairs = target_error_pairs[np.array(indices_kept)]
    n_zeros = n_zeros[np.array(indices_kept)]

    transition_orientation = np.zeros(len(transition_vectors))
    refvec = np.array([1, 0])
    for i in range(len(transition_vectors)):
        # Normalize
        transvecnorm = transition_vectors[i]/np.linalg.norm(transition_vectors[i])
        dot = transvecnorm @ refvec      # dot product
        det = transvecnorm[0]*refvec[1] - transvecnorm[1]*refvec[0]  # determinant
        transition_orientation[i] = np.arctan2(det, dot)

    # Integrate the area under the trajectory minus the straight line
    straight_dev = np.zeros(len(valid_target_pairs))
    # Operator on a copy of trajectory
    cursor_trajectory = deepcopy(dat['behavior'])
    for i in range(len(valid_target_pairs)):
        
        trajectory = cursor_trajectory[transition_times[i][0]:transition_times[i][1], :]

        straight_dev[i] = measure_straight_dev(trajectory, valid_target_pairs[i][0], 
                                               valid_target_pairs[i][1])

    # Augment dictionary with segmented reaches and their characteristics
    dat['vel'] = vel
    dat['target_pairs'] = valid_target_pairs
    dat['transition_times'] = transition_times
    dat['straight_dev'] = straight_dev
    dat['target_pair_error'] = target_error_pairs
    dat['transition_orientation'] = transition_orientation
    dat['npeaks'] = n_zeros

    return dat
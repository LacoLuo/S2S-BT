'''
It is a modified code offered at https://github.com/malrabeiah/CameraPredictsBeams/blob/master/prep_data_struct.py
---------------
Author: Hao Luo
Sep. 2021
'''

import os 
import argparse
import math
import pickle 
import h5py as h5 
import numpy as np 
import scipy.io as sciio 
import skimage.io as imio

def calculate_noise(bw):
    return -174+10.0*math.log10(bw)

NUM_DATA = 5000
NUM_SUBCARRIER = 1
NUM_ANTENNA = 128
POWER = 5 # dbm
BW = 1 # MHz
NOISE = calculate_noise(BW * 1e9) # dbm

def dbm_to_watt(dbm):
    return math.pow(10.0, dbm/10.0) # mW

def getMATLAB(codebook, wireless):
    scenario = os.path.basename(wireless).split('_')[0]
    print("Scenario: {}".format(scenario))

    if scenario == 'colo':
        num_bs = 1
        channels = np.zeros((1, NUM_DATA, NUM_SUBCARRIER, NUM_ANTENNA), dtype=np.complex64)
        locs = np.zeros((1, NUM_DATA, 3))
    elif scenario == 'dist':
        num_bs = 3
        channels = np.zeros((num_bs, NUM_DATA, NUM_SUBCARRIER, NUM_ANTENNA), dtype=np.complex64)
        locs = np.zeros((num_bs, NUM_DATA, 3))
    # Read MATLAB structure
    f = h5.File(wireless, 'r')
    key = list(f.keys())
    for i in range(num_bs):
        raw_data = f[f[key[1]][i][0]]['user']
        print(raw_data)
        for j in range(NUM_DATA):
            channel = f[raw_data[j][0]]['channel'][:]
            loc = np.squeeze(f[raw_data[j][0]]['loc'][:])

            channel = channel.view(np.double).reshape((NUM_SUBCARRIER, NUM_ANTENNA, 2))
            channel = channel.astype(np.float32)
            channel_complex = channel[:, :, 0] + channel[:, :, 1] * 1j  # [ sub-carriers, # of antennas ]
            
            channels[i, j] = channel_complex
            locs[i, j] = loc
    
    # Normalize channels
    rec_pow = np.mean(np.abs(channels)**2)
    channels = channels/np.sqrt(rec_pow)

    raw_data = {
        'ch': channels,
        'loc': locs,
        'norm_fact': rec_pow
    }

    return raw_data, scenario

def beamPredStruct(raw_data, codebook, image_dir, scenario):
    image_names = os.listdir(image_dir)
    image_names = sorted(image_names)
    loc = raw_data['loc'][0, :, 0:2]

    output_dir = image_dir.replace('rgb', 'beam')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tmp = 0.0
    tmp_min = 0.0
    for i in range(NUM_DATA):
        # Find the information in the image: number of BS, coordinates, Block Status
        split_name = image_names[i].split('_')
        x_axis = float(split_name[2])
        y_axis = float(split_name[3].split('.jpg')[0])
        coord = np.array([x_axis, y_axis])
        if scenario == 'dist':
            cam_num = int(split_name[1]) - 1
        elif scenario == 'colo':
            cam_num = 0

        # Find the channel of those coordinates
        diff = np.sum( np.abs(loc - coord), axis=1 )
        data = np.argmin(diff)
        h = raw_data['ch'][cam_num, data, :, :]

        # Find the best beamforming vector
        codebook_H = codebook.conj()
        rec_pow_sub = np.power(np.abs(np.matmul(h, codebook_H)), 2) # per subcarrier
        rate_per_sub = BW * np.log2( 1 + (dbm_to_watt(POWER)/(dbm_to_watt(NOISE)*NUM_SUBCARRIER))*rec_pow_sub ) 

        ave_rate_per_beam = 1. * np.mean(rate_per_sub, axis=0)  # averaged over subcarriers Mbit/s
        beam_ind = np.argmax(ave_rate_per_beam)+1
        output_name = image_names[i][:-4] + '_' + str(beam_ind) + '.bin'
        output_path = output_dir  + '/' + output_name
        with open(output_path, 'wb') as f:
            f.write(ave_rate_per_beam.astype(np.float32).tostring())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare beam-image pair"
    )
    parser.add_argument(
        "--image_dir", required=True, type=str,
        help="root directory of images"
    )
    parser.add_argument(
        "--codebook", default="./data_generation_package/DFT_codebook128.mat", type=str,
        help="path of codebook"
    )
    parser.add_argument(
        "--wireless", required=True, type=str,
        help="path of wireless data"
    )
    args = parser.parse_args()

    codebook = sciio.loadmat(args.codebook)['ans']
    print("Finish Loading Codebook.")

    raw_data, scenario = getMATLAB(codebook=codebook, wireless=args.wireless)

    beamPredStruct(raw_data, codebook, args.image_dir, scenario)

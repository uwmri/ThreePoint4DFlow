import os
import numpy as np
import glob
import h5py

def load_convert_4point(filename=None):

    # Data is in hdf5 which we read in here
    f = h5py.File(filename, 'r')

    velocity = np.array(f['VELOCITY'])
    image = np.array(f['IMAGE'])
    #weight = np.array(f['WEIGHT'])
    magnitude = np.array(f['WEIGHT'])

    weight = generate_weight(velocity, image, magnitude, type='cd')

    return image,weight,velocity

def generate_weight(velocity, image, magnitude, type):

    vel_mag = np.sqrt(np.sum(np.square(velocity), axis=3))

    if type == 'cd':
        cd = magnitude * np.sin(vel_mag * np.pi / 2)
        return cd

    if type == 'inv_vel_freq':
        # Weight as inverse of frequency
        # Bin numbers start from 1, whereas array index starts from 0
        inside_head = (magnitude > 1)
        bin_edges = np.arange(0, 2.1, 0.1)
        vel_freq, _ = np.histogram(vel_mag[inside_head], bin_edges, density=True)
        binned_vel = np.digitize(vel_mag, bin_edges) - 1
        weight = np.zeros(magnitude.shape)

        for i in range(len(bin_edges)-1):
            if vel_freq[i] > 0:
                mask = (binned_vel == i) & inside_head
                weight[mask] = 1 / vel_freq[i]
        weight = magnitude * weight
        return weight


def load_raw_4point(filename=None):

    # Data is in hdf5 which we read in here
    f = h5py.File(filename, 'r')

    # Load 4 encodes
    Im0 = np.array(f['Images']['Encode_000_Frame_000'])
    flow_case = np.zeros( Im0.shape + (8,),dtype=np.float32)
    flow_case[:, :, :, 0] = Im0['real']
    flow_case[:, :, :, 1] = Im0['imag']

    Im1 = np.array(f['Images']['Encode_001_Frame_000'])
    flow_case[:, :, :, 2] = Im1['real']
    flow_case[:, :, :, 3] = Im1['imag']

    Im2 = np.array(f['Images']['Encode_002_Frame_000'])
    flow_case[:, :, :, 4] = Im2['real']
    flow_case[:, :, :, 5] = Im2['imag']

    Im3 = np.array(f['Images']['Encode_003_Frame_000'])
    flow_case[:, :, :, 6] = Im3['real']
    flow_case[:, :, :, 7] = Im3['imag']

    flow_case /= np.mean(np.abs(flow_case))

    return flow_case

def generate_folder_list(base_folder='./'):
    names = os.path.join(base_folder, '*Images_*.h5')
    file_names = []
    for file in glob.glob(names):
        file_names.append(file)

    return file_names







import os
import numpy as np
import glob
import h5py

def load_convert_4point(filename=None):

    # Data is in hdf5 which we read in here
    f = h5py.File(filename, 'r')

    velocity = np.array(f['VELOCITY'])
    image = np.array(f['IMAGE'])
    weight = np.array(f['WEIGHT'])

    return image,weight,velocity

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







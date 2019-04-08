import os
import numpy as np
import tensorflow.keras as keras
import glob
import matplotlib.pyplot as plt
import math
from tensorflow.keras import backend as K
from functools import partial
import h5py
import tensorflow.keras.layers
import tensorflow as tf
from tensorflow.keras import backend as Ktf
import matplotlib.pyplot as plt

from threepoint_network import *
from threepoint_io import *
from flow_tools import *

training_data_dir = './Data'
out_dir = './DataConvert'
all_names = generate_folder_list(training_data_dir)

mrflow = MRI_4DFlow()
mrflow.set_encoding_matrix('4pt-referenced')

for case_name in all_names:
    out_name = os.path.join(out_dir,os.path.basename(case_name))

    # Load Data
    data = load_raw_4point(case_name)



    mrflow.signal = np.expand_dims(data[:, :, :, 0::2] + 1j * data[:, :, :, 1::2], -1)
    mrflow.solve_for_velocity()
    velocity = mrflow.velocity_estimate
    weights = np.squeeze(np.sqrt(np.mean(data ** 2, -1)))

    #Now save to new file
    with h5py.File(out_name, 'w') as hf:
        hf.create_dataset("IMAGE", data=np.squeeze(data[:,:,:,0:6]) )
        hf.create_dataset("VELOCITY", data=np.squeeze(velocity) )
        hf.create_dataset("WEIGHT",data=np.squeeze(weights) )

import os
import numpy as np
import glob
import math
from functools import partial
import h5py
import matplotlib.pyplot as plt

from threepoint_io import *
from flow_tools import *

training_data_dir = './Data2'
out_dir = './DataConvert2'
all_names = generate_folder_list(training_data_dir)

mrflow = MRI_4DFlow()
mrflow.set_encoding_matrix('4pt-referenced')

for case_name in all_names:
    out_name = os.path.join(out_dir,os.path.basename(case_name))
    print(out_name)

    #Now save to new file
    with h5py.File(out_name, 'w') as hf:

        # Load Data
        data = load_raw_4point(case_name)

        mrflow.signal = np.expand_dims(data[:, :, :, 0::2] + 1j * data[:, :, :, 1::2], 0)
        mrflow.solve_for_velocity()

        mrflow.update_angiogram()
        mrflow.background_phase_correct()

        plt.figure()
        plt.imshow(mrflow.velocity_estimate[0,100,:,:,2] )
        plt.clim(-0.1,0.1)
        plt.savefig(out_name + '.png')

        # Save velocity
        velocity = mrflow.velocity_estimate
        hf.create_dataset("VELOCITY", data=np.squeeze(velocity))

        weights = np.squeeze(np.sqrt(np.mean(data ** 2, -1)))
        hf.create_dataset("WEIGHT",data=np.squeeze(weights) )

        #Now save to new file
        hf.create_dataset("IMAGE", data=np.squeeze(data[:,:,:,2:]) )

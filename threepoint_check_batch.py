import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import math
from functools import partial
import h5py
import matplotlib.pyplot as plt
import keras
from keras import backend as K

#pretrained = 'B:/ThreePoint/Graph/2019-04-01-22-57/weights.090-0.04264'
pretrained = 'D:/3POINT/weights.200-0.12259'
data_directory = 'D:/3POINT/DataConvert'
output_directory = 'D:/3POINT/Inferred'

# User variables
from threepoint_io import *
from threepoint_network import *
from flow_tools import *

# Weighted MSE
input_weights = keras.Input(shape=(None, None, None, 1), dtype='float32')
wmse = partial(weighted_mse, weights=input_weights)
wmse.__name__ = 'weighted_mse'

# Import model
unwrap_model = keras.models.load_model(pretrained,custom_objects={'cropped_mse':cropped_mse, 'weighted_mse':wmse})

# Get model summary
unwrap_model.summary(line_length=200)

all_names = generate_folder_list(data_directory)

# Seperate into train/eval
Ntrain = int(0.8*len(all_names))
Tnames = all_names[:Ntrain]
Vnames = all_names[Ntrain:]

for case_name in Vnames:

    # Load case
    image,weight,velocity = load_convert_4point(os.path.join(data_directory,os.path.basename(case_name)))

    # Set cropping
    if unet:
        crop_hsize = 0
        crop_size = 0
        eval_patch_size = patch_size
        patch_output  = patch_size
    else:
        crop_hsize = outer_layers+1
        crop_size = crop_hsize*2
        eval_patch_size = 80 + crop_size
        patch_output  = eval_patch_size - crop_size

    # Grab the label
    output = np.zeros( velocity.shape, dtype=np.float32 )
    block_average = np.zeros( velocity.shape, dtype=np.float32 )

    # Test on whole image
    blocks = np.ceil(np.asarray(image.shape) / patch_output).astype(np.int)

    print('Patch Output = %d' % (patch_output,))

    for bs in (False,True):

        if bs:
            image = np.roll( image, (int(patch_output/2),int(patch_output/2),int(patch_output/2)),axis=(0,1,2) )

        for bz in range(blocks[0]):
            for by in range(blocks[1]):
                for bx in range(blocks[2]):
                    print('Block %d %d %s' % (bz,by,bx) )

                    bx_shift = bx * patch_output
                    by_shift = by * patch_output
                    bz_shift = bz * patch_output

                    # Get start stop
                    istart = bx_shift
                    jstart = by_shift
                    kstart = bz_shift

                    istop = istart + patch_output
                    jstop = jstart + patch_output
                    kstop = kstart + patch_output

                    #Grab the block
                    image_block= image[kstart:kstop,jstart:jstop,istart:istop,:]
                    weight_block = weight[kstart:kstop, jstart:jstop, istart:istop]
                    velocity_block = velocity[kstart:kstop, jstart:jstop, istart:istop, :]

                    # Get the block
                    image_block = np.expand_dims(image_block, 0)
                    weight_block = np.expand_dims(weight_block, 0)
                    velocity_block = np.expand_dims(velocity_block, 0)

                    # Get a prediction
                    predict = unwrap_model.predict(image_block)
                    #block_loss = unwrap_model.evaluate(x=[image_block,weight_block], y=velocity_block)
                    #print(block_loss)

                    # Add to image
                    output[kstart:kstop,jstart:jstop,istart:istop,:] = np.squeeze(predict)

        # Roll back
        if bs:
            print('Roll Back to Correct Coordinates')
            image = np.roll( image, (-int(patch_output / 2), -int(patch_output / 2), -int(patch_output / 2)),axis=(0,1,2) )
            output = np.roll(output, (-int(patch_output / 2), -int(patch_output / 2), -int(patch_output / 2)),axis=(0,1,2) )

        # Average 2 block shifts
        block_average += 0.5*output

    # Export
    print('Export 4 point')
    out_name = os.path.join(output_directory,os.path.basename(case_name))

    try:
        os.remove(out_name)
    except OSError:
        pass

    with h5py.File(out_name, 'w') as hf:
        hf.create_dataset("VEST", data=np.squeeze(block_average))
        hf.create_dataset("VACT", data=np.squeeze(velocity))
        hf.create_dataset("IMAGE", data=np.squeeze(image))







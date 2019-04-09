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

#pretrained = 'B:/ThreePoint/Graph/2019-04-01-22-57/weights.090-0.04264'
pretrained = 'B:/ThreePoint/weights.200-0.12259'

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

# Load case
image,weight,velocity = load_convert_4point('B:/ThreePoint/DataConvert/Images_00131.h5')

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
out_name = 'Infered.h5'  # os.path.join('B:\DeepTrajectory\RECONS', 'Errors.h5')
try:
    os.remove(out_name)
except OSError:
    pass

with h5py.File(out_name, 'w') as hf:
    hf.create_dataset("VEST", data=np.squeeze(block_average))
    hf.create_dataset("VACT", data=np.squeeze(velocity))
    hf.create_dataset("IMAGE", data=np.squeeze(image))

quit()

# Export
print('Export Images')
out_name = 'Predict.h5'  # os.path.join('B:\DeepTrajectory\RECONS', 'Errors.h5')
try:
    os.remove(out_name)
except OSError:
    pass

with h5py.File(out_name, 'w') as hf:
    hf.create_dataset("PREDICTION", data=np.squeeze(output))
    hf.create_dataset("PREDICTION_angle", data=np.squeeze(np.angle(output_complex)))
    hf.create_dataset("PREDICTION_mag", data=np.squeeze(np.abs(output_complex)))

    hf.create_dataset("IMAGE", data=np.squeeze(flow_case))
    hf.create_dataset("IMAGE_angle", data=np.squeeze(np.angle(im_complex)))
    hf.create_dataset("IMAGE_mag", data=np.squeeze(np.abs(im_complex)))







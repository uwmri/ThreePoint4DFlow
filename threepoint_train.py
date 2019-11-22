from threepoint_network import *
from threepoint_io import *
from flow_tools import *

# Global Variables
training_data_dir = 'E:/3POINT/DataConvert2'
patch_size = 64 if unet else 32
batch_size = 8 if unet else 8
CROP_TYPE = 'valid'
data_check = False

# For restarting the training
restart_training = True
pretrained = 'E:/3POINT/Weights_2019_10_21/weights.200-0.00078'
initial_epoch=300 if restart_training else 0

def grab_patch(flow_cases=None, block_index=None):

    # Randomly pull from images
    xidx = int(block_index[2] - int(patch_size / 2))
    yidx = int(block_index[1] - int(patch_size / 2))
    zidx = int(block_index[0] - int(patch_size / 2))

    # Grab that block
    block =np.copy(flow_cases[zidx:zidx + patch_size, yidx:yidx + patch_size, xidx:xidx + patch_size,:])

    # Permute
    label = block[:,:,:,6:8]
    image = block[:,:,:,0:6]

    # Residual Mode
    label -= block[:,:,:,0:2]

    return image, label


def grab_velocity_patch(flow_cases=None, velocity=None, weights=None, block_index=None):

    # Randomly pull from images
    xidx = int(block_index[2] - int(patch_size / 2))
    yidx = int(block_index[1] - int(patch_size / 2))
    zidx = int(block_index[0] - int(patch_size / 2))

    # Grab that block
    image =np.copy(flow_cases[zidx:zidx + patch_size, yidx:yidx + patch_size, xidx:xidx + patch_size,0:6])

    # Permute
    label = np.copy(velocity[zidx:zidx + patch_size, yidx:yidx + patch_size, xidx:xidx + patch_size,:])
    bweights = np.copy(weights[zidx:zidx + patch_size, yidx:yidx + patch_size, xidx:xidx + patch_size])

    return image, bweights, label


# Create a generator
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, batch_size=32, N=320, folder_names=None, shuffle=True):

        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = N
        self.total_cases = len(folder_names)
        self.folder_names = folder_names
        self.current_names = None
        self.flow_case = None
        self.block_idx = None
        self.verbose = False
        self.on_epoch_end()
        self.case_count = 0
        self.velocity=None
        self.mrflow =  MRI_4DFlow()
        self.weights = None

        print('batch_size = '+str(batch_size))

    def __len__(self):

        'Denotes the number of batches per epoch'
        return int(np.floor(self.N * self.total_cases / self.batch_size))

    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):

        self.case_count = 0
        self.current_names = self.folder_names
        np.random.shuffle(self.current_names)


    def __data_generation(self, index):

        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 6), dtype=np.float32)
        Y = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 3), dtype=np.float32)
        W = np.ones((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)

        #print('Index = %d' % index)

        #Reload the case if needed
        if ( (index*self.batch_size) % self.N ) == 0 or self.flow_case is None:
            rd = int(self.case_count/self.N) % len(self.current_names)
            print('Load %s ' % (self.current_names[rd],))
            self.flow_case,self.weights,self.velocity = load_convert_4point(filename=self.current_names[rd])

        # Generate data
        for count in range(self.batch_size):

            patch_hsize = int(patch_size / 2)

            # Grab actual block coordinates
            k_rand = np.random.randint(low=patch_hsize, high=(self.flow_case.shape[0] - patch_hsize))
            j_rand = np.random.randint(low=patch_hsize, high=(self.flow_case.shape[1] - patch_hsize))
            i_rand = np.random.randint(low=patch_hsize, high=(self.flow_case.shape[2] - patch_hsize))

            # Load Example
            image, weights, label = grab_velocity_patch(self.flow_case,self.velocity,self.weights, block_index=(k_rand,j_rand,i_rand))

            # print("Index = " + str(self.block_idx[i]) + " , Block index = " + str(block_index3))

            # Augment by flipping
            if np.random.choice([True, False]):
                np.flip(image, axis=0)
                np.flip(label, axis=0)
                np.flip(weights, axis=0)

            if np.random.choice([True, False]):
                np.flip(image, axis=1)
                np.flip(label, axis=1)
                np.flip(weights, axis=1)

            if np.random.choice([True, False]):
                np.flip(image, axis=2)
                np.flip(label, axis=2)
                np.flip(weights, axis=2)

            # Store sample
            X[count, :, :, :, :] = image
            Y[count, :, :, :, :] = np.squeeze(label)
            W[count, :, :, :, 0] = weights

        self.case_count += self.batch_size

        return [X,W],Y


all_names = generate_folder_list(training_data_dir)

if data_check:
    for case_name in all_names:
        data = load_raw_4point(case_name)
        plt.clf()
        Im = data[:,:,:,0] + 1j*data[:,:,:,1]
        plt.imshow(np.abs(np.squeeze(Im[:,:,160])))
        plt.title(case_name)
        plt.show()


# Seperate into train/eval
Ntrain = int(0.8*len(all_names))
Nval = int(0.1*len(all_names))
Tnames = all_names[:Ntrain]
Vnames = all_names[Ntrain:(Ntrain+Nval)]
Testnames = all_names[(Ntrain+Nval):]

print('Training on %d cases' % (len(Tnames),))
print('Validate on %d cases' % (len(Vnames),))
print('Test on on %d cases' % (len(Testnames),))

# Weighted MSE
#input_weights = keras.Input(shape=(None, None, None, 1), dtype='float32')
#wmse = partial(weighted_mse, weights=input_weights)
#wmse.__name__ = 'weighted_mse'
wmse = weighted_mse

# Build the Network
if not unet:
    unwrap_model = build_keras_model()
else:
    unwrap_model = unet_model_3d(input_shape=(patch_size,patch_size,patch_size,6),
                                 pool_size=(2, 2, 2),
                                 n_labels=3,
                                 initial_learning_rate=1e-4,
                                 deconvolution=True,
                                 depth=3,
                                 n_base_filters=30,
                                 include_label_wise_dice_coefficients=False,
                                 metrics=None,
                                 batch_normalization=True,
                                 activation_name="relu")

if restart_training:
    unwrap_model.load_weights(pretrained)

# Get model summary
unwrap_model.summary(line_length=200)

# Get a data generator
N = batch_size*int(320**3/patch_size**3/batch_size) #blocks per image
print('Examples per volume = %d' % (N,))
Tdatagen = DataGenerator(batch_size=batch_size, N=N, folder_names=Tnames, shuffle=False)
Vdatagen = DataGenerator(batch_size=batch_size, N=N, folder_names=Vnames, shuffle=False)

# Tensorflow
import datetime
now = datetime.datetime.now()
log_dir = './Graph/' + now.strftime("%Y-%m-%d-%H-%M")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)

mc = keras.callbacks.ModelCheckpoint('weights.{epoch:03d}-{val_loss:.5f}', save_weights_only=False, period=1)


# Fit
history = unwrap_model.fit_generator(Tdatagen, steps_per_epoch=len(Tdatagen), validation_data=Vdatagen,
                           validation_steps=len(Vdatagen), initial_epoch=initial_epoch,
                                     epochs=400, callbacks=[tbCallBack,mc], verbose=1, shuffle=False)

# Get the dictionary containing each metric and the loss for each epoch
import json
history_dict = history.history
json.dump(history_dict, open('history.json', 'w'))

print('Export Model')
out_name = os.path.join( log_dir, 'ThreePoint.h5' ) # os.path.join('B:\DeepTrajectory\RECONS', 'Errors.h5')
try:
    os.remove(out_name)
except OSError:
    pass
unwrap_model.save(out_name)



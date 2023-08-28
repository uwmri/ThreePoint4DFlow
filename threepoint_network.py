import keras
from functools import partial
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                          MaxPooling2D, Concatenate, UpSampling2D,
                          Conv3D, Conv3DTranspose, MaxPooling3D,
                          UpSampling3D,BatchNormalization,Activation,Deconvolution3D)
from keras import optimizers as opt

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


patch_size = 64
CROP_TYPE = 'valid'
Venc_min = 1
Venc_max = 2
act_function = 'relu'
outer_layers = 15
initial_layers = 12
layer_growth = 0.25
unet = True

def build_keras_model():
    ksize = (3, 3, 3)

    # Build the network
    act_function = 'relu'
    outer_layers = 15
    initial_layers = 12
    layer_growth = 0.25

    # Input is (3D Patch x 6 channels)
    input = keras.Input(shape=(None, None, None, 6))
    #input = keras.Input(shape=(patch_size, patch_size, patch_size, 6), dtype='float32')

    # Frontend Convolution
    shortcut = keras.layers.Conv3D(initial_layers,
                                   kernel_size=ksize,
                                   strides=(1, 1, 1),
                                   padding=CROP_TYPE,
                                   activation=act_function,
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   use_bias=True
                                   )(input)
    x = keras.layers.BatchNormalization()(shortcut)

    for i in range(outer_layers):

        # Bottleneck
        if i > 0:
            # Convolutional Block
            x = keras.layers.Conv3D(initial_layers,
                                    kernel_size=(1, 1, 1),
                                    strides=(1, 1, 1),
                                    padding=CROP_TYPE,
                                    activation=act_function,
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    use_bias=True
                                    )(shortcut)
        else:
            x = shortcut

        # Convolutional Block
        x = keras.layers.Conv3D(initial_layers,
                                kernel_size=ksize,
                                strides=(1, 1, 1),
                                padding=CROP_TYPE,
                                activation=act_function,
                                kernel_regularizer=None,
                                bias_regularizer=None,
                                use_bias=True
                                )(x)
        x = keras.layers.BatchNormalization()(x)

        # Crop the shortcut
        if CROP_TYPE == 'valid':
            shortcut = keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(shortcut)

        # Merge x to the shortcut
        shortcut = keras.layers.Concatenate(axis=4)([x, shortcut])

        initial_layers += int(initial_layers * layer_growth)

    # Wrap images
    x = keras.layers.Conv3D(initial_layers - 10, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
                                      activation=act_function)(x)
    output = keras.layers.Conv3D(3, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                      activation='linear')(x)


    # Weighted MSE
    input_weights = keras.Input(shape=(None, None, None, 1), dtype='float32')
    wmse = partial(weighted_mse, weights=input_weights)
    wmse.__name__ = 'weighted_mse'

    # Compile the model
    unwrap_model = keras.Model(inputs=[input,input_weights], outputs=output)
    unwrap_model.summary(line_length=200)

    # Model
    print('Compile Model')
    opt = keras.optimizers.Adam(lr=1e-4)
    unwrap_model.compile(optimizer=opt, loss=cropped_mse, metrics=[])

    return unwrap_model

# Build a custom loss function
def cropped_mse(y_true, y_pred):
    if CROP_TYPE == 'valid':
        y_true = keras.layers.Cropping3D( cropping=((outer_layers+1, outer_layers+1), (outer_layers+1,outer_layers+1), (outer_layers+1,outer_layers+1)))(y_true)
    return K.mean(K.square(y_pred - y_true), axis=-1)


def weighted_mse(y_true, y_pred, weights):
    if unet == False:
        if CROP_TYPE == 'valid':
            y_true = keras.layers.Cropping3D(cropping=(
            (outer_layers + 1, outer_layers + 1), (outer_layers + 1, outer_layers + 1),
            (outer_layers + 1, outer_layers + 1)))(y_true)
    return K.mean(K.square(weights) * K.square(y_pred - y_true), axis=-1)

def weighted_mad(y_true, y_pred, weights):
    if CROP_TYPE == 'valid':
        y_true = keras.layers.Cropping3D( cropping=((outer_layers+1, outer_layers+1), (outer_layers+1,outer_layers+1), (outer_layers+1,outer_layers+1)))(y_true)
    return K.mean(K.square(weights)*K.abs(y_pred - y_true), axis=-1)

def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)


def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def create_unet_model2D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5, 5),
                        deconvolution_kernel_size=(5, 5),
                        pool_size=(2, 2),
                        strides=(2, 2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001):
    """
    Create a 2D Unet model
    Example
    -------
    unet_model = create_Unet_model2D( (100,100,1), 1, 4)
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels

    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2 ** (layers[i])

        if i == 0:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same')(inputs)
        else:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same')(pool)

        encoding_convolution_layers.append(Conv2D(filters=number_of_filters,
                                                  kernel_size=convolution_kernel_size,
                                                  activation='relu',
                                                  padding='same')(conv))

        if i < len(layers) - 1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers) - 1]
    for i in range(1, len(layers)):
        number_of_filters = lowest_resolution * 2 ** (len(layers) - layers[i] - 1)
        tmp_deconv = Conv2DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)
        tmp_deconv = UpSampling2D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=3)([tmp_deconv, encoding_convolution_layers[len(layers) - i - 1]])

        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                         activation='relu', padding='same')(outputs)
        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                         activation='relu', padding='same')(outputs)

    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1, 1),
                             activation='sigmoid')(outputs)
        else:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1, 1),
                             activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error,
                               optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy',
                               optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode == 'regression':
        outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1, 1),
                         activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')

    unet_model.summary(line_length=200)

    return unet_model




def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=-1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    outputs = Activation('linear')(final_convolution)

    '''
    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
    '''

    # Weighted MSE
    input_weights = keras.Input(shape=(None, None, None, 1), dtype='float32')
    wmse = partial(weighted_mse, weights=input_weights)
    wmse.__name__ = 'weighted_mse'

    model = Model(inputs=[inputs, input_weights], outputs=outputs)
    model.compile(loss=wmse, optimizer=opt.Adam(lr=initial_learning_rate))
    #unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    #model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)

    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)



def create_unet_model3D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5, 5, 5),
                        deconvolution_kernel_size=(5, 5, 5),
                        pool_size=(2, 2, 2),
                        strides=(2, 2, 2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001,
                        bn=True):
    """
    Create a 3D Unet model
    Example
    -------
    unet_model = create_unet_model3D( (128,128,128,1), 1, 4)
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels

    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2 ** (layers[i])

        if i == 0:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same')(inputs)
            conv = keras.layers.BatchNormalization()(conv) if bn else conv

        else:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same')(pool)
            conv = keras.layers.BatchNormalization()(conv) if bn else conv

        # Convolve again
        conv = Conv3D(filters=number_of_filters,
               kernel_size=convolution_kernel_size,
               activation='relu',
               padding='same')(conv)

        encoding_convolution_layers.append(keras.layers.BatchNormalization()(conv) if bn else conv)

        if i < len(layers) - 1:
            pool = MaxPooling3D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers) - 1]
    for i in range(1, len(layers)):
        number_of_filters = lowest_resolution * 2 ** (len(layers) - layers[i] - 1)
        tmp_deconv = Conv3DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)
        tmp_deconv = UpSampling3D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=4)([tmp_deconv, encoding_convolution_layers[len(layers) - i - 1]])

        outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                         activation='relu', padding='same')(outputs)
        outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                         activation='relu', padding='same')(outputs)



    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1, 1, 1),
                             activation='sigmoid')(outputs)
        else:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1, 1, 1),
                             activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error,
                               optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy',
                               optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode == 'regression':
        outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1, 1, 1),
                         activation=output_activation)(outputs)

        # Weighted MSE
        input_weights = keras.Input(shape=(None, None, None, 1), dtype='float32')
        wmse = partial(weighted_mse, weights=input_weights)
        wmse.__name__ = 'weighted_mse'

        unet_model = Model(inputs=[inputs,input_weights], outputs=outputs)
        #unet_model.compile(loss=wmse, optimizer=opt.Adam(lr=init_lr))
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))

    else:
        raise ValueError('mode must be either `classification` or `regression`')

    return unet_model
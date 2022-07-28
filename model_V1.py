import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Conv2DTranspose
from tensorflow.python.keras.layers import Layer, InputSpec
import os
import numpy as np
import random
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def CAE(input_shape=(256, 256, 3), filters=[16, 32, 64, 128, 64], set_seed=[False,87], summary=False):

    if set_seed[0] is True : #fixed the inital random seeds
        print("setting tf,np random seed = " + str(set_seed[1]))
        set_global_determinism(seed=set_seed[1])
        set_seeds(seed=set_seed[1])

    if input_shape[0] % 8 == 0: 
        pad3 = 'same'
    else:
        pad3 = 'valid'
    # build the convolutional autoencoder by 4 CNN layers.
    inp = Input(shape=input_shape,name="original_img")
    x = Conv2D(filters=filters[0],kernel_size=3,strides=2,activation="relu",padding="same",name="Conv_1")(inp)    
    x = Conv2D(filters=filters[1],kernel_size=3, strides=2,activation="relu",padding="same",name="Conv_2")(x)
    x = Conv2D(filters=filters[2], kernel_size=3, strides=2,activation="relu",padding="same",name="Conv_3")(x)
    x = Conv2D(filters=filters[3], kernel_size=3, strides=2,activation="relu",padding=pad3,name="Conv_4")(x)
    shape_before_flattening = K.int_shape(x)
    x = Flatten(input_shape=shape_before_flattening[1:],name="Flatten")(x)
    encoder_output = Dense(filters[4], activation="linear", name="encoded")(x)
    encoder = Model(inp, encoder_output, name="encoder")

    #decoder model
    decoder_input = encoder_output
    x = Dense(shape_before_flattening[1:][0]*shape_before_flattening[1:][1]*shape_before_flattening[1:][2],activation="relu")(decoder_input)
    x = Reshape((shape_before_flattening[1:]))(x)
    x = Conv2DTranspose(filters[2], 3, strides=2,activation="relu",padding=pad3,name="ConvT_1")(x)
    x = Conv2DTranspose(filters[1], 3, strides=2,activation="relu",padding="same",name="ConvT_2")(x)
    x = Conv2DTranspose(filters[0], 3, strides=2,activation="relu",padding="same",name="ConvT_3")(x)
    decoder_output = Conv2DTranspose(input_shape[2], 3, strides=2,padding="same",name="decoded",activation="linear")(x)
    autoencoder = Model(inputs=inp, outputs=decoder_output, name="autoencoder")
    #decoder model
    decoder_input = Input(shape=(filters[4],), name="encoded_img")
    # retrieve the last layer of the autoencoder model 
    decoder_layer6 = autoencoder.layers[-1]
    decoder_layer5 = autoencoder.layers[-2]
    decoder_layer4 = autoencoder.layers[-3]
    decoder_layer3 = autoencoder.layers[-4]
    decoder_layer2 = autoencoder.layers[-5]
    decoder_layer1 = autoencoder.layers[-6]
    decoder_output=decoder_layer6(decoder_layer5(decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(decoder_input))))))
    decoder = Model(inputs=decoder_input,
                outputs=decoder_output,
                name="decoder")
    if summary:
        autoencoder.summary()
        encoder.summary()
        decoder.summary()
    
    return encoder,decoder,autoencoder

#### clustering layers
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    ###building model
    encoder,decoder,autoencoder = CAE(input_shape=(128, 128, 3), set_seed=[False,42],filters=[12, 24, 36, 48, 64],summary=False)
    clustering_layer = ClusteringLayer(6, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam')
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
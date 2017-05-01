import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import InputLayer, ReshapeLayer
from lasagne.layers import batch_norm
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify
from matplotlib import pyplot as plt
from plotting import test_and_plot

class autoencoder(Model):
    def __init__(self):
           pass

    def convlayer(self, image_input, filter_quantity = 32,code_size = 100,filter_size = 3,
                      pool_factor = 2):

        self.code_size = code_size
        self.filter_quantity = filter_quantity
        self.image_input = image_input
        layer={}
        layer['input'] = InputLayer((None, 3, 64, 64), image_input)
        layer['conv1-1'] = Conv2DLayer(layer['input'],
                    num_filters = filter_quantity * pool_factor,
                    filter_size = filter_size,
                    nonlinearity = rectify,
                    pad='same')
        layer['batch_norm1-1'] = batch_norm(layer['conv1-1'])
        layer['conv1-2'] = Conv2DLayer(layer['batch_norm1-1'],
                    num_filters = filter_quantity * pool_factor,
                    filter_size = filter_size,
                    nonlinearity = rectify,
                    pad='same')
        layer['batch_norm1-2'] = batch_norm(layer['conv1-2'])
        layer['pool1'] = Pool2DLayer(layer['batch_norm1-2'], pool_size = pool_factor)
        layer['conv2-1'] = Conv2DLayer(layer['pool1'],
                    num_filters = filter_quantity * pool_factor**2,
                    filter_size = filter_size,
                    nonlinearity = rectify,
                    pad='same')
        layer['batch_norm2-1'] = batch_norm(layer['conv2-1'])
        layer['conv2-2'] = Conv2DLayer(layer['batch_norm2-1'],
                    num_filters = filter_quantity * pool_factor**2,
                    filter_size = filter_size,
                    nonlinearity = rectify,
                    pad='same')
        layer['batch_norm2-2'] = batch_norm(layer['conv2-2'])
        layer['pool2'] = Pool2DLayer(layer['batch_norm2-2'], pool_size = pool_factor)
        layer['reshape_enc'] = ReshapeLayer(layer['pool2'],shape=([0],-1))
        n_units_before_dense = layer['reshape_enc'].output_shape[1]
        layer['code_dense'] = DenseLayer(layer['reshape_enc'], num_units = code_size)
        layer['dense_up'] = DenseLayer(layer['code_dense'], num_units = n_units_before_dense)
        layer['reshape_dec'] = ReshapeLayer(layer['dense_up'],shape=([0],-1,int(filter_quantity/2), int(filter_quantity/2) ))
        layer['upscale1'] = Upscale2DLayer(layer['reshape_dec'], scale_factor = pool_factor)
        layer['conv1-1_upscaling'] = Conv2DLayer(layer['upscale1'],
                    num_filters = filter_quantity * pool_factor,
                    filter_size = filter_size,
                    nonlinearity = rectify,
                    pad='same')
        layer['batch_norm1-1_upscaling'] = batch_norm(layer['conv1-1_upscaling'])
        layer['conv1-2_upscaling'] = Conv2DLayer(layer['batch_norm1-1_upscaling'],
                    num_filters = filter_quantity * pool_factor,
                    filter_size = filter_size,
                    nonlinearity = rectify,
                    pad='same')
        layer['batch_norm1-2_upscaling'] = batch_norm(layer['conv1-2_upscaling'])
        layer['upscale2'] = Upscale2DLayer(layer['batch_norm1-2_upscaling'], scale_factor = pool_factor)
        layer['conv2-1_upscaling'] = Conv2DLayer(layer['upscale2'],
                    num_filters = filter_quantity * pool_factor**2,
                    filter_size = filter_size,
                    nonlinearity = rectify,
                    pad='same')
        layer['batch_norm2-1_upscaling'] = batch_norm(layer['conv2-1_upscaling'])
        layer['conv2-2_upscaling'] = Conv2DLayer(layer['batch_norm2-1_upscaling'],
                    num_filters = filter_quantity * pool_factor**2,
                    filter_size = filter_size,
                    nonlinearity = rectify,
                    pad='same')
        layer['batch_norm2-2_upscaling'] = batch_norm(layer['conv2-2_upscaling'])
        layer['last_layer'] = Conv2DLayer(layer['batch_norm2-2_upscaling'],
                num_filters = 3,
                filter_size = 1,
                pad='same')
        self.layer = layer['last_layer']


class model(object):
    def __init__(self):
        pass
    def convlayer(self):
        raise NotImplementedError
    def layerfunctions(self, learning_rate = 0.001):

        image_input = self.image_input
        target_var = T.tensor4('target')

        pred_img = lasagne.layers.get_output(self.layer)
        loss = self.loss_function(pred_img, target_var)
        params = lasagne.layers.get_all_params(self.layer, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate = learning_rate)
        training_function = theano.function([image_input,target_var], loss, updates = updates,
                                    allow_input_downcast=True)
        self.training_function = training_function

        valid_pred_imgs = lasagne.layers.get_output(self.layer,deterministic=True)
        valid_loss = self.loss_function(valid_pred_imgs, target_var)
        validation_function = theano.function([image_input, target_var], valid_loss, allow_input_downcast=True)
        self.validation_function = validation_function
        self.get_imgs = theano.function([image_input], lasagne.layers.get_output(self.layer,deterministic = True),
                          allow_input_downcast=True)
    def loss_function(self, prediction, target):
        return T.mean(lasagne.objectives.squared_error(prediction, target))


    def batch_extraction(self, batch):
        inputs, targets, caps = batch

        inputs = np.transpose(inputs, (0,3,1,2))
        targets = np.transpose(targets, (0,3,1,2))

        return inputs, targets, caps

    def plotting_result(self, batch, title= '', subset=15):

        if np.shape(batch[0])[3]==3:
            self.test_and_plot(self.batch_extraction(batch), title, subset)
        else :
            self.test_and_plot(batch, title, subset)
if __name__ == '__main__':
    image_input = T.tensor4('images')
    ae, last_layer =autoencoder().convlayer(image_input)
    layers = lasagne.layers.get_all_layers(ae[last_layer])
    layers2 = lasagne.layers.get_all_layers(ae)
    for l in layers2 :
        print (l, ae[l].output_shape)
import os
import sys
import time
import numpy as np
import theano.tensor as T
import lasagne
from iterator import Iterator
import ae_nets as autoencoder


filename = 'best_model'
learning_rate = 0.01
weight_decay = 0
batch_size = 512
extract_center = True
load_caption = True
n_filters = 32                      
nb_var = 750
filter_size = 3
pool_factor = 2

training_iteration = Iterator(which_set='train', batch_size = batch_size,extract_center = extract_center, load_caption = load_caption)
valid_iteration = Iterator(which_set='valid', batch_size = batch_size,extract_center = extract_center, load_caption = load_caption)
test_iteration = None
training_batches = training_iteration.n_batches
validation_batches = valid_iteration.n_batches
n_batches_test = test_iteration.n_batches if test_iteration is not None else 0
savingpath = '/Users/Omar/anaconda/lib/python3.6/site-packages/spyder/utils/site/load_models/'
loadingpath = '/Users/Omar/anaconda/lib/python3.6/site-packages/spyder/utils/site/load_models/'
weight_path = loadingpath
weight_path = os.path.join(WEIGHTS_PATH, filename, 'best_model.npz')
model_image_input = T.tensor4('image input')
model_captions_input = T.matrix('captions input')
model_target_input = T.tensor4('target input')
model = model.autoencoder()
model.convlayer(image_input = model_image_input,n_filters = n_filters,nb_var = nb_var,filter_size = filter_size,pool_factor = pool_factor)
with np.load(weight_path) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]   
lasagne.layers.set_all_param_values(model.net, param_values)
model.layerfunctions(learning_rate= learning_rate)

def test_plot(model, test_iteration, plot_results = True, max_batch = 5):
    for i, test_batch in enumerate(test_iteration):
        if i>= max_batch:
            break
        test_batch = model.extract_batch(test_batch)
        inputs_test, targets_test,caps_test = test_batch
        if plot_results: 
            model.compute_and_plot_results((inputs_test[100:115], targets_test[100:115],caps_test[100:115]))

test_plot(model, training_iteration,'Model')





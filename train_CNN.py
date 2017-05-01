import os
import sys
import time
from distutils.dir_util import copy_tree
import numpy as np
import theano.tensor as T
from theano import config
import lasagne
import ae_nets as autoencoder
from iterator import Iterator

learning_rate = 0.01
weight_decay = 0
num_epochs = 100
max_patience = 100
batch_size = 512
extract_center = True
load_caption = True
filter_quantity = 32                      
nb_var = 750
filter_size = 3
pool_factor = 2

filename_ = 'best_model'
path_to_save = 'save_models' 
path_to_load = 'load_models'
savepath=os.path.join(sys.path[1],path_to_save, filename_)
loadpath=os.path.join(sys.path[1],path_to_load, filename_)
training_iteration = Iterator(which_set='train', batch_size = batch_size, extract_center = extract_center, load_caption = load_caption)
validation_iteration = Iterator(which_set='valid', batch_size = batch_size,extract_center = extract_center, load_caption = load_caption)
test_iter = None
n_batches_train = training_iteration.n_batches
n_batches_valid = validation_iteration.n_batches
n_batches_test = test_iter.n_batches if test_iter is not None else 0
ae_input_var = T.tensor4('input img bx64x64x3')
ae_captions_var = T.matrix('captions var')
model = autoencoder.AE_contour2center()
model.convnet(input_var=ae_input_var,filter_quantity = filter_quantity,nb_var = nb_var,filter_size = filter_size,pool_factor = pool_factor)
model.layerfunction(learning_rate=learning_rate)
plot_results_train = True
plot_results_valid = True
n_batches_train = 1
n_batches_valid = 1
err_train = []
validation_error = []
best_err_val = 0
patience = 0
reset_best_results = True

for epoch in range(num_epochs):    
    start_time = time.time()
    cost_train_epoch = 0            
    for i, training_batch in enumerate(training_iteration):
        if n_batches_train > 0 and i> n_batches_train:
            break    
        training_batch = model.batch_extraction(training_batch) 
        inputs_train, targets_train, caps_train = training_batch
        cost_training_batch =  model.train_fn(inputs_train, targets_train)
        cost_train_epoch += cost_training_batch
    err_train += [cost_train_epoch/n_batches_train]
    validation_cost_epoch = 0
    for i, valid_batch in enumerate(validation_iteration):   
        if n_batches_valid > 0 and i> n_batches_valid:
            break
        valid_batch = model.batch_extraction(valid_batch)
        inputs_valid, targets_valid, caps_valid = valid_batch
        validation_cost_batch = model.valid_fn(inputs_valid, targets_valid)
        validation_cost_epoch += validation_cost_batch
    validation_error += [validation_cost_epoch/n_batches_valid]
    
    with open(os.path.join(savepath, "model.log"), "a") as f:
        f.write(out_str + "\n")
    if epoch == 0 and reset_best_results:
        best_validation_error = validation_error[epoch]
    elif epoch > 1 and validation_error[epoch] < best_validation_error:
        best_validation_error = validation_error[epoch]
        patience = 0
        np.savez(os.path.join(savepath, 'best_model.npz'),*lasagne.layers.get_all_param_values(model.net))
        np.savez(os.path.join(savepath , "best_error.npz"),err_train=err_train, validation_error=validation_error)
        np.savez(os.path.join(savepath, 'last_model.npz'),*lasagne.layers.get_all_param_values(model.net))
        np.savez(os.path.join(savepath , "last_error.npz"),err_train=err_train, validation_error=validation_error)
    else:
        patience += 1
        np.savez(os.path.join(savepath, 'last_model.npz'), *lasagne.layers.get_all_param_values(model.net))
        np.savez(os.path.join(savepath , "ae_errors_last.npz"), err_train=err_train, validation_error=validation_error)
    if patience == max_patience or epoch == num_epochs-1:
        if savepath != loadpath:
            copy_tree(savepath, loadpath)
        break






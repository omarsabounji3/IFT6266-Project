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

def test_and_plot(self, batch, title, subset=15):

        inputs, targets, caps = batch
        if np.shape(inputs[0])==subset:
            indices = [i for i in range(subset)]
        else:
            indices = np.random.randint(inputs.shape[0], size=subset)

        plt.figure(dpi=subset*15)
        for i in range(subset):

            idx = indices[i]

            fake_imgs = self.get_imgs(inputs[idx:idx+1])
            target_center = np.transpose(targets[idx], (1,2,0))
            fake_center = np.transpose(fake_imgs[0], (1,2,0))
            input_contour = np.transpose(inputs[idx],(1,2,0))
            full_img=np.zeros((64,64,3))
            np.copyto(full_img,input_contour)
            full_img[16:48,16:48, :] = target_center
            full_fake_img = np.zeros((64,64,3))
            np.copyto(full_fake_img, input_contour)
            full_fake_img[16:48,16:48, :] = fake_center
            plot_image = np.concatenate((full_img,full_fake_img), axis=0)
            if i==0:
                all_images = plot_image
            else:
                all_images = np.concatenate((all_images, plot_image), axis = 1)

        plt.axis('off')
        plt.imshow(all_images)
        plt.show()
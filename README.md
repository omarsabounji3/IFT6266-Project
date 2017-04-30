## The first method used is a CNN

CNN methods are reputed efficient for this kind of task. So, I chose to implement this method with basically many filters as described below. From an input, which is the image border, many operations where drawn to improve the learning capacity of the model. The loss function used was mean squared error, trained over 100 epochs with early stopping. Stochastic gradient descent was used with Adam method(RMSProp + momentum) and hyper-parameters of 0.01 for the learning rate. . The non-linearities introduced in each layer were RELU. There wasn't any hyper-parameter search was performed. The upsampling used was used to replicate each value in the image as a 2×2 patch.

<img width="287" alt="capture d ecran 2017-04-30 a 15 02 50" src="https://cloud.githubusercontent.com/assets/18235074/25567190/61749d24-2db6-11e7-8f63-bc5ab73f1fa7.png">


## Methods : 

In the deep learning literature, there are many models able to complete task needed for this project. Among them, there are CNNs(Convolutional neural network), L2 regularization Network, GAN(Generative Adversial Network) and its multiple version. 



(Pathak and al., Context Encoders: Feature Learning by Inpainting, 2016)


## Project description

  The objectives of this blog is to document the methodology and the results of the class project. This project is part of the deep learning course IFT 6266. 

  The purpose of the project is to to generate the middle region of images conditioned on the outside border of the image and a caption describing the image. Here is an ilustration of the concept.

![imagecaptionex](https://cloud.githubusercontent.com/assets/18235074/25565825/f42d7fec-2d9c-11e7-9361-89a9ff9b3dae.png)

  All the images used during this experimentation are taken from the MSCOCO dataset. This dataset contains high resolution images, roughly 500×500 pixels. For the actual project, all images are downsampled to 64×64 and the goal is to complete the middle 32×32 section.

  Many models will be tried to check what are the ones that gives the best results. The evaluation of the performance of those methods are essentially visual because it is quiet hard to evaluate quantitatively their abilities.
  
  The deep learning packages Theano and Lasagne will be used during this experimentation. 
